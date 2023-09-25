//! Provides the top level `BinaryView` for analyzing binaries
use binja_sys::*;

use anyhow::{anyhow, Context, Result};
use log::trace;
use rayon::prelude::*;

use std::borrow::Cow;
use std::convert::TryInto;
use std::ffi::CStr;
use std::ffi::CString;
use std::fmt;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::slice;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use crate::databuffer::DataBuffer;
use crate::filemetadata::FileMetadata;
use crate::function::Function;
use crate::highlevelil::HighLevelILInstruction;
use crate::lowlevelil::LowLevelILInstruction;
use crate::mediumlevelil::MediumLevelILInstruction;
use crate::platform::Platform;
use crate::plugin::Plugins;
use crate::reference::ReferenceSource;
use crate::savesettings::SaveSettings;
use crate::startup::init_plugins;
use crate::stringreference::StringReference;
use crate::symbol::Symbol;
use crate::unsafe_try;
use crate::wrappers::BinjaBinaryView;

/// Top level struct for accessing binary analysis functions
#[derive(Clone)]
pub struct BinaryView {
    /// Handle given by BinjaCore
    handle: Arc<BinjaBinaryView>,
    meta: FileMetadata,
}

unsafe impl Send for BinaryView {}
unsafe impl Sync for BinaryView {}

#[derive(Debug)]
pub enum BinjaError {
    BinaryViewError,
}

impl Drop for BinaryView {
    fn drop(&mut self) {
        // If this is the last active binary view, shut down binary ninja
        if crate::ACTIVE_BINARYVIEWS.fetch_sub(1, Ordering::SeqCst) == 1 {
            trace!("Dropping last active binary view. Shutting down binary ninja");
            unsafe { BNShutdown() }
            timeloop::print!();
        }
    }
}

pub struct BinaryViewBuilder<'a> {
    filename: &'a str,
    base_addr: Option<u64>,
}

impl BinaryViewBuilder<'_> {
    pub fn base_addr(mut self, base_addr: Option<u64>) -> Self {
        self.base_addr = base_addr;
        self
    }

    /// Returns a BinaryView given a filename.
    ///
    /// Note: `update_analysis_and_wait` is automatically called by this function.
    ///
    /// # Examples
    ///
    /// ```
    /// use binja_rs::BinaryView;
    /// let bv = BinaryView::new_from_filename("tests/ls").build().unwrap();
    /// assert_eq!(bv.has_functions(), true);
    /// ```
    pub fn build(self) -> Result<BinaryView> {
        let BinaryViewBuilder {
            filename,
            base_addr,
        } = self;

        // If this is the first binary view, start the profiler
        if crate::ACTIVE_BINARYVIEWS.load(Ordering::SeqCst) == 0 {
            timeloop::start_profiler!();
        }

        timeloop::scoped_timer!(crate::Timer::BinaryView__NewFromFilename);

        if !Path::new(filename).exists() {
            panic!("File not found: {}", filename);
        }

        trace!("env_logger initialized!");

        init_plugins();

        let is_db = !filename.ends_with("bndb");

        // Create the initial binary view to get the available view types
        let view = BinaryView::open(filename)?;

        // Init the logger
        // Go through all found view types (except Raw) for this file and attempt to create that
        // type
        for view_type in view.available_view_types() {
            if view_type.name() == "Raw" {
                continue;
            }

            // Create the view type
            let mut bv = view_type.create(&view)?;

            // If successfully created, update analysis for the view
            let now = Instant::now();

            bv.update_analysis_and_wait();

            // Rebase the view if given a new base address
            if let Some(base_addr) = base_addr {
                unsafe {
                    BNRebase(bv.handle(), base_addr);
                }

                // Create the rebased binary view
                bv = bv.get_view_of_type(&view_type.name().to_string())?;
            }

            if !is_db {
                // Write the database if the file isn't already a db
                bv.create_database()?;
            }

            trace!(
                "Analysis took {}.{} seconds",
                now.elapsed().as_secs(),
                now.elapsed().subsec_nanos()
            );

            // Return the view
            return Ok(bv);
        }

        Err(anyhow!(
            "Failed to find a view type for given file: {}",
            filename
        ))
    }
}

impl BinaryView {
    /// Utility for getting the handle for this `BinaryView`
    pub fn handle(&self) -> *mut BNBinaryView {
        **self.handle
    }

    pub fn new_from_filename(filename: &str) -> BinaryViewBuilder {
        BinaryViewBuilder {
            filename,
            base_addr: None,
        }
    }

    /// Return the `Raw` BinaryViewType for this BinaryView
    pub fn raw_view(&self) -> Result<BinaryView> {
        self.get_view_of_type("Raw")
    }

    /// Return the `Raw` BinaryViewType for this BinaryView
    pub fn get_view_of_type(&self, name: &str) -> Result<BinaryView> {
        let name = CString::new(name).unwrap();
        let handle = unsafe_try!(BNGetFileViewOfType(self.meta.handle(), name.as_ptr()))
            .context("BNGetFileViewOfType failed")?;

        // Add to the current active binary views
        crate::ACTIVE_BINARYVIEWS.fetch_add(1, Ordering::SeqCst);

        Ok(BinaryView {
            handle: Arc::new(BinjaBinaryView::new(handle)),
            meta: self.meta.clone(),
        })
    }

    /// # Example
    ///
    /// ```
    /// use binja_rs::BinaryView;
    /// let bv = BinaryView::new_from_filename("tests/ls").unwrap();
    /// assert_eq!(bv.functions().len(), 192);
    /// ```
    pub fn functions(&self) -> Vec<Function> {
        let mut count = 0;

        unsafe {
            let functions = BNGetAnalysisFunctionList(self.handle(), &mut count);
            let funcs_slice = slice::from_raw_parts(functions, count as usize);
            let result = funcs_slice
                .iter()
                .map(|&f| Function::new(f).unwrap())
                .collect();
            BNFreeFunctionList(functions, count);
            result
        }
    }

    /// Alias for clarification of `get_function_at`
    pub fn get_function_starting_at(&self, addr: u64) -> Result<Option<Function>> {
        self.get_function_at(addr)
    }

    /// Return the function starting at the given `addr`
    pub fn get_function_at(&self, addr: u64) -> Result<Option<Function>> {
        let plat = self
            .platform()
            .ok_or(anyhow!("Can't get_function_at without platform"))?;

        let handle = unsafe { BNGetAnalysisFunction(self.handle(), plat.handle(), addr) };

        if handle == std::ptr::null_mut() {
            return Ok(None);
        }

        Ok(Some(Function::new(handle)?))
    }

    /// # Example
    ///
    /// ```
    /// use binja_rs::BinaryView;
    /// let bv = BinaryView::new_from_filename("tests/ls").unwrap();
    /// let strings = bv.strings(); // Necessary since StringReference holds a Cow<str>
    /// let first_string = strings.iter().nth(0).unwrap();
    /// assert_eq!(first_string.data, "/lib64/ld-linux-x86-64.so.2");
    /// assert_eq!(first_string.length, 27);
    /// assert_eq!(first_string.start, 0x400238);
    /// ```
    pub fn strings(&self) -> Vec<StringReference> {
        timeloop::scoped_timer!(crate::Timer::BinaryView__Strings);

        let mut count = 0;

        unsafe {
            let strings = BNGetStrings(self.handle(), &mut count);
            let strings_slice = slice::from_raw_parts(strings, count as usize);

            let result = strings_slice
                .iter()
                .map(|&s| StringReference::new(s, self.read(s.start, s.length.try_into().unwrap())))
                .collect();

            BNFreeStringReferenceList(strings);
            result
        }
    }

    pub fn update_analysis_and_wait(&self) {
        timeloop::scoped_timer!(crate::Timer::BNUpdateAnalysisAndWait);

        unsafe { BNUpdateAnalysisAndWait(self.handle()) }
    }

    pub fn update_analysis(&self) {
        timeloop::scoped_timer!(crate::Timer::BNUpdateAnalysis);

        unsafe { BNUpdateAnalysis(self.handle()) }
    }

    pub fn entry_point(&self) -> u64 {
        timeloop::scoped_timer!(crate::Timer::BNGetEntryPoint);

        unsafe { BNGetEntryPoint(self.handle()) }
    }

    pub fn has_functions(&self) -> bool {
        timeloop::scoped_timer!(crate::Timer::BNHasFunctions);

        unsafe { BNHasFunctions(self.handle()) }
    }

    /// Retrieve the `filename` of the binary currently being analyzed
    pub fn name(&self) -> PathBuf {
        timeloop::scoped_timer!(crate::Timer::BinaryView__Name);

        self.meta.filename()
    }

    pub fn len(&self) -> u64 {
        timeloop::scoped_timer!(crate::Timer::BNGetViewLength);

        unsafe { BNGetViewLength(self.handle()) }
    }

    pub fn start(&self) -> u64 {
        timeloop::scoped_timer!(crate::Timer::BNGetStartOffset);

        unsafe { BNGetStartOffset(self.handle()) }
    }

    /// # Example
    ///
    /// ```
    /// use binja_rs::BinaryView;
    /// let bv = BinaryView::new_from_filename("tests/ls").unwrap();
    /// assert_eq!(bv.type_name(), "ELF");
    /// ```
    pub fn type_name(&self) -> Cow<str> {
        timeloop::scoped_timer!(crate::Timer::BinaryView__TypeName);

        unsafe {
            let name_ptr = BNGetViewType(self.handle());
            let name = CStr::from_ptr(name_ptr)
                .to_string_lossy()
                .into_owned()
                .into();
            BNFreeString(name_ptr);
            name
        }
    }

    /// Open a BinaryDataView for the given
    fn open(filename: &str) -> Result<BinaryView> {
        timeloop::scoped_timer!(crate::Timer::BinaryView__Open);

        init_plugins();

        let metadata = FileMetadata::from_filename(filename)?;
        let ffi_name = CString::new(filename).unwrap();

        let handle = if filename.ends_with("bndb") {
            // Sanity check the header of the BNDB is correct
            const NEEDLE: &'static str = "SQLite format 3";
            let mut check = [0u8; NEEDLE.len()];
            let mut f = std::fs::File::open(filename)?;
            f.read_exact(&mut check)?;

            // Ensure the header is the needle for the bndb file
            assert!(
                check == NEEDLE.as_bytes(),
                "Binary Ninja database file does not have correct header"
            );

            unsafe_try!(BNOpenExistingDatabase(metadata.handle(), ffi_name.as_ptr()))?
        } else {
            // Add to the current active binary views
            crate::ACTIVE_BINARYVIEWS.fetch_add(1, Ordering::SeqCst);

            unsafe_try!(BNCreateBinaryDataViewFromFilename(
                metadata.handle(),
                ffi_name.as_ptr()
            ))?
        };

        Ok(BinaryView {
            handle: Arc::new(BinjaBinaryView::new(handle)),
            meta: metadata,
        })
    }

    fn read(&self, addr: u64, len: u64) -> DataBuffer {
        unsafe {
            DataBuffer::new_from_handle(BNReadViewBuffer(
                self.handle(),
                addr,
                len.try_into().unwrap(),
            ))
        }
    }

    fn available_view_types(&self) -> Vec<BinaryViewType> {
        timeloop::scoped_timer!(crate::Timer::BinaryView__AvailableViewTypes);

        let mut count = 0;

        unsafe {
            let types = BNGetBinaryViewTypesForData(self.handle(), &mut count);

            let types_slice = slice::from_raw_parts(types, count);

            let result = types_slice
                .iter()
                .map(|&t| BinaryViewType::new(t))
                .collect();

            BNFreeBinaryViewTypeList(types);
            result
        }
    }

    fn platform(&self) -> Option<Platform> {
        timeloop::scoped_timer!(crate::Timer::BinaryView__Platform);

        Platform::new(&self)
    }

    /// Get all LLIL instructions in the binary
    pub fn llil_instructions(&self) -> Vec<LowLevelILInstruction> {
        timeloop::scoped_timer!(crate::Timer::BinaryView__LLILInstructions);

        let mut res = Vec::new();

        let all_instrs: Vec<Vec<LowLevelILInstruction>> = self
            .functions()
            .iter()
            .filter_map(|func| func.llil_instructions().ok())
            .collect();

        for instrs in all_instrs {
            res.extend(instrs);
        }

        res
    }

    /// Get all LLIL instructions in the binary in parallel
    pub fn par_llil_instructions(&self) -> Vec<LowLevelILInstruction> {
        timeloop::scoped_timer!(crate::Timer::BinaryView__ParLLILInstructions);
        let mut res = Vec::new();

        let all_instrs: Vec<Vec<LowLevelILInstruction>> = self
            .functions()
            .par_iter()
            .filter_map(|func| func.llil_instructions().ok())
            .collect();

        for instrs in all_instrs {
            res.extend(instrs);
        }

        res
    }

    /// Get all LLIL expressions in the binary
    pub fn llil_expressions(&self) -> Vec<LowLevelILInstruction> {
        timeloop::scoped_timer!(crate::Timer::BinaryView__LLILExpressions);

        let mut res = Vec::new();

        let all_instrs: Vec<Vec<LowLevelILInstruction>> = self
            .functions()
            .iter()
            .filter_map(|func| func.llil_expressions().ok())
            .collect();

        for instrs in all_instrs {
            res.extend(instrs);
        }

        res
    }

    /// Get all MLIL instructions in the binary
    pub fn mlil_instructions(&self) -> Vec<MediumLevelILInstruction> {
        let mut res = Vec::new();

        let all_instrs: Vec<Vec<MediumLevelILInstruction>> = self
            .functions()
            .iter()
            .filter_map(|func| func.mlil_instructions().ok())
            .collect();

        for instrs in all_instrs {
            res.extend(instrs);
        }

        res
    }

    /// Get all MLIL instructions in the binary
    pub fn par_mlil_instructions(&self) -> Vec<MediumLevelILInstruction> {
        let mut res = Vec::new();

        let all_instrs: Vec<Vec<MediumLevelILInstruction>> = self
            .functions()
            .par_iter()
            .filter_map(|func| func.mlil_instructions().ok())
            .collect();

        for instrs in all_instrs {
            res.extend(instrs);
        }

        res
    }

    /// Return all HLIL instructions in the binary in parallel
    pub fn par_hlil_instructions(&self) -> Vec<HighLevelILInstruction> {
        let mut res = Vec::new();

        let all_instrs: Vec<Vec<HighLevelILInstruction>> = self
            .functions()
            .par_iter()
            .filter_map(|func| func.hlil_instructions().ok())
            .collect();

        for instrs in all_instrs {
            res.extend(instrs);
        }

        res
    }

    /// Return all HLIL instructions in the binary
    pub fn hlil_instructions(&self) -> Vec<HighLevelILInstruction> {
        let mut res = Vec::new();

        let all_instrs: Vec<Vec<HighLevelILInstruction>> = self
            .functions()
            .iter()
            .filter_map(|func| func.hlil_instructions().ok())
            .collect();

        for instrs in all_instrs {
            res.extend(instrs);
        }

        res
    }

    /// Return all HLIL instructions in the binary
    pub fn hlilssa_instructions(&self) -> Vec<HighLevelILInstruction> {
        let mut res = Vec::new();

        let all_instrs: Vec<Vec<HighLevelILInstruction>> = self
            .functions()
            .par_iter()
            .filter_map(|func| func.hlilssa_instructions().ok())
            .collect();

        for instrs in all_instrs {
            res.extend(instrs);
        }

        res
    }

    /// Return all HLIL expressions in the binary in parallel
    pub fn hlilssa_expressions(&self) -> Vec<HighLevelILInstruction> {
        let mut res = Vec::new();

        let all_instrs: Vec<Vec<HighLevelILInstruction>> = self
            .functions()
            .iter()
            .filter_map(|func| func.hlilssa_expressions().ok())
            .collect();

        for instrs in all_instrs {
            res.extend(instrs);
        }

        res
    }

    /// Return all HLIL expressions in the binary in parallel
    pub fn par_hlil_expressions(&self) -> Vec<HighLevelILInstruction> {
        let mut res = Vec::new();

        let all_instrs: Vec<Vec<HighLevelILInstruction>> = self
            .functions()
            .par_iter()
            .filter_map(|func| func.hlil_expressions().ok())
            .collect();

        for instrs in all_instrs {
            res.extend(instrs);
        }

        res
    }

    /// Return all HLIL expressions in the binary in parallel
    pub fn par_hlilssa_expressions(&self) -> Vec<HighLevelILInstruction> {
        let mut res = Vec::new();

        let all_instrs: Vec<Vec<HighLevelILInstruction>> = self
            .functions()
            .par_iter()
            .filter_map(|func| func.hlilssa_expressions().ok())
            .collect();

        for instrs in all_instrs {
            res.extend(instrs);
        }

        res
    }

    /// Return all HLIL expressions in the binary, filtered by the given filter function
    pub fn hlilssa_expressions_filtered(
        &self,
        filter: &(dyn Fn(&BinaryView, &HighLevelILInstruction) -> bool + 'static + Sync),
    ) -> Vec<HighLevelILInstruction> {
        // Initialize the result
        let mut res = Vec::new();

        print!("Getting HLILSSA expressions\n");

        // Gather all the filtered HLILSSA expressions from all functions
        let all_instrs: Vec<Vec<HighLevelILInstruction>> = self
            .functions()
            .iter()
            .filter_map(|func| func.hlilssa_expressions_filtered(&self, filter).ok())
            .collect();

        // Flatten the Vec<Vec<>> to a Vec<>
        for instrs in all_instrs {
            res.extend(instrs);
        }

        // Return the result
        res
    }

    /// Return all HLIL expressions in the binary, filtered by the given filter function
    /// in parallel
    pub fn par_hlil_expressions_filtered(
        &self,
        filter: &(dyn Fn(&BinaryView, &HighLevelILInstruction) -> bool + 'static + Sync),
    ) -> Vec<HighLevelILInstruction> {
        // Initialize the result
        let mut res = Vec::new();

        print!("Getting HLIL expressions\n");

        // Gather all the filtered HLILSSA expressions from all functions in parallel
        let all_instrs: Vec<Vec<HighLevelILInstruction>> = self
            .functions()
            .par_iter()
            .filter_map(|func| func.hlil_expressions_filtered(&self, filter).ok())
            .collect();

        // Flatten the Vec<Vec<>> to a Vec<>
        for instrs in all_instrs {
            res.extend(instrs);
        }

        // Return the result
        res
    }

    /// Return all HLIL expressions in the binary, filtered by the given filter function
    /// in parallel
    pub fn par_hlilssa_expressions_filtered(
        &self,
        filter: &(dyn Fn(&BinaryView, &HighLevelILInstruction) -> bool + 'static + Sync),
    ) -> Vec<HighLevelILInstruction> {
        // Initialize the result
        let mut res = Vec::new();

        print!("Getting HLILSSA expressions\n");

        // Gather all the filtered HLILSSA expressions from all functions in parallel
        let all_instrs: Vec<Vec<HighLevelILInstruction>> = self
            .functions()
            .par_iter()
            .filter_map(|func| func.hlilssa_expressions_filtered(&self, filter).ok())
            .collect();

        // Flatten the Vec<Vec<>> to a Vec<>
        for instrs in all_instrs {
            res.extend(instrs);
        }

        // Return the result
        res
    }

    /// Return all HLIL instructions in the binary, filtered by the given filter function
    pub fn par_hlil_instructions_filtered(
        &self,
        filter: &(dyn Fn(&BinaryView, &HighLevelILInstruction) -> bool + 'static + Sync),
    ) -> Vec<HighLevelILInstruction> {
        // Initialize the result
        let mut res = Vec::new();

        print!("Getting HLILSSA instructions\n");

        // Gather all the filtered HLILSSA expressions from all functions in parallel
        let all_instrs: Vec<Vec<HighLevelILInstruction>> = self
            .functions()
            .par_iter()
            .filter_map(|func| func.hlil_instructions_filtered(&self, filter).ok())
            .collect();

        // Flatten the Vec<Vec<>> to a Vec<>
        for instrs in all_instrs {
            res.extend(instrs);
        }

        // Return the result
        res
    }

    /// Get a list of cross-references or xrefs to the provided virtual address
    pub fn get_code_refs(&self, addr: u64) -> Vec<ReferenceSource> {
        let mut count = 0;

        let mut res = Vec::new();

        unsafe {
            let xrefs = BNGetCodeReferences(self.handle(), addr, &mut count);
            let xrefs_slice = slice::from_raw_parts(xrefs, count as usize);
            for xref in xrefs_slice {
                if let Ok(new_ref) = ReferenceSource::new(*xref) {
                    res.push(new_ref);
                }
            }

            BNFreeCodeReferences(xrefs, count);
        }

        res
    }

    /// Get a list of cross-references or xrefs to the provided virtual address
    pub fn get_callers(&self, addr: u64) -> Vec<ReferenceSource> {
        let mut count = 0;

        let mut res = Vec::new();

        unsafe {
            let xrefs = BNGetCallers(self.handle(), addr, &mut count);
            let xrefs_slice = slice::from_raw_parts(xrefs, count as usize);
            for xref in xrefs_slice {
                if let Ok(new_ref) = ReferenceSource::new(*xref) {
                    res.push(new_ref);
                }
            }

            BNFreeCodeReferences(xrefs, count);
        }

        res
    }

    /// Get a list of `Function` called by the given `ReferenceSource`
    pub fn get_callees(&self, ref_source: &ReferenceSource) -> Result<Vec<Function>> {
        let mut count = 0;

        let mut res = Vec::new();

        unsafe {
            // Convert the given `ReferenceSource` to a `BNReferenceSource` for the
            // binja core call `BNGetCallees`
            let mut bn_ref = ref_source.as_bnreferencesource();

            // Get the starting addresses of functions called by the xref
            let addrs = BNGetCallees(self.handle(), &mut bn_ref, &mut count);

            // Get a slice from the raw buffer returned
            let addrs_slice = slice::from_raw_parts(addrs, count);

            // Get the functions from the starting addresses passed back from binja core
            for addr in addrs_slice {
                if let Some(func) = self.get_function_at(*addr)? {
                    res.push(func);
                }
            }

            // Free the list from core
            BNFreeAddressList(addrs);
        }

        Ok(res)
    }

    /// Attempt to get the symbol at the given `addr`
    pub fn get_symbol_at(&self, addr: u64) -> Result<Symbol> {
        let sym_handle = unsafe_try!(BNGetSymbolByAddress(
            self.handle(),
            addr,
            0 as *const BNNameSpace
        ))?;

        Ok(Symbol::new(sym_handle))
    }

    /// Abort analysis of the currently running analysis
    pub fn abort_analysis(&self) {
        unsafe {
            BNAbortAnalysis(self.handle());
        }
    }

    /// Create a database for this binary view
    pub fn create_database(&self) -> Result<()> {
        unsafe {
            let mut bndb = self.meta.filename();
            bndb.set_extension("bndb");

            // Get a generic save settings to save with
            let settings = SaveSettings::new().context("Failed to create SaveSettings")?;

            // Create the filename for the database
            let ptr = CString::new(bndb.to_str().unwrap()).unwrap();

            // Get the raw BinaryViewType for this BinaryView
            let raw_type = self.raw_view().context("Failed to create raw view")?;

            // Create the database
            let res = BNCreateDatabase(raw_type.handle(), ptr.as_ptr(), settings.handle());
            if !res {
                Err(anyhow!("Failed to create database"))
            } else {
                print!("Saved database to {:?}\n", bndb);
                Ok(())
            }
        }
    }

    /// Save the current view into the database
    pub fn save_auto_snapshot(&self) -> Result<bool> {
        unsafe {
            let settings = SaveSettings::new()?;
            Ok(BNSaveAutoSnapshot(self.handle(), settings.handle()))
        }
    }

    /// Execute the `PDB\Load` command
    pub fn load_pdb(&self) -> Result<()> {
        // Get the current list of loaded plugins
        let plugins = Plugins::get_list();

        for plugin in plugins.iter() {
            // Look only for the `PDB\Load`
            if plugin.name() != "PDB\\Load" {
                continue;
            }

            // Found the PDB Load command
            print!("Found PDB Load\n");

            // Execute the PBD load command
            plugin.execute(&self)?;

            return Ok(());
        }

        Err(anyhow!("Could not find PDB\\Load plugin"))
    }
}

impl fmt::Display for BinaryView {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // write!(f, "<BinaryView: {:?}, start: {:#x}, len: {:#x}>", self.name(), self.start(), self.len())
        write!(f, "<BinaryView: {:?}, len: {:#x}>", self.name(), self.len())
    }
}

impl fmt::Debug for BinaryView {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "<BinaryView: {:?}, start: {:#x}, len: {:#x}>",
            self.name(),
            self.start(),
            self.len()
        )
    }
}

/*
impl From<*mut BNBinaryView> for BinaryView {
    fn from(handle: *mut BNBinaryView) -> Self {
        if handle.is_null() {
            panic!("Cannot create BinaryView from handle.. Null pointer");
        }

        let metadata = unsafe {
            FileMetadata {
                handle: BNGetFileForView(handle)
            }
        };

        BinaryView { handle, metadata }
    }
}
*/

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BinaryViewType {
    handle: *mut BNBinaryViewType,
}

impl BinaryViewType {
    pub fn new(bvt: *mut BNBinaryViewType) -> BinaryViewType {
        BinaryViewType { handle: bvt }
    }

    pub fn name(&self) -> Cow<str> {
        unsafe {
            let name_ptr = BNGetBinaryViewTypeName(self.handle);
            let name = CStr::from_ptr(name_ptr)
                .to_string_lossy()
                .into_owned()
                .into();
            BNFreeString(name_ptr);
            name
        }
    }

    pub fn long_name(&self) -> Cow<str> {
        unsafe {
            let name_ptr = BNGetBinaryViewTypeLongName(self.handle);
            let name = CStr::from_ptr(name_ptr)
                .to_string_lossy()
                .into_owned()
                .into();
            BNFreeString(name_ptr);
            name
        }
    }

    /// Use this BinaryViewType on the given `BinaryView`
    fn create(&self, data: &BinaryView) -> Result<BinaryView> {
        let handle = unsafe_try!(BNCreateBinaryViewOfType(self.handle, data.handle()))?;

        // Add to the current active binary views
        crate::ACTIVE_BINARYVIEWS.fetch_add(1, Ordering::SeqCst);

        Ok(BinaryView {
            handle: Arc::new(BinjaBinaryView::new(handle)),
            meta: data.meta.clone(),
        })
    }
}
