//! Provides analysis for functions in a binary as well as an entry point into the LLIL, MLIL,
//! and HLIL functions
//!
//! Example:
//!
//! ```rust
//! let bv = binaryview::BinaryView::new_from_filename("/bin/ls").unwrap();
//! for func in bv.functions() {
//!     print!("Func {} @ {:#x}\n", func.name(), func.start());
//! }
//!
use core::*;

use anyhow::Result;

use std::convert::TryInto;
use std::fmt;
use std::sync::Arc;

use crate::architecture::CoreArchitecture;
use crate::basicblock::BasicBlock;
use crate::binaryview::BinaryView;
use crate::binjastr::BinjaStr;
use crate::highlevelil::{HighLevelILFunction, HighLevelILInstruction};
use crate::lowlevelil::{LowLevelILFunction, LowLevelILInstruction};
use crate::mediumlevelil::{MediumLevelILFunction, MediumLevelILInstruction};
use crate::reference::ReferenceSource;
use crate::symbol::{Symbol, SymbolType};
use crate::traits::{BasicBlockTrait, FunctionTrait};
use crate::unsafe_try;
use crate::wrappers::BinjaFunction;

/// Struct helping analyzing a particular function
#[derive(Clone)]
#[allow(dead_code)]
pub struct Function {
    handle: Arc<BinjaFunction>,
    advanced_analysis_requests: u64,
    symbol: Option<Symbol>,
    start: u64,
    symbol_type: Option<SymbolType>,
}

unsafe impl Send for Function {}
unsafe impl Sync for Function {}

impl Function {
    /// Create a new `Function` from a `BNFunction` from Binary Ninja
    pub fn new(func: *mut BNFunction) -> Result<Function> {
        let handle = unsafe_try!(BNNewFunctionReference(func))?;

        let start = unsafe { BNGetFunctionStart(handle) };
        let function_symbol = unsafe { BNGetFunctionSymbol(handle) };

        let mut curr_symbol = None;
        let mut symbol_type = None;
        if !function_symbol.is_null() {
            let symbol = Symbol::new(function_symbol);
            symbol_type = Some(symbol.symbol_type());
            curr_symbol = Some(symbol);
        }

        Ok(Function {
            handle: Arc::new(BinjaFunction::new(handle)),
            advanced_analysis_requests: 0,
            symbol: curr_symbol,
            start,
            symbol_type,
        })
    }

    /// Create a `Function` from an existing `Arc<BinjaFunction>`
    pub fn from_arc(func: Arc<BinjaFunction>) -> Function {
        let start = unsafe { BNGetFunctionStart(**func) };
        let function_symbol = unsafe { BNGetFunctionSymbol(**func) };

        let mut curr_symbol = None;
        let mut symbol_type = None;
        if !function_symbol.is_null() {
            let symbol = Symbol::new(function_symbol);
            symbol_type = Some(symbol.symbol_type());
            curr_symbol = Some(symbol);
        }

        Function {
            handle: func,
            advanced_analysis_requests: 0,
            symbol: curr_symbol,
            start,
            symbol_type,
        }
    }

    pub fn handle(&self) -> *mut BNFunction {
        **self.handle
    }

    /*
    pub fn new_from_block(block: *mut BNBasicBlock) -> Option<Function> {
        let func = unsafe_try!(BNGetBasicBlockFunction(block)).unwrap();
        Function::new(func)
    }
    */

    /// Get the start of the current function
    pub fn start(&self) -> u64 {
        self.start
    }

    /// Get the name of the current function
    ///
    /// # Example
    ///
    /// ```
    /// use binja_rs::binaryview::BinaryView;
    /// let bv = BinaryView::new_from_filename("tests/ls").unwrap();
    /// ```
    pub fn name(&self) -> Option<BinjaStr> {
        if let Some(sym) = &self.symbol {
            return Some(sym.name());
        }
        None
    }

    /// Get the architecture for this function
    pub fn arch(&self) -> Result<CoreArchitecture> {
        CoreArchitecture::new_from_func(self.handle())
    }

    pub fn blocks(&self) -> Vec<BasicBlock> {
        let mut count = 0;

        let mut res = Vec::new();

        print!("Blocks start... ");

        unsafe {
            let blocks = BNGetFunctionBasicBlockList(self.handle(), &mut count);
            let blocks_slice = std::slice::from_raw_parts(blocks, count as usize);
            for block in blocks_slice {
                res.push(BasicBlock::new(*block, self.clone()));
            }

            BNFreeBasicBlockList(blocks, count);
        }

        print!("end {}\n", count);

        res
    }

    /// Retrieve Low Level IL for the current function
    pub fn low_level_il(&self) -> Result<LowLevelILFunction> {
        LowLevelILFunction::new(self.handle.clone())
    }

    /// Alias for `self.low_level_il`
    pub fn llil(&self) -> Result<LowLevelILFunction> {
        self.low_level_il()
    }

    /// Alias for `self.low_level_il.ssa_form`
    pub fn llilssa(&self) -> Result<LowLevelILFunction> {
        self.low_level_il()?.ssa_form()
    }

    /// Retrieve Medium Level IL for the current function
    pub fn medium_level_il(&self) -> Result<MediumLevelILFunction> {
        MediumLevelILFunction::new(self.handle.clone())
    }

    /// Alias for `self.medium_level_il`
    pub fn mlil(&self) -> Result<MediumLevelILFunction> {
        self.medium_level_il()
    }

    /// Retrieve the Medium Level IL SSA version of this function
    pub fn mlilssa(&self) -> Result<MediumLevelILFunction> {
        self.medium_level_il()?.ssa_form()
    }

    /// Retrieve High Level IL for the current function
    pub fn high_level_il(&self) -> Result<HighLevelILFunction> {
        HighLevelILFunction::new(self.handle.clone())
    }

    /// Alias for `self.high_level_il`
    pub fn hlil(&self) -> Result<HighLevelILFunction> {
        self.high_level_il()
    }

    /// Retrieve the High Level IL SSA version of this function
    pub fn hlilssa(&self) -> Result<HighLevelILFunction> {
        self.high_level_il()?.ssa_form()
    }

    /// Returns all of the LLIL instructions for this function
    pub fn llil_instructions(&self) -> Result<Vec<LowLevelILInstruction>> {
        let mut res = Vec::new();

        for bb in self.llil()?.blocks() {
            for instr in bb.il() {
                res.push(instr);
            }
        }

        Ok(res)
    }

    /// Returns all of the MLIL instructions for this function
    pub fn mlil_instructions(&self) -> Result<Vec<MediumLevelILInstruction>> {
        let mut res = Vec::new();

        for bb in self.mlil()?.blocks() {
            for instr in bb.il() {
                res.push(instr);
            }
        }

        Ok(res)
    }

    /// Returns all of the HLIL instructions for this function
    pub fn hlil_instructions(&self) -> Result<Vec<HighLevelILInstruction>> {
        let mut res = Vec::new();

        for bb in self.hlil()?.blocks() {
            for instr in bb.il() {
                res.push(instr);
            }
        }

        Ok(res)
    }

    /// Returns all of the HLIL expressions for this function. This is not by instruction, but
    /// each individual expression available in this function
    pub fn hlil_expressions(&self) -> Result<Vec<HighLevelILInstruction>> {
        // Get the HLIL form of this function
        let curr_func = self.hlil()?;

        // Initialize the resulting Vec
        let mut res = Vec::new();

        // Get the number of expressions in this function
        let expr_len = unsafe { BNGetHighLevelILExprCount(curr_func.handle()) };

        // For each expression, attempt to get the HLIL instruction and add it to the result
        for index in 0..expr_len {
            if let Ok(instr) = HighLevelILInstruction::from_expr(
                curr_func.clone(),
                index.try_into().unwrap(),
                None,
            ) {
                res.push(instr);
            }
        }

        // Sanity check we have all the instructions
        assert!(expr_len == res.len(), "Didn't find all HLIL expressions");

        // Return the result
        Ok(res)
    }

    /// Returns all of the HLILSSA instructions for this function
    pub fn hlilssa_instructions(&self) -> Result<Vec<HighLevelILInstruction>> {
        let mut res = Vec::new();

        for bb in self.hlil()?.ssa_form()?.blocks() {
            for instr in bb.il() {
                res.push(instr);
            }
        }

        Ok(res)
    }

    /// Returns all of the HLILSSA expressions for this function. This is not by instruction, but
    /// each individual expression available in this function
    pub fn hlilssa_expressions(&self) -> Result<Vec<HighLevelILInstruction>> {
        // Get the HLILSSA form of this function
        let curr_func = self.hlil()?.ssa_form()?;

        // Initialize the resulting Vec
        let mut res = Vec::new();

        // Get the number of expressions in this function
        let expr_len = unsafe { BNGetHighLevelILExprCount(curr_func.handle()) };

        // For each expression, attempt to get the HLIL instruction and add it to the result
        for index in 0..expr_len {
            if let Ok(instr) = HighLevelILInstruction::from_expr(
                curr_func.clone(),
                index.try_into().unwrap(),
                None,
            ) {
                res.push(instr);
            }
        }

        // Sanity check we have all the instructions
        assert!(expr_len == res.len(), "Didn't find all HLILSSA expressions");

        // Return the result
        Ok(res)
    }

    /// Return all MLIL expressions in the binary, filtered by the given filter function
    pub fn mlilssa_expressions_filtered(
        &self,
        bv: &BinaryView,
        filter: &(dyn Fn(&BinaryView, &MediumLevelILInstruction) -> bool + 'static + Sync),
    ) -> Result<Vec<MediumLevelILInstruction>> {
        // Get the HLILSSA form of this function
        let curr_func = self.mlil()?.ssa_form()?;

        // Initialize the resulting Vec
        let mut res = Vec::new();

        // Get the number of expressions in this function
        let expr_len = unsafe { BNGetMediumLevelILExprCount(curr_func.handle()) };

        // For each expression, attempt to get the MLIL instruction and add it to the result
        for index in 0..expr_len {
            let instr = MediumLevelILInstruction::from_expr(
                curr_func.clone(),
                index.try_into().unwrap(),
                None,
            );

            if filter(&bv, &instr) {
                res.push(instr);
            }
        }

        // Return the result
        Ok(res)
    }

    /// Return all HLIL expressions in the binary, filtered by the given filter function
    pub fn hlil_expressions_filtered(
        &self,
        bv: &BinaryView,
        filter: &(dyn Fn(&BinaryView, &HighLevelILInstruction) -> bool + 'static + Sync),
    ) -> Result<Vec<HighLevelILInstruction>> {
        // Get the HLILSSA form of this function
        let curr_func = self.hlil()?;

        // Initialize the resulting Vec
        let mut res = Vec::new();

        // Get the number of expressions in this function
        let expr_len = unsafe { BNGetHighLevelILExprCount(curr_func.handle()) };

        // For each expression, attempt to get the HLIL instruction and add it to the result
        for index in 0..expr_len {
            if let Ok(instr) = HighLevelILInstruction::from_expr(
                curr_func.clone(),
                index.try_into().unwrap(),
                None,
            ) {
                if filter(&bv, &instr) {
                    res.push(instr);
                }
            }
        }

        // Return the result
        Ok(res)
    }

    /// Return all HLIL expressions in the binary, filtered by the given filter function
    pub fn hlilssa_expressions_filtered(
        &self,
        bv: &BinaryView,
        filter: &(dyn Fn(&BinaryView, &HighLevelILInstruction) -> bool + 'static + Sync),
    ) -> Result<Vec<HighLevelILInstruction>> {
        // Get the HLILSSA form of this function
        let curr_func = self.hlil()?.ssa_form()?;

        // Initialize the resulting Vec
        let mut res = Vec::new();

        // Get the number of expressions in this function
        let expr_len = unsafe { BNGetHighLevelILExprCount(curr_func.handle()) };

        // For each expression, attempt to get the HLIL instruction and add it to the result
        for index in 0..expr_len {
            if let Ok(instr) = HighLevelILInstruction::from_expr(
                curr_func.clone(),
                index.try_into().unwrap(),
                None,
            ) {
                if filter(&bv, &instr) {
                    res.push(instr);
                }
            }
        }

        // Return the result
        Ok(res)
    }

    /// Return all HLIL instructions in the binary, filtered by the given filter function
    pub fn hlil_instructions_filtered(
        &self,
        bv: &BinaryView,
        filter: &(dyn Fn(&BinaryView, &HighLevelILInstruction) -> bool + 'static + Sync),
    ) -> Result<Vec<HighLevelILInstruction>> {
        let mut res = Vec::new();

        for bb in self.hlil()?.blocks() {
            for instr in bb.il() {
                if filter(&bv, &instr) {
                    res.push(instr);
                }
            }
        }

        Ok(res)
    }

    pub fn get_low_level_il_at(&self, addr: u64) -> Result<LowLevelILInstruction> {
        let index =
            unsafe { BNGetLowLevelILForInstruction(self.handle(), self.arch()?.handle(), addr) };

        if index >= self.llil()?.len() {
            return Err(anyhow!(
                "LLIL instruction index is not in the LLIL instructions"
            ));
        }

        Ok(LowLevelILInstruction::from_func_index(self.llil()?, index))
    }

    /// Returns possible call sites contained in this function.
    /// This includes ordinary calls, tail calls, and indirect jumps. Not all of the
    /// returned call sites are necessarily true call sites; some may simply be
    /// unresolved indirect jumps, for example.
    pub fn call_sites(&self) -> Vec<ReferenceSource> {
        let mut count = 0;

        let mut res = Vec::new();

        unsafe {
            let xrefs = BNGetFunctionCallSites(self.handle(), &mut count);
            let xrefs_slice = std::slice::from_raw_parts(xrefs, count as usize);
            for xref in xrefs_slice {
                if let Ok(new_ref) = ReferenceSource::new(*xref) {
                    res.push(new_ref);
                }
            }

            BNFreeCodeReferences(xrefs, count);
        }

        res
    }

    /// Return the list of functions that this function calls
    /// BinaryView is needed to be passed because the function doesn't hold a reference
    /// to the BinaryView itself.
    pub fn callees(&self, bv: &BinaryView) -> Result<Vec<Function>> {
        let mut res = Vec::new();

        for site in self.call_sites() {
            res.extend(bv.get_callees(&site)?);
        }

        Ok(res)
    }

    /// Return the list of functions that call this function
    /// BinaryView is needed to be passed because the function doesn't hold a reference
    /// to the BinaryView itself.
    pub fn callers(&self, bv: &BinaryView) -> Vec<ReferenceSource> {
        bv.get_callers(self.start)
    }
}

impl fmt::Display for Function {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl fmt::Debug for Function {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut result = f.debug_struct("Function");
        result.field("start", &format_args!("{:#x}", self.start()));

        if let Some(symbol_type) = &self.symbol_type {
            result.field("symbol_type", &format!("{:?}", symbol_type));
        }

        if let Some(symbol) = &self.symbol {
            result.field("symbol", &symbol.name());
        }

        if let Some(name) = self.name() {
            result.field("name", &name);
        }

        result.finish()
    }
}

/*
impl std::ops::Drop for Function {
    fn drop(&mut self) {
        // Stop any of the analysis requests if any are pending
        if self.advanced_analysis_requests > 0 {
            unsafe { BNReleaseAdvancedFunctionAnalysisDataMultiple(self.handle(),
                                                                   self.advanced_analysis_requests)
            }
        }

        // Drop the created symbol if we have one
        if let Some(symbol) = &self.symbol {
            print!("{}\n", symbol.name());
            drop(symbol);
        }
    }
}
*/
