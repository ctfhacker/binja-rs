//! Provides `FileMetadata` 
use core::*;

use anyhow::Result;

use std::path::PathBuf;
use std::sync::Arc;
use std::ffi::{CStr, CString};

use crate::unsafe_try;
use crate::startup::init_plugins;
use crate::wrappers::BinjaFileMetadata;

/// Struct containing handle to `FileMetadata` from Binary Ninja
#[derive(Clone)]
pub struct FileMetadata {
    pub handle: Arc<BinjaFileMetadata>,
}

impl FileMetadata {
    pub fn new() -> Result<FileMetadata> {
        init_plugins();

        let handle = unsafe_try!(BNCreateFileMetadata())?;

        Ok(FileMetadata { handle: Arc::new(BinjaFileMetadata::new(handle)) })
    }

    pub fn handle(&self) -> *mut BNFileMetadata {
        **self.handle
    }

    /// Create a `FileMetadata` from a filename
    pub fn from_filename(name: &str) -> Result<FileMetadata> {
        let metadata = FileMetadata::new()?;
        let ffi_name = CString::new(name).unwrap();
        unsafe { BNSetFilename(metadata.handle(), ffi_name.as_ptr()) }
        Ok(metadata)
    }

    /// Retrieve the filename for this `FileMetadata`
    pub fn filename(&self) -> PathBuf {
        unsafe { 
            let filename = BNGetFilename(self.handle());
            let name = CStr::from_ptr(filename).to_string_lossy().into_owned();
            BNFreeString(filename);
            PathBuf::from(name)
        }
    }

}
