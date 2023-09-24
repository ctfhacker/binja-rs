//! Provides `Databuffer` used for handling byte buffers returned from Binary Ninja core
use core::*;

use std::borrow::Cow;
use std::string::String;
use std::sync::Arc;

use crate::wrappers::BinjaDataBuffer;

pub struct DataBuffer {
    handle: Arc<BinjaDataBuffer>,
}

impl DataBuffer {
    /// Create a new `DataBuffer` from a `BNDataBuffer`
    pub fn new_from_handle(handle: *mut BNDataBuffer) -> DataBuffer {
        DataBuffer {
            handle: Arc::new(BinjaDataBuffer::new(handle)),
        }
    }

    pub fn handle(&self) -> *mut BNDataBuffer {
        **self.handle
    }

    /// Get the length of the data buffer
    pub fn len(&self) -> usize {
        unsafe { BNGetDataBufferLength(self.handle()) as usize }
    }

    /// Get the underlying pointer to the data
    fn as_ptr(&self) -> *const u8 {
        unsafe { BNGetDataBufferContents(self.handle()) as usize as *const u8 }
    }

    /// Get the data contents of this buffer
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    /// Attempt to return the contents of the `DataBuffer` as a `String`
    pub fn as_str<'a>(&'a self) -> Cow<'a, str> {
        String::from_utf8_lossy(self.as_bytes())
    }
}
