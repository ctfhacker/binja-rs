//! Provides abstractions over `StringReference` handled by Binary Ninja core
use binja_sys::*;

use std::string::String;

use crate::databuffer::DataBuffer;

pub struct StringReference {
    start: u64,
    data: String,
    // type_: BNStringType,
}

impl StringReference {
    pub fn new(string_ref: BNStringReference, data_buffer: DataBuffer) -> StringReference {
        StringReference {
            start:  string_ref.start,
            // type_:  string_ref.type_,
            data: data_buffer.as_str().to_string()
        }
    }

    /// Alias for self.start
    pub fn address(&self) -> u64 {
        self.start
    }

    /// Get the address of the start of the string buffer
    pub fn start(&self) -> u64 {
        self.start
    }

    /// Get the length of the `DataBuffer` for this `StringReference`
    pub fn len(&self) -> usize {
        self.data.len()
    }
}

impl std::ops::Deref for StringReference {
    type Target = str;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::fmt::Display for StringReference {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.data)
    }
}
