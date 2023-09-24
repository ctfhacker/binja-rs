//! Provides custom type used for storing string references handed off by Binary Ninja core
use binja_sys::*;
use std::borrow::{Cow, Borrow};
use std::ffi::CStr;
use std::hash::{Hash, Hasher};
use std::fmt;
use std::fmt::{Display, Debug};
use std::ops::{Drop, Deref};

pub struct BinjaStr {
    data: *mut std::os::raw::c_char,
    len: usize
}

impl From<*mut std::os::raw::c_char> for BinjaStr {
    fn from(data: *mut std::os::raw::c_char) -> BinjaStr {
        unsafe {
            let s = CStr::from_ptr(data).to_str().expect("Invalid UTF8 string");
            // If we get here, we know we have a valid utf8 string
            BinjaStr { data, len: s.len() }
        }
    }
}

impl BinjaStr {
    pub fn new(data: *mut ::std::os::raw::c_char) -> BinjaStr {
        unsafe {
            let s = CStr::from_ptr(data).to_str().expect("Invalid UTF8 string");
            // If we get here, we know we have a valid utf8 string
            BinjaStr { data, len: s.len() }
        }

    }
    pub fn as_str(&self) -> &str {
        use std::{str, slice};
        unsafe {
            str::from_utf8_unchecked(slice::from_raw_parts(self.data as *const u8, self.len))
        }
    }

    pub fn to_string_lossy(&self) -> Cow<'_, str> {
        unsafe { 
            CStr::from_ptr(self.data).to_string_lossy()
        }
    }
}

impl Drop for BinjaStr {
    fn drop(&mut self) {
        unsafe { BNFreeString(self.data) };
    }
}

impl Deref for BinjaStr {
    type Target = str;

    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl Display for BinjaStr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Debug for BinjaStr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl Borrow<str> for BinjaStr {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl Hash for BinjaStr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}
