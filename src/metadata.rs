//! Provides `Metadata`
use binja_sys::*;

use anyhow::Result;

use std::ffi::{CStr, CString};
use std::path::PathBuf;
use std::sync::Arc;

use crate::startup::init_plugins;
use crate::unsafe_try;
use crate::wrappers::BinjaMetadata;

/// Struct containing handle to `Metadata` from Binary Ninja
#[derive(Clone)]
pub struct Metadata {
    pub handle: Arc<BinjaMetadata>,
}

impl Metadata {
    pub fn new() -> Result<Metadata> {
        let handle = unsafe_try!(BNCreateMetadataOfType(BNMetadataType_KeyValueDataType))?;

        Ok(Metadata {
            handle: Arc::new(BinjaMetadata::new(handle)),
        })
    }

    pub fn handle(&self) -> *mut BNMetadata {
        **self.handle
    }

    pub fn len(&self) -> usize {
        unsafe { BNMetadataSize(self.handle()) }
    }

    pub fn insert(&mut self, key: &'static str, val: MetadataOption) -> bool {
        match val {
            MetadataOption::Str(value) => self.insert_str(key, value.as_str()),
            MetadataOption::U64(value) => self.insert_u64(key, value),
            MetadataOption::I64(value) => self.insert_i64(key, value),
            MetadataOption::F64(value) => self.insert_f64(key, value),
            MetadataOption::Bool(value) => self.insert_bool(key, value),
        }
    }

    /// Insert a string value into the metadata
    pub fn insert_str(&mut self, key: &str, value: &str) -> bool {
        let key_str = CString::new(key).unwrap();
        let value_str = CString::new(value).unwrap();

        println!("Insert {key} -> {value}");

        unsafe {
            // let value_handle = BNCreateMetadataStringData(value.as_ptr().cast());
            let value_handle = BNCreateMetadataStringData(value_str.as_ptr());
            println!("Value handle: {:#x}", value_handle as usize);

            BNMetadataSetValueForKey(self.handle(), key_str.as_ptr(), value_handle)
        }
    }

    /// Insert a u64 value into the metadata
    pub fn insert_u64(&mut self, key: &str, value: u64) -> bool {
        let key_str = CString::new(key).unwrap();

        unsafe {
            let value_handle = BNCreateMetadataUnsignedIntegerData(value);
            BNMetadataSetValueForKey(self.handle(), key_str.as_ptr(), value_handle)
        }
    }

    /// Insert a i64 value into the metadata
    pub fn insert_i64(&mut self, key: &str, value: i64) -> bool {
        let key_str = CString::new(key).unwrap();

        unsafe {
            let value_handle = BNCreateMetadataSignedIntegerData(value);
            BNMetadataSetValueForKey(self.handle(), key_str.as_ptr(), value_handle)
        }
    }

    /// Insert a f64 value into the metadata
    pub fn insert_f64(&mut self, key: &str, value: f64) -> bool {
        let key_str = CString::new(key).unwrap();

        unsafe {
            let value_handle = BNCreateMetadataDoubleData(value);
            BNMetadataSetValueForKey(self.handle(), key_str.as_ptr(), value_handle)
        }
    }

    /// Insert a bool value into the metadata
    pub fn insert_bool(&mut self, key: &str, value: bool) -> bool {
        let key_str = CString::new(key).unwrap();

        unsafe {
            let value_handle = BNCreateMetadataBooleanData(value);
            BNMetadataSetValueForKey(self.handle(), key_str.as_ptr(), value_handle)
        }
    }
}

/// The type of a key for metadata
#[derive(Debug)]
pub enum MetadataOption {
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
}

impl From<String> for MetadataOption {
    fn from(val: String) -> MetadataOption {
        MetadataOption::Str(val)
    }
}

impl From<&'static str> for MetadataOption {
    fn from(val: &'static str) -> MetadataOption {
        MetadataOption::Str(val.to_string())
    }
}

impl From<u8> for MetadataOption {
    fn from(val: u8) -> MetadataOption {
        MetadataOption::U64(val as u64)
    }
}

impl From<u16> for MetadataOption {
    fn from(val: u16) -> MetadataOption {
        MetadataOption::U64(val as u64)
    }
}

impl From<u32> for MetadataOption {
    fn from(val: u32) -> MetadataOption {
        MetadataOption::U64(val as u64)
    }
}

impl From<u64> for MetadataOption {
    fn from(val: u64) -> MetadataOption {
        MetadataOption::U64(val)
    }
}

impl From<i8> for MetadataOption {
    fn from(val: i8) -> MetadataOption {
        MetadataOption::I64(val as i64)
    }
}

impl From<i16> for MetadataOption {
    fn from(val: i16) -> MetadataOption {
        MetadataOption::I64(val as i64)
    }
}

impl From<i32> for MetadataOption {
    fn from(val: i32) -> MetadataOption {
        MetadataOption::I64(val as i64)
    }
}

impl From<i64> for MetadataOption {
    fn from(val: i64) -> MetadataOption {
        MetadataOption::I64(val)
    }
}

impl From<f64> for MetadataOption {
    fn from(val: f64) -> MetadataOption {
        MetadataOption::F64(val)
    }
}

impl From<bool> for MetadataOption {
    fn from(val: bool) -> MetadataOption {
        MetadataOption::Bool(val)
    }
}
