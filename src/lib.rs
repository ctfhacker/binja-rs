#![feature(type_ascription)]
#![feature(associated_type_defaults)]
#![feature(in_band_lifetimes)]

extern crate binja_sys as core;

#[macro_use] extern crate anyhow;
extern crate rayon;

// Good ole logging functionality
#[macro_use]
extern crate log;

pub mod binaryview;
pub mod filemetadata;
pub mod startup;
pub mod binjastr;
pub mod platform;
pub mod types;
pub mod function;
pub mod architecture;
pub mod reference;
pub mod stringreference;
pub mod databuffer;
pub mod lowlevelil;
pub mod instruction;
pub mod traits;
pub mod basicblock;
pub mod wrappers;
pub mod mediumlevelil;
pub mod il;
pub mod highlevelil;
pub mod binjalog;
pub mod symbol;

/// Used to easily wrap an option around the BinjaCore calls
///
/// Example:
///
/// pub fn new() -> Result<FileMetadata> {
///     let meta = FileMetadata{ 
///         handle: unsafe_try!(BNCreateFileMetadata())? 
///     }
/// }
#[macro_export]
macro_rules! unsafe_try {
    ($e:expr) => {{
        unsafe {
            // Call the given BinjaCore function
            let res = $e;

            if res.is_null() {
                // If the result is 0, return the anyhow error
                Err(anyhow!("{} failed", stringify!($e)))
            } else {
                // Otherwise return the result
                Ok(res)
            }
        }
    }}
}

