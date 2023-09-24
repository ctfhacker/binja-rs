#![feature(type_ascription)]
#![feature(associated_type_defaults)]

extern crate binja_sys as core;

#[macro_use]
extern crate anyhow;
extern crate rayon;

// Good ole logging functionality
#[macro_use]
extern crate log;

pub mod architecture;
pub mod basicblock;
pub mod binaryview;
pub mod binjalog;
pub mod binjastr;
pub mod databuffer;
pub mod filemetadata;
pub mod function;
pub mod highlevelil;
pub mod il;
pub mod instruction;
pub mod lowlevelil;
pub mod mediumlevelil;
pub mod platform;
pub mod plugin;
pub mod reference;
pub mod savesettings;
pub mod startup;
pub mod stringreference;
pub mod symbol;
pub mod traits;
pub mod types;
pub mod wrappers;

use std::sync::atomic::AtomicU64;

pub static ACTIVE_BINARYVIEWS: AtomicU64 = AtomicU64::new(0);

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
    }};
}
