//! Provides basic `new` and `Drop` implementations for Binary Ninja core types for easier 
//! multithreading access of these types

use core::*;

/// Used for boiler plate creation of `new` and `Drop` functions for basic Binary Ninja core types
///
/// Example:
///
/// ```
/// impl_rust_binja_core!(wrapper BinjaFunction, kind BNFunction, freefunc BNFreeFunction);
/// ```
#[macro_export]
macro_rules! impl_rust_binja_core {
    (wrapper $name:ident, kind $typ:ident, freefunc $freefunc:expr) => {
        pub struct $name {
            ptr: *mut $typ,
        }

        impl $name {
            pub fn new(ptr: *mut $typ) -> Self {
                trace!("Making   {}: {:#x}", stringify!($typ), ptr as u64);
                Self { ptr }
            }
        }

        impl std::ops::Drop for $name {
            fn drop(&mut self) {
                trace!("Dropping {}: {:#x}", stringify!($typ), self.ptr as u64);
                unsafe { $freefunc(self.ptr); }
            }
        }

        impl std::ops::Deref for $name {
            type Target = *mut $typ;

            fn deref(&self) -> &Self::Target {
                &self.ptr
            }
        }
    }
}

impl_rust_binja_core!(wrapper BinjaFunction, kind BNFunction, freefunc BNFreeFunction);
impl_rust_binja_core!(wrapper BinjaMetadata, kind BNMetadata, freefunc BNFreeMetadata);
impl_rust_binja_core!(wrapper BinjaDataBuffer, kind BNDataBuffer, freefunc BNFreeDataBuffer);
impl_rust_binja_core!(wrapper BinjaBinaryView, kind BNBinaryView, freefunc BNFreeBinaryView);
impl_rust_binja_core!(wrapper BinjaBasicBlock, kind BNBasicBlock, freefunc BNFreeBasicBlock);
impl_rust_binja_core!(wrapper BinjaPossibleValueSet, kind BNPossibleValueSet, 
                      freefunc BNFreePossibleValueSet);
impl_rust_binja_core!(wrapper BinjaPlatform, kind BNPlatform, freefunc BNFreePlatform);
impl_rust_binja_core!(wrapper BinjaLowLevelILFunction, kind BNLowLevelILFunction, 
                      freefunc BNFreeLowLevelILFunction);
impl_rust_binja_core!(wrapper BinjaMediumLevelILFunction, kind BNMediumLevelILFunction, 
                      freefunc BNFreeMediumLevelILFunction);
impl_rust_binja_core!(wrapper BinjaType, kind BNType, freefunc BNFreeType);
impl_rust_binja_core!(wrapper BinjaHighLevelILFunction, kind BNHighLevelILFunction, 
                      freefunc BNFreeHighLevelILFunction);
impl_rust_binja_core!(wrapper BinjaSymbol, kind BNSymbol, freefunc BNFreeSymbol);
impl_rust_binja_core!(wrapper BinjaFileMetadata, kind BNFileMetadata, freefunc BNFreeFileMetadata);
