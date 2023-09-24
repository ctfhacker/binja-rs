#![feature(type_ascription)]
#![feature(associated_type_defaults)]

use anyhow::anyhow;

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

timeloop::impl_enum!(
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub enum Timer {
        BNGetFunctionArchitecture,
        BNGetArchitectureRegisterStackName,
        BNGetBasicBlockImmediateDominator,
        BNGetFileViewOfType,
        BNOpenExistingDatabase,
        BNCreateBinaryDataViewFromFilename,
        BNGetSymbolByAddress,
        BNCreateBinaryViewOfType,
        BNCreateFileMetadata,
        BNNewFunctionReference,
        BNGetFunctionHighLevelIL,
        BNGetHighLevelILSSAForm,
        BNNewBasicBlockReference,
        BNGetFunctionLowLevelIL,
        BNGetLowLevelILSSAForm,
        BNGetFunctionMediumLevelIL,
        BNGetMediumLevelILSSAForm,
        BNGetMediumLevelILBasicBlockForInstruction,
        BNCreateSaveSettings,
        BNUpdateAnalysisAndWait,
        BNUpdateAnalysis,
        BNGetEntryPoint,
        BNHasFunctions,
        BNGetViewLength,
        BNGetStartOffset,

        Startup__InitPlugins,
        BinaryView__NewFromFilename,
        BinaryView__Strings,
        BinaryView__UpdateAnalysisAndWait,
        BinaryView__UpdateAnalysis,
        BinaryView__EntryPoint,
        BinaryView__ParLLILInstructions,
        BinaryView__HasFunctions,
        BinaryView__Len,
        BinaryView__Start,
        BinaryView__Name,
        BinaryView__TypeName,
        BinaryView__Open,
        BinaryView__AvailableViewTypes,
        BinaryView__LLILInstructions,
        BinaryView__Platform,
    }
);

timeloop::create_profiler!(Timer);

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
    ($func:ident($($arg:expr),*)) => {{
        timeloop::scoped_timer!(crate::Timer::$func);

        unsafe {
            // Call the given BinjaCore function
            // let res = $e;
            let res = $func($($arg),*);

            if res.is_null() {
                // If the result is 0, return the anyhow error
                Err(crate::anyhow!("{} failed", stringify!($e)))
            } else {
                // Otherwise return the result
                Ok(res)
            }
        }
    }};
}
