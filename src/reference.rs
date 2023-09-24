//! Provides abstractions for `ReferenceSource` for cross references
use anyhow::Result;

use binja_sys::BNReferenceSource;

use crate::function::Function;
use crate::architecture::CoreArchitecture;
use crate::lowlevelil::LowLevelILInstruction;
use crate::mediumlevelil::MediumLevelILInstruction;
use crate::highlevelil::HighLevelILInstruction;

#[derive(Clone)]
pub struct ReferenceSource {
    pub func: Function,
    pub arch: CoreArchitecture,
    pub addr: u64
}

impl ReferenceSource {
    pub fn new(xref: BNReferenceSource) -> Result<Self> {
        let func = Function::new(xref.func).expect("Failed to make function in xref");
        let arch = CoreArchitecture::new(xref.arch);
        let addr = xref.addr;

        Ok(ReferenceSource {
            func, arch, addr
        })
    }

    /// Get the `ReferenceSource` as a `BNResourceSource`
    pub fn as_bnreferencesource(&self) -> BNReferenceSource {
        BNReferenceSource {
            func: self.func.handle(),
            arch: self.arch.handle(),
            addr: self.addr
        }
    }

    /// Get the LLIL instruction for this xref
    pub fn low_level_il(&self) -> Result<LowLevelILInstruction> {
        self.func.get_low_level_il_at(self.addr)
    }

    /// Get the LLIL instruction for this xref
    pub fn llil(&self) -> Result<LowLevelILInstruction> {
        self.low_level_il()
    }

    /// Get the LLILSSA instruction for this xref
    pub fn llilssa(&self) -> Result<LowLevelILInstruction> {
        self.llil()?.ssa_form()
    }

    /// Get the MLIL instruction for this xref
    pub fn medium_level_il(&self) -> Result<MediumLevelILInstruction> {
        self.low_level_il()?.medium_level_il()
    }

    /// Get the MLIL instruction for this xref
    pub fn mlil(&self) -> Result<MediumLevelILInstruction> {
        self.medium_level_il()
    }

    /// Get the MLIL instruction for this xref
    pub fn mlilssa(&self) -> Result<MediumLevelILInstruction> {
        self.llilssa()?.mlilssa()
    }

    /// Get the HLIL instruction for this xref
    pub fn high_level_il(&self) -> Result<HighLevelILInstruction> {
        self.low_level_il()?.medium_level_il()?.high_level_il()
    }

    /// Get the HLIL instruction for this xref
    pub fn hlil(&self) -> Result<HighLevelILInstruction> {
        self.high_level_il()
    }

    /// Get the HLIL instruction for this xref
    pub fn hlilssa(&self) -> Result<HighLevelILInstruction> {
        self.hlil()?.ssa_form()
    }

    /// Alias for self.addr
    pub fn address(&self) -> u64 {
        self.addr
    }
}
