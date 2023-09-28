//! Provides Architecture abstractions
use binja_sys::*;

use anyhow::Result;

use std::borrow::Cow;
use std::collections::HashMap;
use std::ffi::CStr;
use std::sync::Arc;

use crate::binjastr::BinjaStr;
use crate::il::Register;
use crate::unsafe_try;

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct BinjaCoreArchitecture {
    ptr: *mut BNArchitecture,
}

impl BinjaCoreArchitecture {
    pub fn new(ptr: *mut BNArchitecture) -> Self {
        Self { ptr }
    }
}

impl std::ops::Deref for BinjaCoreArchitecture {
    type Target = *mut BNArchitecture;

    fn deref(&self) -> &Self::Target {
        &self.ptr
    }
}

#[derive(Clone, PartialEq, Hash, Eq)]
pub struct CoreArchitecture {
    handle: Arc<BinjaCoreArchitecture>,
}

unsafe impl Send for CoreArchitecture {}
unsafe impl Sync for CoreArchitecture {}

impl CoreArchitecture {
    pub fn new(handle: *mut BNArchitecture) -> CoreArchitecture {
        CoreArchitecture {
            handle: Arc::new(BinjaCoreArchitecture::new(handle)),
        }
    }

    pub fn new_from_func(func: *mut BNFunction) -> Result<CoreArchitecture> {
        let handle = unsafe_try!(BNGetFunctionArchitecture(func))?;
        Ok(CoreArchitecture::new(handle))
    }

    pub fn handle(&self) -> *mut BNArchitecture {
        **self.handle
    }

    /// Get the name of the architecture
    pub fn name(&self) -> Cow<'_, str> {
        unsafe {
            let ptr = BNGetArchitectureName(self.handle());
            CStr::from_ptr(ptr).to_string_lossy()
        }
    }

    /// Get the `Register` of index `reg` for this architecture
    pub fn get_reg(&self, reg: u32) -> Register {
        Register::new(self.clone(), reg)
    }

    /// Get the name of `Register` of index `reg` for this architecture
    pub fn get_reg_name(&self, reg: u32) -> BinjaStr {
        unsafe { BinjaStr::new(BNGetArchitectureRegisterName(self.handle(), reg)) }
    }

    /// Get the flag name of the given `flag` index
    pub fn get_flag_name(&self, flag: u32) -> Option<BinjaStr> {
        let flags = self.flags_by_index();
        let string = flags.get(&flag)?;
        unsafe { Some(BinjaStr::new(BNAllocString(*string))) }
    }

    /// Get the intrinsic name of the given `index`
    pub fn get_intrinsic_name(&self, index: u32) -> BinjaStr {
        unsafe { BinjaStr::new(BNGetArchitectureIntrinsicName(self.handle(), index)) }
    }

    /// Get the register stack name for the register stack number
    pub fn get_register_stack_name(&self, index: u32) -> Option<BinjaStr> {
        let string = unsafe_try!(BNGetArchitectureRegisterStackName(self.handle(), index)).ok()?;
        Some(BinjaStr::new(string))
    }

    pub fn flags_by_index(&self) -> HashMap<u32, *mut i8> {
        let mut count = 0;
        // let mut flags = HashMap::new();
        let mut flags_by_index = HashMap::new();

        unsafe {
            let found_flags = BNGetAllArchitectureFlags(self.handle(), &mut count);

            let flags_slice = std::slice::from_raw_parts(found_flags, count as usize);

            for flag in flags_slice {
                let name = BNGetArchitectureFlagName(self.handle(), *flag);
                // flags.insert(name, flag);
                flags_by_index.insert(*flag, name);
            }

            BNFreeRegisterList(found_flags);
        }

        flags_by_index
    }
}

impl std::fmt::Debug for CoreArchitecture {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("CoreArchitecture")
            .field("name", &self.name())
            .finish()
    }
}
