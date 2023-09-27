//! Provides [`CallingConvention`]

use anyhow::Result;
use binja_sys::*;

use crate::architecture::BinjaCoreArchitecture;
use crate::function::Function;
use crate::wrappers::BinjaCallingConvention;

use std::ffi::CStr;
use std::sync::Arc;

/// Struct containing handle to `CallingConvention` from Binary Ninja
#[derive(Debug, Clone)]
pub struct CallingConvention {
    pub handle: Arc<BinjaCallingConvention>,
    pub arch: Arc<BinjaCoreArchitecture>,
    pub name: String,

    pub caller_saved_regs: Vec<String>,
    pub callee_saved_regs: Vec<String>,
    pub int_arg_regs: Vec<String>,
    pub float_arg_regs: Vec<String>,
    pub implicitly_defined_regs: Vec<String>,

    pub arg_regs_share_index: bool,
    pub arg_regs_for_varargs: bool,
    pub stack_reserved_for_arg_regs: bool,
    pub stack_adjusted_on_return: bool,
    pub eligible_for_heuristics: bool,

    pub int_return_reg: Option<String>,
    pub high_int_return_reg: Option<String>,
    pub float_return_reg: Option<String>,
    pub global_pointer_reg: Option<String>,
}

impl CallingConvention {
    pub fn new_from_function(func: &Function) -> Result<Self> {
        timeloop::scoped_timer!(crate::Timer::CallingConvention__new_from_function);

        let func_handle = func.handle();
        let handle = unsafe { BNGetFunctionCallingConvention(func_handle) };

        // Get the architecture for this calling convention
        let arch = unsafe { BNGetCallingConventionArchitecture(handle.convention) };
        let arch = Arc::new(BinjaCoreArchitecture::new(arch));

        let name = unsafe { BNGetCallingConventionName(handle.convention) };
        let name = unsafe { CStr::from_ptr(name) }
            .to_str()
            .unwrap()
            .to_string();

        let arg_regs_share_index = unsafe { BNAreArgumentRegistersSharedIndex(handle.convention) };
        let arg_regs_for_varargs =
            unsafe { BNAreArgumentRegistersUsedForVarArgs(handle.convention) };
        let stack_reserved_for_arg_regs =
            unsafe { BNIsStackReservedForArgumentRegisters(handle.convention) };
        let stack_adjusted_on_return = unsafe { BNIsStackAdjustedOnReturn(handle.convention) };
        let eligible_for_heuristics = unsafe { BNIsEligibleForHeuristics(handle.convention) };

        macro_rules! read_list {
            ($get_func:ident, $free_func:ident) => {{
                let mut count = 0;
                let mut res = Vec::new();
                unsafe {
                    let registers = $get_func(handle.convention, &mut count);
                    let regs_slice = std::slice::from_raw_parts(registers, count as usize);
                    for reg_index in regs_slice {
                        res.push(func.arch()?.get_reg_name(*reg_index).to_string());
                    }

                    $free_func(registers);
                }

                res
            }};
        }

        let caller_saved_regs = read_list!(BNGetCallerSavedRegisters, BNFreeRegisterList);
        let callee_saved_regs = read_list!(BNGetCalleeSavedRegisters, BNFreeRegisterList);
        let int_arg_regs = read_list!(BNGetIntegerArgumentRegisters, BNFreeRegisterList);
        let float_arg_regs = read_list!(BNGetFloatArgumentRegisters, BNFreeRegisterList);
        let implicitly_defined_regs =
            read_list!(BNGetImplicitlyDefinedRegisters, BNFreeRegisterList);

        macro_rules! read_reg {
            ($func:ident) => {{
                let val = unsafe { BNGetIntegerReturnValueRegister(handle.convention) };
                if val == 0xffff_ffff {
                    None
                } else {
                    Some(func.arch()?.get_reg_name(val).to_string())
                }
            }};
        }

        Ok(Self {
            handle: Arc::new(BinjaCallingConvention::new(handle)),
            arch,
            name,
            callee_saved_regs,
            caller_saved_regs,
            int_arg_regs,
            float_arg_regs,
            implicitly_defined_regs,
            arg_regs_share_index,
            arg_regs_for_varargs,
            stack_reserved_for_arg_regs,
            stack_adjusted_on_return,
            eligible_for_heuristics,
            int_return_reg: read_reg!(BNGetIntegerReturnValueRegister),
            high_int_return_reg: read_reg!(BNGetHighIntegerReturnValueRegister),
            float_return_reg: read_reg!(BNGetFloatReturnValueRegister),
            global_pointer_reg: read_reg!(BNGetGlobalPointerRegister),
        })
    }
}
