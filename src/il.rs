//! Provides variables used in IL operations
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

use anyhow::{anyhow, Result};
use log::{debug, info};

use std::sync::Arc;

use crate::architecture::CoreArchitecture;
use crate::binaryview::BinaryView;
use crate::binjastr::BinjaStr;
use crate::function::Function;
use crate::highlevelil::{HighLevelILFunction, HighLevelILInstruction, HighLevelILOperation};
use crate::mediumlevelil::{MediumLevelILInstruction, MediumLevelILOperation};
use crate::wrappers::BinjaType;
use binja_sys::{BNGetGotoLabelName, BNGetHighLevelILSSAVarDefinition};
use binja_sys::{BNGetMediumLevelILSSAVarDefinition, BNGetMediumLevelILSSAVarUses};
use binja_sys::{BNGetVariableName, BNGetVariableType, BNToVariableIdentifier, BNVariable};

#[derive(Clone, Hash, PartialEq, Eq)]
pub struct Register {
    arch: CoreArchitecture,
    pub index: u32,
}

impl Register {
    pub fn new(arch: CoreArchitecture, index: u32) -> Self {
        Register { arch, index }
    }

    /// Get the name of this register
    pub fn name(&self) -> String {
        if self.index & 0x8000_0000 > 0 {
            format!("temp{}", self.index & 0x7fff_ffff)
        } else {
            self.arch
                .get_reg_name(self.index)
                .to_string_lossy()
                .to_string()
        }
    }
}

impl std::fmt::Display for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::fmt::Debug for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name())
        /*
        f.debug_struct("Register")
            .field("name", &self.name())
            .finish()
        */
    }
}

#[derive(Clone)]
pub struct Flag {
    arch: CoreArchitecture,
    index: u32,
}

impl Flag {
    pub fn new(arch: CoreArchitecture, index: u32) -> Self {
        Flag { arch, index }
    }

    /// Get the name of this register
    pub fn name(&self) -> Option<String> {
        if self.index & 0x8000_0000 > 0 {
            Some(format!("cond:{}", self.index & 0x7fff_ffff))
        } else {
            Some(
                self.arch
                    .get_flag_name(self.index)?
                    .to_string_lossy()
                    .to_string(),
            )
        }
    }
}

impl std::fmt::Display for Flag {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if let Some(name) = self.name() {
            write!(f, "{}", name)
        } else {
            write!(f, "FlagNotFound")
        }
    }
}

impl std::fmt::Debug for Flag {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if let Some(name) = self.name() {
            write!(f, "{}", name)
        } else {
            write!(f, "FlagNotFound")
        }
    }
}

/// An SSA Register
#[derive(Clone, Hash, PartialEq, Eq)]
pub struct SSARegister {
    pub reg: Register,
    pub version: u32,
}

impl SSARegister {
    pub fn new(reg: Register, version: u32) -> Self {
        SSARegister { reg, version }
    }
}

impl std::fmt::Debug for SSARegister {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("SSARegister")
            .field("reg", &self.reg)
            .field("version", &self.version)
            .finish()
    }
}

/// An SSA Flag
#[derive(Clone)]
pub struct SSAFlag {
    flag: Flag,
    version: u32,
}

impl SSAFlag {
    pub fn new(flag: Flag, version: u32) -> Self {
        SSAFlag { flag, version }
    }
}

impl std::fmt::Debug for SSAFlag {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("SSAFlag")
            .field("flag", &self.flag)
            .field("version", &self.version)
            .finish()
    }
}

#[derive(Clone)]
pub struct RegisterStack {
    arch: CoreArchitecture,
    index: u32,
}

impl RegisterStack {
    pub fn new(arch: CoreArchitecture, index: u32) -> Self {
        RegisterStack { arch, index }
    }

    /// Get the name of this register
    pub fn name(&self) -> String {
        if self.index & 0x8000_0000 > 0 {
            format!("temp{}", self.index & 0x7fff_ffff)
        } else {
            self.arch
                .get_reg_name(self.index)
                .to_string_lossy()
                .to_string()
        }
    }
}

impl std::fmt::Display for RegisterStack {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl std::fmt::Debug for RegisterStack {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.name())
        /*
        f.debug_struct("RegisterStack")
            .field("name", &self.name())
            .finish()
            */
    }
}

/*
#[repr(u32)]
pub enum VariableSourceType {
    Stack,
    Register,
    Flag
}

impl VariableSourceType {
    pub fn from(val: u32) -> Self {
        match val {
            BNVariableSourceType_StackVariableSourceType    => VariableSourceType::Stack,
            BNVariableSourceType_RegisterVariableSourceType => VariableSourceType::Register,
            BNVariableSourceType_FlagVariableSourceType     => VariableSourceType::Flag,
        }
    }
}
*/

#[derive(Clone)]
#[allow(dead_code)]
pub struct Variable {
    func: Function,
    pub var: BNVariable,
    type_: Arc<BinjaType>,
}

impl Variable {
    pub fn new(func: Function, var: BNVariable) -> Self {
        // Get the BNType from the BNVariable
        let var_type_conf = unsafe { BNGetVariableType(func.handle(), &var) };

        // Create the BinjaType from the type handle
        let type_ = Arc::new(BinjaType::new(var_type_conf.type_));

        Variable { func, var, type_ }
    }

    /*
    pub fn source_type(&self) -> VariableSourceType {
        VariableSourceType::from(self.var.type_)
    }
    */

    pub fn index(&self) -> u32 {
        self.var.index
    }

    pub fn storage(&self) -> i64 {
        self.var.storage
    }

    pub fn identifier(&self) -> u64 {
        unsafe { BNToVariableIdentifier(&self.var) }
    }

    pub fn name(&self) -> BinjaStr {
        unsafe { BinjaStr::new(BNGetVariableName(self.func.handle(), &self.var)) }
    }

    pub fn bnvar(&self) -> BNVariable {
        self.var
    }
}

impl std::fmt::Display for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "<var {}>", self.name())
    }
}

impl std::fmt::Debug for Variable {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "<var {} {:?}>", self.name(), self.var)
        /*
        f.debug_struct("RegisterStack")
            .field("name", &self.name())
            .finish()
            */
    }
}

#[derive(Clone, Copy, Debug)]
pub enum AnalysisLevel {
    Low,
    Medium,
    High,
}

#[derive(Clone)]
pub struct SSAVariable {
    pub var: Variable,
    pub version: usize,
}

impl SSAVariable {
    pub fn new(var: Variable, version: usize) -> Self {
        Self { var, version }
    }

    /// Get the HLIL instruction where this SSAVariable is defined. If the SSA variable
    /// is version 0, attempt to get the xrefs of the found function and return the
    /// instruction where this variable originates
    pub fn hlil_definition(&self, bv: &BinaryView) -> Result<Vec<HighLevelILInstruction>> {
        // For version 0, but not arg, then there isn't another definition for this var
        if self.version == 0 && !self.var.name().contains("arg") {
            return Ok(Vec::new());
        }

        // If version 0, but has arg in the name, then attempt to xref the function and
        // get the parameters
        if self.version == 0 {
            // Initialize the resulting Vec
            let mut res = Vec::new();

            // print!("Xref func: {:#x}\n", self.var.func.start());

            // Get all of the HLILSSA xrefs of this function
            // let xrefs: Vec<_> = bv.get_code_refs(self.var.func.start()).iter()
            let mut xrefs: Vec<_> = Vec::new();

            for xref in bv.get_code_refs(self.var.func.start()).iter() {
                let res = xref.hlilssa()?;
                if !matches!(&*res.operation, HighLevelILOperation::Nop {}) {
                    xrefs.push(res);
                }

                // HLIL xrefs can result in a `nop`. For a bit better coverage,
                // attempt to look for a `Call` or `CallSsa` that directly calls
                // the xref function
                use self::HighLevelILOperation::{Call, CallSsa, ConstPtr};

                // let expr_index = xref.hlilssa().ok().unwrap().expr_index;
                let expr_count = xref.func.hlilssa()?.expr_count();
                for curr_expr_index in 0..expr_count {
                    let hlilssa = xref.func.hlilssa()?;
                    // let curr_expr_index = expr_index + i;
                    let expr = HighLevelILInstruction::from_expr(hlilssa, curr_expr_index, None)?
                        .ssa_form()?;

                    if let Call { dest, .. } | CallSsa { dest, .. } = &*expr.operation {
                        if let ConstPtr { constant } = &*dest.operation {
                            if *constant == self.var.func.start() {
                                info!("{:#x} -- {:#x}", expr.address, xref.address());
                                xrefs.push(expr);
                            }
                        }
                    }
                }
            }

            debug!(
                "XREF for {} from {:#x}: {}\n",
                self,
                self.var.func.start(),
                xrefs.len()
            );

            // For all HLIL instructions, traverse the graph of operations until we find
            // a CallSsa instruction, which contains the parameters for the Call. Once
            // found, grab the HLIL instruction for the same index as the current variable
            //
            // Example:
            //  Curent: SSAVar arg3#0
            //  Result: All params[2] from all xrefs to the function holding arg3#0
            'xref: for curr_xref in xrefs {
                debug!("Checking xref: {}\n", curr_xref);

                // Resulting variable that is rewritten on each loop iteration
                let mut xref = curr_xref.clone();
                let mut count = 0;
                loop {
                    debug!(
                        "[{}][{:#x}->{:#x}] Graph {}",
                        count,
                        self.var.func.start(),
                        xref.address,
                        xref
                    );
                    debug!(
                        "[{}][{:#x}->{:#x}] Graph {:?}",
                        count,
                        self.var.func.start(),
                        xref.address,
                        xref
                    );
                    match *xref.operation {
                        HighLevelILOperation::Assign { src, .. } => {
                            xref = src;
                        }
                        HighLevelILOperation::AssignUnpack { src, .. } => {
                            xref = src;
                        }
                        HighLevelILOperation::VarInitSsa { src, .. } => {
                            xref = src;
                        }
                        HighLevelILOperation::DerefSsa { src, .. } => {
                            xref = src;
                        }
                        // HighLevelILOperation::Nop { .. } => { Nop doesn't go anywhere, this xref isn't useful break; }
                        HighLevelILOperation::CallSsa { ref params, .. }
                        | HighLevelILOperation::Call { ref params, .. } => {
                            // Found the Call operation, attempt to get the variable

                            // Get the parameter number from the arg name
                            // Convert arg3 -> 3
                            let param_index: usize =
                                self.var.name().replace("arg", "").parse().unwrap_or(!0);

                            debug!("Checking param: {}", self.var.name());

                            // arg1 is the first function argument, so we need to subtract
                            // one to get the params index
                            if let Some(index) = param_index.checked_sub(1) {
                                debug!("Checking index: {}", index);

                                // Attempt to get the parameter from the found params
                                // If found, add the found instruction to the result
                                if let Some(instr) = params.get(index) {
                                    if matches!(&*instr.operation, HighLevelILOperation::Var { .. })
                                    {
                                    }
                                    debug!("Checking instr: {}", instr);
                                    debug!("Checking instr: {:?}", instr);
                                    res.push(instr.clone());
                                }
                            }
                            break;
                        }
                        HighLevelILOperation::ConstPtr { constant: _ } => {
                            //let symbol = bv.get_symbol_at(constant);
                            //if let Ok(sym) = &symbol {
                            //print!("Symbol found: {}\n", sym);
                            //}
                            break;
                        }
                        HighLevelILOperation::Add {
                            ref left,
                            ref right,
                        } => {
                            res.push(left.clone());
                            res.push(right.clone());
                        }
                        HighLevelILOperation::AssignMemSsa { .. }
                        | HighLevelILOperation::Tailcall { .. }
                        | HighLevelILOperation::Sub { .. }
                        | HighLevelILOperation::UnsignedLessThan { .. }
                        | HighLevelILOperation::VarSsa { .. }
                        | HighLevelILOperation::Const { .. } => {
                            debug!("Pushing xref: {:?}\n", &xref);
                            res.push(xref);
                            break;
                        }
                        _ => {
                            eprint!(
                                "Unknown xref: {:?} FROM\n{:#x} -> {:#x}: {:?}\n",
                                xref.operation_name(),
                                self.var.func.start(),
                                curr_xref.address,
                                curr_xref.operation_name()
                            );

                            continue 'xref;

                            // return Err(anyhow!("Unknown operation: {:#x} {:?}\n", xref.address, xref.operation));
                        }
                    }

                    count += 1;
                }
            }

            return Ok(res);
        }

        // Get the HLILSSA function for the variable
        let hlilssa = self.var.func.hlil()?.ssa_form()?;

        // Construct the BNVariable
        let var = binja_sys::BNVariable {
            type_: self.var.var.type_,
            index: self.var.var.index,
            storage: self.var.var.storage,
        };

        // Get the expression index for the definition of this ssa variable
        let expr_index =
            unsafe { BNGetHighLevelILSSAVarDefinition(hlilssa.handle(), &var, self.version) };

        // Return the new HLIL instruction
        Ok(vec![HighLevelILInstruction::from_expr(
            hlilssa, expr_index, None,
        )?])
    }

    /// Get the MLIL instruction where this SSAVariable is defined. If the SSA variable
    /// is version 0, attempt to get the xrefs of the found function and return the
    /// instruction where this variable originates
    pub fn mlil_definition(&self, bv: &BinaryView) -> Result<Vec<MediumLevelILInstruction>> {
        // For version 0, but not arg, then there isn't another definition for this var
        if self.version == 0 && !self.var.name().contains("arg") {
            return Ok(Vec::new());
        }

        // If version 0, but has arg in the name, then attempt to xref the function and
        // get the parameters
        if self.version == 0 {
            // Initialize the resulting Vec
            let mut res = Vec::new();

            // print!("Xref func: {:#x}\n", self.var.func.start());

            // Get all of the MLILSSA xrefs of this function
            // let xrefs: Vec<_> = bv.get_code_refs(self.var.func.start()).iter()
            let mut xrefs: Vec<_> = Vec::new();

            // Get the MLILSSA version of each XREF
            for xref in bv.get_code_refs(self.var.func.start()).iter() {
                xrefs.push(xref.mlilssa()?);
            }

            debug!(
                "XREF for {} from {:#x}: {}\n",
                self,
                self.var.func.start(),
                xrefs.len()
            );

            // For all HLIL instructions, traverse the graph of operations until we find
            // a CallSsa instruction, which contains the parameters for the Call. Once
            // found, grab the HLIL instruction for the same index as the current variable
            //
            // Example:
            //  Curent: SSAVar arg3#0
            //  Result: All params[2] from all xrefs to the function holding arg3#0
            for curr_xref in xrefs {
                debug!("Checking xref: {}\n", curr_xref);

                // Resulting variable that is rewritten on each loop iteration
                let xref = curr_xref.clone();
                let count = 0;
                loop {
                    debug!(
                        "[{}][{:#x}->{:#x}] Graph {}",
                        count,
                        self.var.func.start(),
                        xref.address,
                        xref
                    );
                    debug!(
                        "[{}][{:#x}->{:#x}] Graph {:?}",
                        count,
                        self.var.func.start(),
                        xref.address,
                        xref
                    );
                    match *xref.operation {
                        MediumLevelILOperation::CallSsa { ref params, .. }
                        | MediumLevelILOperation::Call { ref params, .. } => {
                            // Found the Call operation, attempt to get the variable

                            // Get the parameter number from the arg name
                            // Convert arg3 -> 3
                            let param_index: usize =
                                self.var.name().replace("arg", "").parse().unwrap_or(!0);

                            debug!("Checking param: {}", self.var.name());

                            // arg1 is the first function argument, so we need to subtract
                            // one to get the params index
                            if let Some(index) = param_index.checked_sub(1) {
                                debug!("Checking index: {}", index);

                                // Attempt to get the parameter from the found params
                                // If found, add the found instruction to the result
                                if let Some(instr) = params.get(index) {
                                    if matches!(
                                        &*instr.operation,
                                        MediumLevelILOperation::Var { .. }
                                    ) {}
                                    debug!("Checking instr: {}", instr);
                                    debug!("Checking instr: {:?}", instr);
                                    res.push(instr.clone());
                                }
                            }
                            break;
                        }
                        MediumLevelILOperation::ConstPtr { constant: _ } => {
                            //let symbol = bv.get_symbol_at(constant);
                            //if let Ok(sym) = &symbol {
                            //print!("Symbol found: {}\n", sym);
                            //}
                            break;
                        }
                        /*
                        MediumLevelILOperation::Tailcall { .. } |
                        MediumLevelILOperation::Sub { .. } |
                        MediumLevelILOperation::UnsignedLessThan { .. } |
                        MediumLevelILOperation::VarSsa { .. } |
                        MediumLevelILOperation::Const { .. }
                        => {
                            debug!("Pushing xref: {:?}\n", &xref);
                            res.push(xref);
                            break;
                        }
                        */
                        _ => {
                            eprint!(
                                "Unknown xref: {:?} FROM\n{:#x} -> {:#x}: {:?}\n",
                                xref.operation_name(),
                                self.var.func.start(),
                                curr_xref.address,
                                curr_xref.operation_name()
                            );
                            return Err(anyhow!(
                                "Unknown operation: {:#x} {:?}\n",
                                xref.address,
                                xref.operation
                            ));
                        }
                    }
                }
            }

            return Ok(res);
        }

        // Get the MLILSSA function for the variable
        let mlilssa = self.var.func.mlilssa()?;

        // Construct the BNVariable
        let var = binja_sys::BNVariable {
            type_: self.var.var.type_,
            index: self.var.var.index,
            storage: self.var.var.storage,
        };

        // Get the expression index for the definition of this ssa variable
        let func_index =
            unsafe { BNGetMediumLevelILSSAVarDefinition(mlilssa.handle(), &var, self.version) };

        // Return the new HLIL instruction
        Ok(vec![MediumLevelILInstruction::from_func_index(
            mlilssa, func_index,
        )])
    }

    /// Get the MLIL instruction(s) where this SSAVariable is used.
    pub fn mlil_uses(&self) -> Result<Vec<MediumLevelILInstruction>> {
        // Construct the BNVariable
        let var = binja_sys::BNVariable {
            type_: self.var.var.type_,
            index: self.var.var.index,
            storage: self.var.var.storage,
        };

        let mut res = Vec::new();

        // Get the MLILSSA function for the variable
        let mlilssa = self.var.func.mlilssa()?;

        // Get the expression index for the definition of this ssa variable
        unsafe {
            let mut count = 0;
            let version = self.version;
            let uses = BNGetMediumLevelILSSAVarUses(mlilssa.handle(), &var, version, &mut count);
            let indexes_slice = std::slice::from_raw_parts(uses, count as usize);
            for index in indexes_slice {
                let instr = MediumLevelILInstruction::from_func_index(mlilssa.clone(), *index);
                res.push(instr);
            }
        };

        // Return the new MLIL instructions
        Ok(res)
    }
}

impl std::fmt::Display for SSAVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "<ssavar: {} version: {}>", self.var.name(), self.version)
    }
}

impl std::fmt::Debug for SSAVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "<ssavar: {} version: {}>", self.var.name(), self.version)
        /*
        f.debug_struct("RegisterStack")
            .field("name", &self.name())
            .finish()
            */
    }
}

#[derive(Clone)]
pub struct SSAVariableDestSrc {
    pub dest: SSAVariable,
    pub src: SSAVariable,
}

impl SSAVariableDestSrc {
    pub fn new(dest: SSAVariable, src: SSAVariable) -> Self {
        Self { dest, src }
    }
}

impl std::fmt::Display for SSAVariableDestSrc {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "<ssa dest: {:?} src: {:?}>", self.dest, self.src)
    }
}

impl std::fmt::Debug for SSAVariableDestSrc {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "<ssa dest: {:?} src: {:?}>", self.dest, self.src)
        /*
        f.debug_struct("RegisterStack")
            .field("name", &self.name())
            .finish()
            */
    }
}

#[derive(Clone)]
pub struct Intrinsic {
    arch: CoreArchitecture,
    index: u32,
}

impl Intrinsic {
    pub fn new(arch: CoreArchitecture, index: u32) -> Self {
        Self { arch, index }
    }

    pub fn name(&self) -> BinjaStr {
        self.arch.get_intrinsic_name(self.index)
    }
}

impl std::fmt::Display for Intrinsic {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "<intrinsic: {}>", self.name())
    }
}

impl std::fmt::Debug for Intrinsic {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "<intrinsic: {}>", self.name())
        /*
        f.debug_struct("RegisterStack")
            .field("name", &self.name())
            .finish()
            */
    }
}

#[derive(Clone)]
pub struct GotoLabel {
    func: HighLevelILFunction,
    id: u64,
}

impl GotoLabel {
    pub fn new(func: HighLevelILFunction, id: u64) -> Self {
        Self { func, id }
    }

    pub fn name(&self) -> BinjaStr {
        unsafe { BinjaStr::from(BNGetGotoLabelName(self.func.function().handle(), self.id)) }
    }
}

impl std::fmt::Display for GotoLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "<label: {}>", self.name())
    }
}

impl std::fmt::Debug for GotoLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "<label: {}>", self.name())
        /*
        f.debug_struct("RegisterStack")
            .field("name", &self.name())
            .finish()
            */
    }
}

#[derive(Clone)]
#[allow(dead_code)]
pub struct ConstantData {
    func: Function,
    type_: u64,
    value: u64,
    size: usize,
}

impl ConstantData {
    pub fn new(func: Function, type_: u64, value: u64, size: usize) -> Self {
        Self {
            func,
            type_,
            value,
            size,
        }
    }
}

impl std::fmt::Debug for ConstantData {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "<const data: TODO>")
    }
}
