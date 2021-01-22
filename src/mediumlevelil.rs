//! Provides Medium Level IL analysis
#![allow(non_upper_case_globals)]
#![allow(unused_assignments)]

use core::*;

use anyhow::Result;

use std::slice;
use std::mem;
use std::fmt;
use std::sync::Arc;
use std::collections::HashMap;

use crate::unsafe_try;
use crate::function::Function;
use crate::architecture::CoreArchitecture;
use crate::il::{SSAVariable, SSAVariableDestSrc, Variable, Intrinsic};
use crate::instruction::InstructionTextToken;
use crate::traits::{FunctionTrait, BasicBlockTrait};
use crate::wrappers::{BinjaMediumLevelILFunction, BinjaFunction, BinjaBasicBlock};
use crate::highlevelil::HighLevelILInstruction;

#[derive(Clone)]
pub struct MediumLevelILFunction {
    handle: Arc<BinjaMediumLevelILFunction>,
    func: Arc<BinjaFunction>,
}

unsafe impl Send for MediumLevelILFunction {}
unsafe impl Sync for MediumLevelILFunction {}

impl MediumLevelILFunction {
    pub fn new(func: Arc<BinjaFunction>) -> Result<MediumLevelILFunction> {
        let handle = unsafe_try!(BNGetFunctionMediumLevelIL(**func))?;
        Ok(MediumLevelILFunction{ 
            handle: Arc::new(BinjaMediumLevelILFunction::new(handle)), 
            func,
        })
    }

    pub fn handle(&self) -> *mut BNMediumLevelILFunction {
        **self.handle
    }

    /// Get the owner function for this MLIL function
    pub fn function(&self) -> Function {
        Function::from_arc(self.func.clone())
    }

    /// Get the address of this function
    pub fn address(&self) -> u64 {
        unsafe { BNMediumLevelILGetCurrentAddress(self.handle()) }
    }

    /// Get the number of instructions in this function
    pub fn len(&self) -> u64 {
        unsafe { BNGetMediumLevelILInstructionCount(self.handle()) }
    }


    /// Get the architecture for this function
    pub fn arch(&self) -> Result<CoreArchitecture> {
        CoreArchitecture::new_from_func(**self.func)
    }

    fn get_index_for_instruction(&self, i: u64) -> u64 {
        unsafe { BNGetMediumLevelILIndexForInstruction(self.handle(), i) }
    }

    pub fn ssa_form(&self) -> Result<Self> {
        let handle = unsafe_try!(BNGetMediumLevelILSSAForm(self.handle()))?;
        Ok(MediumLevelILFunction { 
            handle: Arc::new(BinjaMediumLevelILFunction::new(handle)), 
            func: self.func.clone(),
        })
    }

    /*
    pub fn get_medium_level_il_expr_index(&self, i: usize) -> usize {
        unsafe {
            let result = BNGetMediumLevelILExprIndex(self.handle, i);
            return result;
            if result < self.function().unwrap().medium_level_il().count() {
                return result
            }
        }
        panic!("Cannot create MLIL expr index: {} {:?}", self, i);
    }
    */
}

impl fmt::Display for MediumLevelILFunction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl fmt::Debug for MediumLevelILFunction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut result = f.debug_struct("MLILFunction");
        result.field("address", &format_args!("{:#x}", self.address()));
        result.field("name", &self.function().name());
        result.finish()
    }
}

impl FunctionTrait for MediumLevelILFunction {
    type Ins   = MediumLevelILInstruction;
    type Block = MediumLevelILBasicBlock;
    type Func  = MediumLevelILFunction;

    /// Retrieve a MediumLevelILInstruction for a given index
    fn instruction(&self, i: u64) -> Result<Self::Ins> {
        let res = MediumLevelILInstruction::from_func_index(self.clone(), i);
        Ok(res)
    }

    fn blocks(&self) -> Vec<Self::Block> {
        let mut count = 0;

        unsafe { 
            let blocks = BNGetMediumLevelILBasicBlockList(self.handle(), &mut count);
            trace!("Function {} returned {} blocks", self, count);

            let blocks_slice = slice::from_raw_parts(blocks, count as usize);

            let result = blocks_slice.iter()
                                     .map(|&b| MediumLevelILBasicBlock::new(b, self.clone()).unwrap())
                                     .collect();

            BNFreeBasicBlockList(blocks, count);
            result
        }
    }

    fn ssa_form(&self) -> Result<MediumLevelILFunction> {
        self.ssa_form()
    }

    /// Construct the text for a given MediumLevelILInstruction index
    fn text(&self, i: u64) -> Result<String> {
        let mut count = 0;

        unsafe { 
            // Initialize pointer to get data from BNGetMediumLevelILInstructionText
            let mut list = mem::zeroed();

            BNGetMediumLevelILInstructionText(self.handle(), **self.func, self.arch()?.handle(), i, 
                                           &mut list, &mut count);

            if list.is_null() {
                return Err(anyhow!("Failed to retrieve MLILInstruction tokens"));
            }

            let list_slice = slice::from_raw_parts(list, count as usize);
            
            let result: Vec<InstructionTextToken> = list_slice.iter()
                                   .map(|&l| InstructionTextToken::new_from_token(l))
                                   .collect();

            BNFreeInstructionText(list, count);

            Ok(result.iter().fold(String::new(), |acc, x| { format!("{}{}", acc, x.text) }))
        }
    }
}

#[derive(Clone)]
pub struct MediumLevelILBasicBlock {
    handle: Arc<BinjaBasicBlock>,
    func: MediumLevelILFunction
}

impl MediumLevelILBasicBlock {
    pub fn new(handle: *mut BNBasicBlock, func: MediumLevelILFunction) -> Result<MediumLevelILBasicBlock> {
        let handle = unsafe_try!(BNNewBasicBlockReference(handle))?;
        Ok(MediumLevelILBasicBlock{ handle: Arc::new(BinjaBasicBlock::new(handle)), func })
    }
}

impl BasicBlockTrait for MediumLevelILBasicBlock {
    type Ins = MediumLevelILInstruction;
    type Func = MediumLevelILFunction;

    fn handle(&self) -> *mut BNBasicBlock {
        **self.handle
    }

    fn func(&self) -> Option<&Self::Func> {
        Some(&self.func)
    }
}

unsafe impl Send for MediumLevelILBasicBlock {}
unsafe impl Sync for MediumLevelILBasicBlock {}

pub struct MediumLevelILInstruction {
    pub operation: Box<MediumLevelILOperation>,
    pub source_operand: u32,
    pub size: u64,
    pub operands: [u64; 5usize],
    pub address: u64,
    pub function: MediumLevelILFunction,
    pub expr_index: u64,
    pub instr_index: Option<u64>,
    pub text: Result<String>
}

unsafe impl Send for MediumLevelILInstruction {}
unsafe impl Sync for MediumLevelILInstruction {}

impl MediumLevelILInstruction {
    /// Get the MLIL instruction from the given `func` at the given `instr_index`
    pub fn from_func_index(func: MediumLevelILFunction, instr_index: u64) -> MediumLevelILInstruction { 
        // Get the raw index for the given instruction index
        let expr_index = func.get_index_for_instruction(instr_index);

        MediumLevelILInstruction::from_expr(func, expr_index, Some(instr_index))
    }

    /// Get the MLIL instruction from the given internal `expr` at the given `instr_index`
    pub fn from_expr(func: MediumLevelILFunction, expr_index: u64, mut instr_index: Option<u64>) 
            -> MediumLevelILInstruction { 
        // Get the IL for the given index
        let instr = unsafe { BNGetMediumLevelILByIndex(func.handle(), expr_index) };
        
        if instr_index.is_none() {
            instr_index = Some(unsafe {
                BNGetMediumLevelILInstructionForExpr(func.handle(), expr_index)
            });
        }

        // If we have the instruction index, grab the text for that instruction
        let text = if let Some(index) = instr_index {
            func.text(index)
        } else {
            Err(anyhow!("text() for None from_expr unimpl"))// unimplemented!()
        };

        MediumLevelILInstruction {
            operation: Box::new(MediumLevelILOperation::from_instr(instr, &func, expr_index)),
            source_operand: instr.sourceOperand,
            size: instr.size,
            operands: instr.operands,
            address: instr.address,
            function: func,
            expr_index,
            instr_index,
            text
        }
    }

    /// Convert MLIL instruction into MLIL SSA (Alias for ssa_form)
    pub fn ssa(&self) -> Result<MediumLevelILInstruction> {
        self.ssa_form()
    }

    /// Convert MLIL instruction into MLIL SSA
    pub fn ssa_form(&self) -> Result<MediumLevelILInstruction> {
        let func_ssa = self.function.ssa_form()?;
        let expr_index = unsafe { 
            BNGetMediumLevelILSSAExprIndex(self.function.handle(), self.expr_index)
        };
        Ok(MediumLevelILInstruction::from_expr(func_ssa, expr_index, self.instr_index))
    }

    /// Get the HLIL instruction for this MLIL instruction
    pub fn high_level_il(&self) -> Result<HighLevelILInstruction> {
        let hlil_expr_index = unsafe {
            BNGetHighLevelILExprIndex(self.function.handle(), self.expr_index) 
        };

        let hlil = self.function.function().hlil()?;

        HighLevelILInstruction::from_expr(hlil, hlil_expr_index, None)
    }

    /// Get the HLIL instruction for this MLIL instruction. 
    /// Alias for `self.high_level_il()`
    pub fn hlil(&self) -> Result<HighLevelILInstruction> {
        self.high_level_il()
    }

    /// Get the HLILSSA instruction for this MLIL instruction. 
    /// Alias for `self.high_level_il().ssa_form()`
    pub fn hlilssa(&self) -> Result<HighLevelILInstruction> {
        self.high_level_il()?.ssa_form()
    }
}

impl fmt::Display for MediumLevelILInstruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.text {
            Ok(text) => write!(f, "{} ", text),
            Err(_) => write!(f, "[{}:{}] Invalid text!", self.function, self.address)
        }
    }
}

impl fmt::Debug for MediumLevelILInstruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("MediumLevelILInstruction")
            .field("Operation", &self.operation)
            .finish()
    }
}

#[derive(Debug)]
pub enum MediumLevelILOperation {
    Nop { },
    SetVar { dest: Variable, src: MediumLevelILInstruction, },
    SetVarField { dest: Variable, offset: u64, src: MediumLevelILInstruction, },
    SetVarSplit { high: Variable, low: Variable, src: MediumLevelILInstruction, },
    Load { src: MediumLevelILInstruction, },
    LoadStruct { src: MediumLevelILInstruction, offset: u64, },
    Store { dest: MediumLevelILInstruction, src: MediumLevelILInstruction, },
    StoreStruct { dest: MediumLevelILInstruction, offset: u64, src: MediumLevelILInstruction, },
    Var { src: Variable, },
    VarField { src: Variable, offset: u64, },
    VarSplit { high: Variable, low: Variable, },
    AddressOf { src: Variable, },
    AddressOfField { src: Variable, offset: u64, },
    Const { constant: u64, },
    ConstPtr { constant: u64, },
    ExternPtr { constant: u64, offset: u64, },
    // FloatConst { constant: float, },
    Import { constant: u64, },
    Add { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    Adc { left: MediumLevelILInstruction, right: MediumLevelILInstruction, carry: MediumLevelILInstruction, },
    Sub { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    Sbb { left: MediumLevelILInstruction, right: MediumLevelILInstruction, carry: MediumLevelILInstruction, },
    And { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    Or { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    Xor { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    Lsl { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    Lsr { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    Asr { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    Rol { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    Rlc { left: MediumLevelILInstruction, right: MediumLevelILInstruction, carry: MediumLevelILInstruction, },
    Ror { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    Rrc { left: MediumLevelILInstruction, right: MediumLevelILInstruction, carry: MediumLevelILInstruction, },
    Mul { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    MuluDp { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    MulsDp { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    Divu { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    DivuDp { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    Divs { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    DivsDp { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    Modu { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    ModuDp { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    Mods { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    ModsDp { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    Neg { src: MediumLevelILInstruction, },
    Not { src: MediumLevelILInstruction, },
    SignExtend { src: MediumLevelILInstruction, },
    ZeroExtend { src: MediumLevelILInstruction, },
    LowPart { src: MediumLevelILInstruction, },
    Jump { dest: MediumLevelILInstruction, },
    JumpTo { dest: MediumLevelILInstruction, targets: HashMap<u64, u64>, },
    RetHint { dest: MediumLevelILInstruction, },
    Call { output: Vec<Variable>, dest: MediumLevelILInstruction, params: Vec<MediumLevelILInstruction>, },
    CallUntyped { output: MediumLevelILInstruction, dest: MediumLevelILInstruction, params: MediumLevelILInstruction, stack: MediumLevelILInstruction, },
    CallOutput { dest: Vec<Variable>, },
    CallParam { src: Vec<Variable>, },
    Ret { src: Vec<MediumLevelILInstruction>, },
    Noret { },
    If { condition: MediumLevelILInstruction, true_: u64, false_: u64, },
    Goto { dest: u64, },
    Equals { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    NotEquals { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    SignedLessThan { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    UnsignedLessThan { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    SignedLessThanEquals { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    UnsignedLessThanEquals { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    SignedGreaterThanEquals { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    UnsignedGreaterThanEquals { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    SsignedGreaterThan { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    UnsignedGreaterThan { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    TestBit { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    BoolToInt { src: MediumLevelILInstruction, },
    AddOverflow { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    Syscall { output: Vec<Variable>, params: Vec<MediumLevelILInstruction>, },
    SyscallUntyped { output: MediumLevelILInstruction, params: MediumLevelILInstruction, stack: MediumLevelILInstruction, },
    Tailcall { output: Vec<Variable>, dest: MediumLevelILInstruction, params: Vec<MediumLevelILInstruction>, },
    TailcallUntyped { output: MediumLevelILInstruction, dest: MediumLevelILInstruction, params: MediumLevelILInstruction, stack: MediumLevelILInstruction, },
    Intrinsic { output: Vec<Variable>, intrinsic: Intrinsic, params: Vec<MediumLevelILInstruction>, },
    FreeVarSlot { dest: Variable, },
    Bp { },
    Trap { vector: u64, },
    Undef { },
    Unimpl { },
    UnimplMem { src: MediumLevelILInstruction, },
    Fadd { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    Fsub { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    Fmul { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    Fdiv { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    Fsqrt { src: MediumLevelILInstruction, },
    Fneg { src: MediumLevelILInstruction, },
    Fabs { src: MediumLevelILInstruction, },
    FloatToInt { src: MediumLevelILInstruction, },
    IntToFloat { src: MediumLevelILInstruction, },
    FloatConv { src: MediumLevelILInstruction, },
    RoundToInt { src: MediumLevelILInstruction, },
    Floor { src: MediumLevelILInstruction, },
    Ceil { src: MediumLevelILInstruction, },
    Ftrunc { src: MediumLevelILInstruction, },
    FcmpE { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    FcmpNe { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    FcmpLt { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    FcmpLe { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    FcmpGe { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    FcmpGt { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    FcmpO { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    FcmpUo { left: MediumLevelILInstruction, right: MediumLevelILInstruction, },
    SetVarSsa { dest: SSAVariable, src: MediumLevelILInstruction, },
    SetVarSsaField { prev: SSAVariableDestSrc, offset: u64, src: MediumLevelILInstruction, },
    SetVarSplitSsa { high: SSAVariable, low: SSAVariable, src: MediumLevelILInstruction, },
    SetVarAliased { prev: SSAVariableDestSrc, src: MediumLevelILInstruction, },
    SetVarAliasedField { prev: SSAVariableDestSrc, offset: u64, src: MediumLevelILInstruction, },
    VarSsa { src: SSAVariable, },
    VarSsaField { src: SSAVariable, offset: u64, },
    VarAliased { src: SSAVariable, },
    VarAliasedField { src: SSAVariable, offset: u64, },
    VarSplitSsa { high: SSAVariable, low: SSAVariable, },
    CallSsa { output: MediumLevelILInstruction, dest: MediumLevelILInstruction, params: Vec<MediumLevelILInstruction>, src_memory: u64, },
    CallUntypedSsa { output: MediumLevelILInstruction, dest: MediumLevelILInstruction, params: MediumLevelILInstruction, stack: MediumLevelILInstruction, },
    SyscallSsa { output: MediumLevelILInstruction, params: Vec<MediumLevelILInstruction>, src_memory: u64, },
    SyscallUntypedSsa { output: MediumLevelILInstruction, params: MediumLevelILInstruction, stack: MediumLevelILInstruction, },
    TailcallSsa { output: MediumLevelILInstruction, dest: MediumLevelILInstruction, params: Vec<MediumLevelILInstruction>, src_memory: u64, },
    TailcallUntypedSsa { output: MediumLevelILInstruction, dest: MediumLevelILInstruction, params: MediumLevelILInstruction, stack: MediumLevelILInstruction, },
    CallParamSsa { src_memory: u64, src: Vec<SSAVariable>, },
    CallOutputSsa { dest_memory: u64, dest: Vec<SSAVariable>, },
    LoadSsa { src: MediumLevelILInstruction, src_memory: u64, },
    LoadStructSsa { src: MediumLevelILInstruction, offset: u64, src_memory: u64, },
    StoreSsa { dest: MediumLevelILInstruction, dest_memory: u64, src_memory: u64, src: MediumLevelILInstruction, },
    StoreStructSsa { dest: MediumLevelILInstruction, offset: u64, dest_memory: u64, src_memory: u64, src: MediumLevelILInstruction, },
    IntrinsicSsa { output: Vec<SSAVariable>, intrinsic: Intrinsic, params: Vec<MediumLevelILInstruction>, },
    FreeVarSlotSsa { prev: SSAVariableDestSrc, },
    VarPhi { dest: SSAVariable, src: Vec<SSAVariable>, },
    MemPhi { dest_memory: u64, src_memory: Vec<u64>, },
}
impl MediumLevelILOperation {
    pub fn from_instr(instr: BNMediumLevelILInstruction, func: &MediumLevelILFunction, expr_index: u64)
            -> MediumLevelILOperation {
        let arch = func.arch().expect("Failed to get arch for LLIL").clone();
        let mut operand_index = 0;

        // Macros used to define each of the types of arguments in each operation
        macro_rules! expr {
            () => {{
                let res = MediumLevelILInstruction::from_expr(func.clone(), instr.operands[operand_index], None);
                operand_index += 1;
                res
            }}
        }

        macro_rules! int {
            () => {{
                let res = (instr.operands[operand_index] & 0x7fff_ffff) 
                    - (instr.operands[operand_index] & (1 << 63));
                operand_index += 1;
                res
            }}
        }

        macro_rules! expr_list {
            () => {{
                // Initialize the resulting instructions vec
                let mut instrs = Vec::new();
                let mut count = 0;
            
                unsafe { 
                    // Get the pointer to instruction indexes from binja core
                    let operands = BNMediumLevelILGetOperandList(func.handle(), expr_index, 
                                                              operand_index as u64, &mut count);

                    operand_index += 1;

                    // Get the slice from the found pointer
                    let operands_slice = slice::from_raw_parts(operands, count as usize);

                    // Create each instruction
                    for op in operands_slice {
                        let i = MediumLevelILInstruction::from_expr(func.clone(), *op, None);
                        instrs.push(i);
                    }

                    // Free the binja core pointer
                    BNMediumLevelILFreeOperandList(operands);
                }

                instrs
            }}
        }

        macro_rules! target_map {
            () => {{
                // Initialize the target map
                let mut target_map = HashMap::new();
                let mut count = 0;

                unsafe { 
                    // Get the operands from the binja core
                    let operands = BNMediumLevelILGetOperandList(func.handle(), expr_index, 
                                                              operand_index as u64, &mut count);

                    operand_index += 1;

                    // Cast the result from binja core into the slice of operands
                    let operands_slice = slice::from_raw_parts(operands, count as usize);

                    // Extract the key, value pairs from the found slice
                    for i in (0..count).step_by(2) {
                        let key   = operands_slice[i as usize];
                        let value = operands_slice[i as usize + 1];
                        target_map.insert(key, value);
                    }

                    // Free the operands
                    BNMediumLevelILFreeOperandList(operands);
                }

                target_map
            }}
        }

        macro_rules! int_list {
            () => {{
                // Generate the int list from the binja core
                let mut count = 0;
                let mut int_list = Vec::new();
            
                unsafe { 
                    let operands = BNMediumLevelILGetOperandList(func.handle(), expr_index, 
                                                            operand_index as u64, &mut count);

                    operand_index += 1;

                    let operands_slice = slice::from_raw_parts(operands, count as usize);

                    for i in 0..count {
                        int_list.push(operands_slice[i as usize]);
                    }

                    BNMediumLevelILFreeOperandList(operands);
                }

                int_list
            }}
        }

        macro_rules! var {
            () => {{
                let bnvar = unsafe { BNFromVariableIdentifier(instr.operands[operand_index]) };
                operand_index += 1;
                let res = Variable::new(func.function(), bnvar);
                res
            }}
        }

        macro_rules! var_ssa {
            () => {{
                let var = var!();
                let version = instr.operands[operand_index] as u32;
                operand_index += 1;
                SSAVariable::new(var, version)
            }}
        }

        macro_rules! var_list {
            () => {{
                let mut count = 0;
                let mut vars = Vec::new();
            
                unsafe { 
                    let operands = BNMediumLevelILGetOperandList(func.handle(), expr_index, 
                                                                 operand_index as u64, &mut count);

                    operand_index += 1;

                    let operands_slice = slice::from_raw_parts(operands, count as usize);

                    for i in 0..count {
                        let id      = operands_slice[i as usize];
                        let bnvar = BNFromVariableIdentifier(id);
                        let var = Variable::new(func.function(), bnvar);
                        vars.push(var);
                    }
                    BNMediumLevelILFreeOperandList(operands);
                }

                vars
            }}
        }

        macro_rules! var_ssa_list {
            () => {{
                let mut count = 0;
                let mut vars = Vec::new();
            
                unsafe { 
                    let operands = BNMediumLevelILGetOperandList(func.handle(), expr_index, 
                                                                 operand_index as u64, &mut count);

                    operand_index += 1;

                    let operands_slice = slice::from_raw_parts(operands, count as usize);

                    for i in (0..count).step_by(2) {
                        let id      = operands_slice[i as usize];
                        let version = operands_slice[i as usize + 1] as u32;
                        let bnvar = BNFromVariableIdentifier(id);
                        let var = Variable::new(func.function(), bnvar);
                        vars.push(SSAVariable::new(var, version));
                    }

                    BNMediumLevelILFreeOperandList(operands);
                }

                vars
            }}
        }
        
        macro_rules! intrinsic {
            () => {{
                let res = Intrinsic::new(arch.clone(), instr.operands[operand_index] as u32);
                operand_index += 1;
                res
            }}
        }

        macro_rules! var_ssa_dest_and_src {
            () => {{
                let var = var!();
                let dest_version = instr.operands[operand_index];
                let src_version  = instr.operands[operand_index + 1];
                operand_index += 2;
                let dest_ssa = SSAVariable::new(var.clone(), dest_version as u32);
                let src_ssa  = SSAVariable::new(var, src_version as u32);
                SSAVariableDestSrc::new(dest_ssa, src_ssa)
            }}
        }

        match instr.operation {
            BNMediumLevelILOperation_MLIL_NOP => {
                MediumLevelILOperation::Nop {
                    
                }
            }
            BNMediumLevelILOperation_MLIL_SET_VAR => {
                let dest = var!();
                let src = expr!();
                MediumLevelILOperation::SetVar {
                    dest, src
                }
            }
            BNMediumLevelILOperation_MLIL_SET_VAR_FIELD => {
                let dest = var!();
                let offset = int!();
                let src = expr!();
                MediumLevelILOperation::SetVarField {
                    dest, offset, src
                }
            }
            BNMediumLevelILOperation_MLIL_SET_VAR_SPLIT => {
                let high = var!();
                let low = var!();
                let src = expr!();
                MediumLevelILOperation::SetVarSplit {
                    high, low, src
                }
            }
            BNMediumLevelILOperation_MLIL_LOAD => {
                let src = expr!();
                MediumLevelILOperation::Load {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_LOAD_STRUCT => {
                let src = expr!();
                let offset = int!();
                MediumLevelILOperation::LoadStruct {
                    src, offset
                }
            }
            BNMediumLevelILOperation_MLIL_STORE => {
                let dest = expr!();
                let src = expr!();
                MediumLevelILOperation::Store {
                    dest, src
                }
            }
            BNMediumLevelILOperation_MLIL_STORE_STRUCT => {
                let dest = expr!();
                let offset = int!();
                let src = expr!();
                MediumLevelILOperation::StoreStruct {
                    dest, offset, src
                }
            }
            BNMediumLevelILOperation_MLIL_VAR => {
                let src = var!();
                MediumLevelILOperation::Var {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_VAR_FIELD => {
                let src = var!();
                let offset = int!();
                MediumLevelILOperation::VarField {
                    src, offset
                }
            }
            BNMediumLevelILOperation_MLIL_VAR_SPLIT => {
                let high = var!();
                let low = var!();
                MediumLevelILOperation::VarSplit {
                    high, low
                }
            }
            BNMediumLevelILOperation_MLIL_ADDRESS_OF => {
                let src = var!();
                MediumLevelILOperation::AddressOf {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_ADDRESS_OF_FIELD => {
                let src = var!();
                let offset = int!();
                MediumLevelILOperation::AddressOfField {
                    src, offset
                }
            }
            BNMediumLevelILOperation_MLIL_CONST => {
                let constant = int!();
                MediumLevelILOperation::Const {
                    constant
                }
            }
            BNMediumLevelILOperation_MLIL_CONST_PTR => {
                let constant = int!();
                MediumLevelILOperation::ConstPtr {
                    constant
                }
            }
            BNMediumLevelILOperation_MLIL_EXTERN_PTR => {
                let constant = int!();
                let offset = int!();
                MediumLevelILOperation::ExternPtr {
                    constant, offset
                }
            }
            BNMediumLevelILOperation_MLIL_FLOAT_CONST => {
                unimplemented!();
                /*
                let constant = float!();
                MediumLevelILOperation::FloatConst {
                    constant
                }
                */
            }
            BNMediumLevelILOperation_MLIL_IMPORT => {
                let constant = int!();
                MediumLevelILOperation::Import {
                    constant
                }
            }
            BNMediumLevelILOperation_MLIL_ADD => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::Add {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_ADC => {
                let left = expr!();
                let right = expr!();
                let carry = expr!();
                MediumLevelILOperation::Adc {
                    left, right, carry
                }
            }
            BNMediumLevelILOperation_MLIL_SUB => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::Sub {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_SBB => {
                let left = expr!();
                let right = expr!();
                let carry = expr!();
                MediumLevelILOperation::Sbb {
                    left, right, carry
                }
            }
            BNMediumLevelILOperation_MLIL_AND => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::And {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_OR => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::Or {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_XOR => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::Xor {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_LSL => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::Lsl {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_LSR => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::Lsr {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_ASR => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::Asr {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_ROL => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::Rol {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_RLC => {
                let left = expr!();
                let right = expr!();
                let carry = expr!();
                MediumLevelILOperation::Rlc {
                    left, right, carry
                }
            }
            BNMediumLevelILOperation_MLIL_ROR => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::Ror {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_RRC => {
                let left = expr!();
                let right = expr!();
                let carry = expr!();
                MediumLevelILOperation::Rrc {
                    left, right, carry
                }
            }
            BNMediumLevelILOperation_MLIL_MUL => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::Mul {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_MULU_DP => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::MuluDp {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_MULS_DP => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::MulsDp {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_DIVU => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::Divu {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_DIVU_DP => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::DivuDp {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_DIVS => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::Divs {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_DIVS_DP => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::DivsDp {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_MODU => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::Modu {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_MODU_DP => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::ModuDp {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_MODS => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::Mods {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_MODS_DP => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::ModsDp {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_NEG => {
                let src = expr!();
                MediumLevelILOperation::Neg {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_NOT => {
                let src = expr!();
                MediumLevelILOperation::Not {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_SX => {
                let src = expr!();
                MediumLevelILOperation::SignExtend {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_ZX => {
                let src = expr!();
                MediumLevelILOperation::ZeroExtend {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_LOW_PART => {
                let src = expr!();
                MediumLevelILOperation::LowPart {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_JUMP => {
                let dest = expr!();
                MediumLevelILOperation::Jump {
                    dest
                }
            }
            BNMediumLevelILOperation_MLIL_JUMP_TO => {
                let dest = expr!();
                let targets = target_map!();
                MediumLevelILOperation::JumpTo {
                    dest, targets
                }
            }
            BNMediumLevelILOperation_MLIL_RET_HINT => {
                let dest = expr!();
                MediumLevelILOperation::RetHint {
                    dest
                }
            }
            BNMediumLevelILOperation_MLIL_CALL => {
                let output = var_list!();
                let dest = expr!();
                let params = expr_list!();
                MediumLevelILOperation::Call {
                    output, dest, params
                }
            }
            BNMediumLevelILOperation_MLIL_CALL_UNTYPED => {
                let output = expr!();
                let dest = expr!();
                let params = expr!();
                let stack = expr!();
                MediumLevelILOperation::CallUntyped {
                    output, dest, params, stack
                }
            }
            BNMediumLevelILOperation_MLIL_CALL_OUTPUT => {
                let dest = var_list!();
                MediumLevelILOperation::CallOutput {
                    dest
                }
            }
            BNMediumLevelILOperation_MLIL_CALL_PARAM => {
                let src = var_list!();
                MediumLevelILOperation::CallParam {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_RET => {
                let src = expr_list!();
                MediumLevelILOperation::Ret {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_NORET => {
                MediumLevelILOperation::Noret {
                    
                }
            }
            BNMediumLevelILOperation_MLIL_IF => {
                let condition = expr!();
                let true_ = int!();
                let false_ = int!();
                MediumLevelILOperation::If {
                    condition, true_, false_
                }
            }
            BNMediumLevelILOperation_MLIL_GOTO => {
                let dest = int!();
                MediumLevelILOperation::Goto {
                    dest
                }
            }
            BNMediumLevelILOperation_MLIL_CMP_E => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::Equals {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_CMP_NE => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::NotEquals {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_CMP_SLT => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::SignedLessThan {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_CMP_ULT => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::UnsignedLessThan {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_CMP_SLE => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::SignedLessThanEquals {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_CMP_ULE => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::UnsignedLessThanEquals {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_CMP_SGE => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::SignedGreaterThanEquals {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_CMP_UGE => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::UnsignedGreaterThanEquals {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_CMP_SGT => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::SsignedGreaterThan {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_CMP_UGT => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::UnsignedGreaterThan {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_TEST_BIT => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::TestBit {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_BOOL_TO_INT => {
                let src = expr!();
                MediumLevelILOperation::BoolToInt {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_ADD_OVERFLOW => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::AddOverflow {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_SYSCALL => {
                let output = var_list!();
                let params = expr_list!();
                MediumLevelILOperation::Syscall {
                    output, params
                }
            }
            BNMediumLevelILOperation_MLIL_SYSCALL_UNTYPED => {
                let output = expr!();
                let params = expr!();
                let stack = expr!();
                MediumLevelILOperation::SyscallUntyped {
                    output, params, stack
                }
            }
            BNMediumLevelILOperation_MLIL_TAILCALL => {
                let output = var_list!();
                let dest = expr!();
                let params = expr_list!();
                MediumLevelILOperation::Tailcall {
                    output, dest, params
                }
            }
            BNMediumLevelILOperation_MLIL_TAILCALL_UNTYPED => {
                let output = expr!();
                let dest = expr!();
                let params = expr!();
                let stack = expr!();
                MediumLevelILOperation::TailcallUntyped {
                    output, dest, params, stack
                }
            }
            BNMediumLevelILOperation_MLIL_INTRINSIC => {
                let output = var_list!();
                let intrinsic = intrinsic!();
                let params = expr_list!();
                MediumLevelILOperation::Intrinsic {
                    output, intrinsic, params
                }
            }
            BNMediumLevelILOperation_MLIL_FREE_VAR_SLOT => {
                let dest = var!();
                MediumLevelILOperation::FreeVarSlot {
                    dest
                }
            }
            BNMediumLevelILOperation_MLIL_BP => {
                MediumLevelILOperation::Bp {
                    
                }
            }
            BNMediumLevelILOperation_MLIL_TRAP => {
                let vector = int!();
                MediumLevelILOperation::Trap {
                    vector
                }
            }
            BNMediumLevelILOperation_MLIL_UNDEF => {
                MediumLevelILOperation::Undef {
                    
                }
            }
            BNMediumLevelILOperation_MLIL_UNIMPL => {
                MediumLevelILOperation::Unimpl {
                    
                }
            }
            BNMediumLevelILOperation_MLIL_UNIMPL_MEM => {
                let src = expr!();
                MediumLevelILOperation::UnimplMem {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_FADD => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::Fadd {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_FSUB => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::Fsub {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_FMUL => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::Fmul {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_FDIV => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::Fdiv {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_FSQRT => {
                let src = expr!();
                MediumLevelILOperation::Fsqrt {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_FNEG => {
                let src = expr!();
                MediumLevelILOperation::Fneg {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_FABS => {
                let src = expr!();
                MediumLevelILOperation::Fabs {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_FLOAT_TO_INT => {
                let src = expr!();
                MediumLevelILOperation::FloatToInt {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_INT_TO_FLOAT => {
                let src = expr!();
                MediumLevelILOperation::IntToFloat {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_FLOAT_CONV => {
                let src = expr!();
                MediumLevelILOperation::FloatConv {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_ROUND_TO_INT => {
                let src = expr!();
                MediumLevelILOperation::RoundToInt {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_FLOOR => {
                let src = expr!();
                MediumLevelILOperation::Floor {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_CEIL => {
                let src = expr!();
                MediumLevelILOperation::Ceil {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_FTRUNC => {
                let src = expr!();
                MediumLevelILOperation::Ftrunc {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_FCMP_E => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::FcmpE {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_FCMP_NE => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::FcmpNe {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_FCMP_LT => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::FcmpLt {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_FCMP_LE => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::FcmpLe {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_FCMP_GE => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::FcmpGe {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_FCMP_GT => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::FcmpGt {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_FCMP_O => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::FcmpO {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_FCMP_UO => {
                let left = expr!();
                let right = expr!();
                MediumLevelILOperation::FcmpUo {
                    left, right
                }
            }
            BNMediumLevelILOperation_MLIL_SET_VAR_SSA => {
                let dest = var_ssa!();
                let src = expr!();
                MediumLevelILOperation::SetVarSsa {
                    dest, src
                }
            }
            BNMediumLevelILOperation_MLIL_SET_VAR_SSA_FIELD => {
                let prev = var_ssa_dest_and_src!();
                let offset = int!();
                let src = expr!();
                MediumLevelILOperation::SetVarSsaField {
                    prev, offset, src
                }
            }
            BNMediumLevelILOperation_MLIL_SET_VAR_SPLIT_SSA => {
                let high = var_ssa!();
                let low = var_ssa!();
                let src = expr!();
                MediumLevelILOperation::SetVarSplitSsa {
                    high, low, src
                }
            }
            BNMediumLevelILOperation_MLIL_SET_VAR_ALIASED => {
                let prev = var_ssa_dest_and_src!();
                let src = expr!();
                MediumLevelILOperation::SetVarAliased {
                    prev, src
                }
            }
            BNMediumLevelILOperation_MLIL_SET_VAR_ALIASED_FIELD => {
                let prev = var_ssa_dest_and_src!();
                let offset = int!();
                let src = expr!();
                MediumLevelILOperation::SetVarAliasedField {
                    prev, offset, src
                }
            }
            BNMediumLevelILOperation_MLIL_VAR_SSA => {
                let src = var_ssa!();
                MediumLevelILOperation::VarSsa {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_VAR_SSA_FIELD => {
                let src = var_ssa!();
                let offset = int!();
                MediumLevelILOperation::VarSsaField {
                    src, offset
                }
            }
            BNMediumLevelILOperation_MLIL_VAR_ALIASED => {
                let src = var_ssa!();
                MediumLevelILOperation::VarAliased {
                    src
                }
            }
            BNMediumLevelILOperation_MLIL_VAR_ALIASED_FIELD => {
                let src = var_ssa!();
                let offset = int!();
                MediumLevelILOperation::VarAliasedField {
                    src, offset
                }
            }
            BNMediumLevelILOperation_MLIL_VAR_SPLIT_SSA => {
                let high = var_ssa!();
                let low = var_ssa!();
                MediumLevelILOperation::VarSplitSsa {
                    high, low
                }
            }
            BNMediumLevelILOperation_MLIL_CALL_SSA => {
                let output = expr!();
                let dest = expr!();
                let params = expr_list!();
                let src_memory = int!();
                MediumLevelILOperation::CallSsa {
                    output, dest, params, src_memory
                }
            }
            BNMediumLevelILOperation_MLIL_CALL_UNTYPED_SSA => {
                let output = expr!();
                let dest = expr!();
                let params = expr!();
                let stack = expr!();
                MediumLevelILOperation::CallUntypedSsa {
                    output, dest, params, stack
                }
            }
            BNMediumLevelILOperation_MLIL_SYSCALL_SSA => {
                let output = expr!();
                let params = expr_list!();
                let src_memory = int!();
                MediumLevelILOperation::SyscallSsa {
                    output, params, src_memory
                }
            }
            BNMediumLevelILOperation_MLIL_SYSCALL_UNTYPED_SSA => {
                let output = expr!();
                let params = expr!();
                let stack = expr!();
                MediumLevelILOperation::SyscallUntypedSsa {
                    output, params, stack
                }
            }
            BNMediumLevelILOperation_MLIL_TAILCALL_SSA => {
                let output = expr!();
                let dest = expr!();
                let params = expr_list!();
                let src_memory = int!();
                MediumLevelILOperation::TailcallSsa {
                    output, dest, params, src_memory
                }
            }
            BNMediumLevelILOperation_MLIL_TAILCALL_UNTYPED_SSA => {
                let output = expr!();
                let dest = expr!();
                let params = expr!();
                let stack = expr!();
                MediumLevelILOperation::TailcallUntypedSsa {
                    output, dest, params, stack
                }
            }
            BNMediumLevelILOperation_MLIL_CALL_PARAM_SSA => {
                let src_memory = int!();
                let src = var_ssa_list!();
                MediumLevelILOperation::CallParamSsa {
                    src_memory, src
                }
            }
            BNMediumLevelILOperation_MLIL_CALL_OUTPUT_SSA => {
                let dest_memory = int!();
                let dest = var_ssa_list!();
                MediumLevelILOperation::CallOutputSsa {
                    dest_memory, dest
                }
            }
            BNMediumLevelILOperation_MLIL_LOAD_SSA => {
                let src = expr!();
                let src_memory = int!();
                MediumLevelILOperation::LoadSsa {
                    src, src_memory
                }
            }
            BNMediumLevelILOperation_MLIL_LOAD_STRUCT_SSA => {
                let src = expr!();
                let offset = int!();
                let src_memory = int!();
                MediumLevelILOperation::LoadStructSsa {
                    src, offset, src_memory
                }
            }
            BNMediumLevelILOperation_MLIL_STORE_SSA => {
                let dest = expr!();
                let dest_memory = int!();
                let src_memory = int!();
                let src = expr!();
                MediumLevelILOperation::StoreSsa {
                    dest, dest_memory, src_memory, src
                }
            }
            BNMediumLevelILOperation_MLIL_STORE_STRUCT_SSA => {
                let dest = expr!();
                let offset = int!();
                let dest_memory = int!();
                let src_memory = int!();
                let src = expr!();
                MediumLevelILOperation::StoreStructSsa {
                    dest, offset, dest_memory, src_memory, src
                }
            }
            BNMediumLevelILOperation_MLIL_INTRINSIC_SSA => {
                let output = var_ssa_list!();
                let intrinsic = intrinsic!();
                let params = expr_list!();
                MediumLevelILOperation::IntrinsicSsa {
                    output, intrinsic, params
                }
            }
            BNMediumLevelILOperation_MLIL_FREE_VAR_SLOT_SSA => {
                let prev = var_ssa_dest_and_src!();
                MediumLevelILOperation::FreeVarSlotSsa {
                    prev
                }
            }
            BNMediumLevelILOperation_MLIL_VAR_PHI => {
                let dest = var_ssa!();
                let src = var_ssa_list!();
                MediumLevelILOperation::VarPhi {
                    dest, src
                }
            }
            BNMediumLevelILOperation_MLIL_MEM_PHI => {
                let dest_memory = int!();
                let src_memory = int_list!();
                MediumLevelILOperation::MemPhi {
                    dest_memory, src_memory
                }
            }
            _ => unreachable!()
        }
    }
}
