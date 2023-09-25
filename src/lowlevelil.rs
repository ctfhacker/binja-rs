//! Provides Low Level IL analysis
#![allow(non_upper_case_globals)]
#![allow(unused_assignments)]

use binja_sys::*;

use anyhow::{anyhow, Result};
use log::trace;

use std::collections::HashMap;
use std::convert::TryInto;
use std::fmt;
use std::mem;
use std::slice;
use std::sync::Arc;

use crate::architecture::CoreArchitecture;
use crate::function::Function;
use crate::il::{Flag, Register, SSAFlag, SSARegister};
use crate::instruction::InstructionTextToken;
use crate::mediumlevelil::MediumLevelILInstruction;
use crate::traits::{BasicBlockTrait, FunctionTrait};
use crate::unsafe_try;
use crate::wrappers::{BinjaBasicBlock, BinjaFunction, BinjaLowLevelILFunction};

#[derive(Clone)]
pub struct LowLevelILFunction {
    handle: Arc<BinjaLowLevelILFunction>,
    func: Arc<BinjaFunction>,
}

unsafe impl Send for LowLevelILFunction {}
unsafe impl Sync for LowLevelILFunction {}

impl LowLevelILFunction {
    pub fn new(func: Arc<BinjaFunction>) -> Result<LowLevelILFunction> {
        // timeloop::scoped_timer!(crate::Timer::LowLevelILFunction__new);

        let handle = unsafe_try!(BNGetFunctionLowLevelIL(**func))?;
        Ok(LowLevelILFunction {
            handle: Arc::new(BinjaLowLevelILFunction::new(handle)),
            func,
        })
    }

    pub fn handle(&self) -> *mut BNLowLevelILFunction {
        **self.handle
    }

    /// Get the owner function for this LLIL function
    pub fn function(&self) -> Function {
        // timeloop::scoped_timer!(crate::Timer::LowLevelILFunction__function);

        Function::from_arc(self.func.clone())
    }

    /// Get the address of this function
    pub fn address(&self) -> u64 {
        timeloop::scoped_timer!(crate::Timer::BNLowLevelILGetCurrentAdddress);

        unsafe { BNLowLevelILGetCurrentAddress(self.handle()) }
    }

    /// Get the number of instructions in this function
    pub fn len(&self) -> usize {
        timeloop::scoped_timer!(crate::Timer::BNLowLevelILInstructionCount);

        unsafe { BNGetLowLevelILInstructionCount(self.handle()) }
    }

    /// Get the architecture for this function
    pub fn arch(&self) -> Result<CoreArchitecture> {
        // timeloop::scoped_timer!(crate::Timer::LowLevelILFunction__Arch);

        CoreArchitecture::new_from_func(**self.func)
    }

    fn get_index_for_instruction(&self, i: usize) -> usize {
        timeloop::scoped_timer!(crate::Timer::BNGetLowLevelILIndexForInstruction);

        unsafe { BNGetLowLevelILIndexForInstruction(self.handle(), i) }
    }

    pub fn ssa_form(&self) -> Result<Self> {
        timeloop::scoped_timer!(crate::Timer::LowLevelILFunction__SsaForm);

        let handle = unsafe_try!(BNGetLowLevelILSSAForm(self.handle()))?;
        Ok(LowLevelILFunction {
            handle: Arc::new(BinjaLowLevelILFunction::new(handle)),
            func: self.func.clone(),
        })
    }

    pub fn non_ssa_form(&self) -> Result<Self> {
        timeloop::scoped_timer!(crate::Timer::LowLevelILFunction__NonSsaForm);

        let handle = unsafe_try!(BNGetLowLevelILNonSSAForm(self.handle()))?;
        Ok(LowLevelILFunction {
            handle: Arc::new(BinjaLowLevelILFunction::new(handle)),
            func: self.func.clone(),
        })
    }

    pub fn get_ssa_reg_definition(&self, ssavar: &SSARegister) -> LowLevelILInstruction {
        let index = unsafe {
            BNGetLowLevelILSSARegisterDefinition(
                self.handle(),
                ssavar.reg.index,
                ssavar.version.try_into().unwrap(),
            )
        };

        LowLevelILInstruction::from_func_index(self.clone(), index)
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

impl fmt::Display for LowLevelILFunction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl fmt::Debug for LowLevelILFunction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut result = f.debug_struct("LLILFunction");
        result.field("address", &format_args!("{:#x}", self.address()));
        result.field("name", &self.function().name());
        result.finish()
    }
}

impl FunctionTrait for LowLevelILFunction {
    type Ins = LowLevelILInstruction;
    type Block = LowLevelILBasicBlock;
    type Func = LowLevelILFunction;

    /// Retrieve a LowLevelILInstruction for a given index
    fn instruction(&self, i: usize) -> Result<Self::Ins> {
        timeloop::scoped_timer!(crate::Timer::LowLevelIL__Instruction);

        let res = LowLevelILInstruction::from_func_index(self.clone(), i);
        Ok(res)
    }

    fn blocks(&self) -> Vec<Self::Block> {
        timeloop::scoped_timer!(crate::Timer::LowLevelIL__Blocks);

        let mut count = 0;

        unsafe {
            let blocks = BNGetLowLevelILBasicBlockList(self.handle(), &mut count);
            trace!("Function {} returned {} blocks", self, count);

            let blocks_slice = slice::from_raw_parts(blocks, count as usize);

            let result = blocks_slice
                .iter()
                .map(|&b| LowLevelILBasicBlock::new(b, self.clone()).unwrap())
                .collect();

            BNFreeBasicBlockList(blocks, count);
            result
        }
    }

    fn ssa_form(&self) -> Result<LowLevelILFunction> {
        self.ssa_form()
    }

    fn non_ssa_form(&self) -> Result<LowLevelILFunction> {
        self.non_ssa_form()
    }

    /// Construct the text for a given LowLevelILInstruction index
    fn text(&self, i: usize) -> Result<String> {
        timeloop::scoped_timer!(crate::Timer::LowLevelILFunction__Text);

        let mut count = 0;

        unsafe {
            // Initialize pointer to get data from BNGetLowLevelILInstructionText
            let mut list = mem::zeroed();

            BNGetLowLevelILInstructionText(
                self.handle(),
                **self.func,
                self.arch()?.handle(),
                i,
                /* settings */ std::ptr::null_mut(),
                &mut list,
                &mut count,
            );

            if list.is_null() {
                println!(
                    "[Possible bug] Cannot retrieve LowLevelILInstruction Tokens: {} index {:#x}",
                    self, i
                );
                return Err(anyhow!("Failed to retrieve LLILInstruction tokens"));
            }

            let list_slice = slice::from_raw_parts(list, count as usize);

            let result: Vec<InstructionTextToken> = list_slice
                .iter()
                .map(|&l| InstructionTextToken::new_from_token(l))
                .collect();

            BNFreeInstructionText(list, count);

            Ok(result
                .iter()
                .fold(String::new(), |acc, x| format!("{}{}", acc, x.text)))
        }
    }

    /// Construct the text for a given LowLevelILInstruction index
    fn expr_text(&self, _expr_index: usize) -> Result<String> {
        panic!("Expr text not impl for LLIL");
    }
}

#[derive(Clone)]
pub struct LowLevelILBasicBlock {
    handle: Arc<BinjaBasicBlock>,
    func: LowLevelILFunction,
}

impl LowLevelILBasicBlock {
    pub fn new(
        handle: *mut BNBasicBlock,
        func: LowLevelILFunction,
    ) -> Result<LowLevelILBasicBlock> {
        timeloop::scoped_timer!(crate::Timer::LowLevelILBasicBlock__new);

        let handle = unsafe_try!(BNNewBasicBlockReference(handle))?;
        Ok(LowLevelILBasicBlock {
            handle: Arc::new(BinjaBasicBlock::new(handle)),
            func,
        })
    }
}

impl BasicBlockTrait for LowLevelILBasicBlock {
    type Ins = LowLevelILInstruction;
    type Func = LowLevelILFunction;

    fn handle(&self) -> *mut BNBasicBlock {
        timeloop::scoped_timer!(crate::Timer::LowLevelILBasicBlock__handle);

        **self.handle
    }

    fn func(&self) -> Option<&Self::Func> {
        timeloop::scoped_timer!(crate::Timer::LowLevelILBasicBlock__func);

        Some(&self.func)
    }

    fn raw_function(&self) -> Function {
        timeloop::scoped_timer!(crate::Timer::LowLevelILBasicBlock__raw_function);

        self.func.function()
    }
}

pub struct LowLevelILInstruction {
    pub operation: Box<LowLevelILOperation>,
    pub source_operand: u32,
    pub size: usize,
    pub flags: u32,
    pub operands: [u64; 4usize],
    pub address: u64,
    pub function: LowLevelILFunction,
    pub expr_index: usize,
    pub instr_index: Option<usize>,
    pub text: Result<String>,
}

impl LowLevelILInstruction {
    /// Get the LLIL instruction from the given `func` at the given `instr_index`
    pub fn from_func_index(func: LowLevelILFunction, instr_index: usize) -> LowLevelILInstruction {
        timeloop::scoped_timer!(crate::Timer::LowLevelILInstruction__from_func_index);

        // Get the raw index for the given instruction index
        let expr_index = func.get_index_for_instruction(instr_index);

        LowLevelILInstruction::from_expr(func, expr_index, Some(instr_index))
    }

    /// Get the LLIL instruction from the given internal `expr` at the given `instr_index`
    pub fn from_expr(
        func: LowLevelILFunction,
        expr_index: usize,
        instr_index: Option<usize>,
    ) -> LowLevelILInstruction {
        timeloop::scoped_timer!(crate::Timer::LowLevelILInstruction__from_expr);

        // Get the IL for the given index
        let instr = unsafe { BNGetLowLevelILByIndex(func.handle(), expr_index) };

        // If we have the instruction index, grab the text for that instruction
        let text = if let Some(index) = instr_index {
            if index as i64 == -1 {
                println!("BAD INDEX: {}", std::backtrace::Backtrace::force_capture());
            }

            func.text(index)
        } else {
            Err(anyhow!("text() for None from_expr unimpl")) // unimplemented!()
        };

        LowLevelILInstruction {
            operation: Box::new(LowLevelILOperation::from_instr(instr, &func, expr_index)),
            source_operand: instr.sourceOperand,
            size: instr.size,
            flags: instr.flags,
            operands: instr.operands,
            address: instr.address,
            function: func,
            expr_index,
            instr_index,
            text,
        }
    }

    /// Convert LLIL instruction into LLIL SSA (Alias for ssa_form)
    pub fn ssa(&self) -> Result<LowLevelILInstruction> {
        timeloop::scoped_timer!(crate::Timer::LowLevelILInstruction__ssa);

        self.ssa_form()
    }

    /// Convert LLIL instruction into LLIL SSA
    pub fn ssa_form(&self) -> Result<LowLevelILInstruction> {
        timeloop::scoped_timer!(crate::Timer::LowLevelILInstruction__ssa_form);

        let func_ssa = self.function.ssa_form()?;
        unsafe {
            let expr_index = BNGetLowLevelILSSAExprIndex(self.function.handle(), self.expr_index);
            Ok(LowLevelILInstruction::from_expr(
                func_ssa,
                expr_index,
                self.instr_index,
            ))
        }
    }

    /// Convert LLIL SSA instruction into LLIL
    pub fn non_ssa_form(&self) -> Result<LowLevelILInstruction> {
        timeloop::scoped_timer!(crate::Timer::LowLevelILInstruction__non_ssa_form);

        let non_func_ssa = self.function.non_ssa_form()?;
        unsafe {
            let expr_index =
                BNGetLowLevelILNonSSAExprIndex(self.function.handle(), self.expr_index);
            let instr_index = self
                .instr_index
                .map(|x| BNGetLowLevelILNonSSAInstructionIndex(self.function.handle(), x));
            Ok(LowLevelILInstruction::from_expr(
                non_func_ssa,
                expr_index,
                instr_index,
            ))
        }
    }

    /// Get the MLIL instruction for this LLIL instruction
    pub fn medium_level_il(&self) -> Result<MediumLevelILInstruction> {
        timeloop::scoped_timer!(crate::Timer::LowLevelILInstruction__medium_level_il);

        let mlil_expr_index =
            unsafe { BNGetMediumLevelILExprIndex(self.function.handle(), self.expr_index) };

        let mlil = self.function.function().mlil()?;

        Ok(MediumLevelILInstruction::from_expr(
            mlil,
            mlil_expr_index,
            None,
        ))
    }

    /// Get the MLIL instruction for this LLIL instruction.
    /// Alias for `self.medium_level_il()`
    pub fn mlil(&self) -> Result<MediumLevelILInstruction> {
        timeloop::scoped_timer!(crate::Timer::LowLevelIL__mlil);

        self.medium_level_il()
    }

    /// Get the MLIL instruction for this LLIL instruction.
    /// Alias for `self.medium_level_il()`
    pub fn mlilssa(&self) -> Result<MediumLevelILInstruction> {
        timeloop::scoped_timer!(crate::Timer::LowLevelIL__mlilssa);

        self.medium_level_il()?.ssa_form()
    }
}

impl fmt::Display for LowLevelILInstruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.text {
            Ok(text) => write!(f, "{} ", text),
            Err(_) => write!(f, "[{}:{}] Invalid text!", self.function, self.address),
        }
    }
}

impl fmt::Debug for LowLevelILInstruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("LowLevelILInstruction")
            .field("Operation", &self.operation)
            .finish()
    }
}

/*
macro_rules! reg_stack {
    () => {{
        unimplemented!("reg_stack");
    }}
}

macro_rules! reg_stack_ssa_dest_and_src {
    () => {{
        unimplemented!("reg_stack_ssa_dest_and_src");
    }}
}

macro_rules! float {
    () => {{
        unimplemented!("float");
    }}
}

macro_rules! sem_group {
    () => {{
        unimplemented!("sem_group");
    }}
}

macro_rules! reg_stack_ssa_list {
    () => {{
        unimplemented!("reg_stack_ssa_list");
    }}
}

macro_rules! cond {
    () => {{
        unimplemented!("cond");
    }}
}

macro_rules! reg_stack_ssa {
    () => {{
        unimplemented!("reg_stack_ssa");
    }}
}

macro_rules! sem_class {
    () => {{
        unimplemented!("sem_class");
    }}
}

macro_rules! intrinsic {
    () => {{
        unimplemented!("intrinsic");
    }}
}

macro_rules! reg_or_flag_list {
    () => {{
        unimplemented!("reg_or_flag_list");
    }}
}

macro_rules! reg_or_flag_ssa_list {
    () => {{
        unimplemented!("reg_or_flag_ssa_list");
    }}
}
*/

#[derive(Debug)]
pub enum LowLevelILOperation {
    Nop {},
    SetReg {
        dest: Register,
        src: LowLevelILInstruction,
    },
    SetRegSplit {
        hi: Register,
        lo: Register,
        src: LowLevelILInstruction,
    },
    SetFlag {
        dest: Flag,
        src: LowLevelILInstruction,
    },
    // SetRegStackRel { stack: reg_stack, dest: LowLevelILInstruction, src: LowLevelILInstruction, },
    // RegStackPush { stack: reg_stack, src: LowLevelILInstruction, },
    Load {
        src: LowLevelILInstruction,
    },
    Store {
        dest: LowLevelILInstruction,
        src: LowLevelILInstruction,
    },
    Push {
        src: LowLevelILInstruction,
    },
    Pop {},
    Reg {
        src: Register,
    },
    RegSplit {
        hi: Register,
        lo: Register,
    },
    // RegStackRel { stack: reg_stack, src: LowLevelILInstruction, },
    // RegStackPop { stack: reg_stack, },
    RegStackFreeReg {
        dest: Register,
    },
    // RegStackFreeRel { stack: reg_stack, dest: LowLevelILInstruction, },
    Const {
        constant: u64,
    },
    ConstPtr {
        constant: u64,
    },
    ExternPtr {
        constant: u64,
        offset: u64,
    },
    FloatConst {
        constant: f64,
    },
    Flag {
        src: Flag,
    },
    FlagBit {
        src: Flag,
        bit: u64,
    },
    Add {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    Adc {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
        carry: LowLevelILInstruction,
    },
    Sub {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    Sbb {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
        carry: LowLevelILInstruction,
    },
    And {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    Or {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    Xor {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    Lsl {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    Lsr {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    Asr {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    Rol {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    Rlc {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
        carry: LowLevelILInstruction,
    },
    Ror {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    Rrc {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
        carry: LowLevelILInstruction,
    },
    Mul {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    MuluDp {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    MulsDp {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    Divu {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    DivuDp {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    Divs {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    DivsDp {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    Modu {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    ModuDp {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    Mods {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    ModsDp {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    Neg {
        src: LowLevelILInstruction,
    },
    Not {
        src: LowLevelILInstruction,
    },
    SignExtend {
        src: LowLevelILInstruction,
    },
    ZeroExtend {
        src: LowLevelILInstruction,
    },
    LowPart {
        src: LowLevelILInstruction,
    },
    Jump {
        dest: LowLevelILInstruction,
    },
    JumpTo {
        dest: LowLevelILInstruction,
        targets: HashMap<u64, u64>,
    },
    Call {
        dest: LowLevelILInstruction,
    },
    CallStackAdjust {
        dest: LowLevelILInstruction,
        stack_adjustment: u64,
        reg_stack_adjustments: HashMap<u64, u32>,
    },
    Tailcall {
        dest: LowLevelILInstruction,
    },
    Ret {
        dest: LowLevelILInstruction,
    },
    Noret {},
    If {
        condition: LowLevelILInstruction,
        true_: u64,
        false_: u64,
    },
    Goto {
        dest: u64,
    },
    // FlagCond { condition: cond, semantic_class: sem_class, },
    // FlagGroup { semantic_group: sem_group, },
    Equals {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    NotEquals {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    SignedLessThan {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    UnsignedLessThan {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    SignedLessThanEquals {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    UnsignedLessThanEquals {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    SignedGreaterThanEquals {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    UnsignedGreaterThanEquals {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    SsignedGreaterThan {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    UnsignedGreaterThan {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    TestBit {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    BoolToInt {
        src: LowLevelILInstruction,
    },
    AddOverflow {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    Syscall {},
    Bp {},
    Trap {
        vector: u64,
    },
    // Intrinsic { output: reg_or_flag_list, intrinsic: intrinsic, param: LowLevelILInstruction, },
    Undef {},
    Unimpl {},
    UnimplMem {
        src: LowLevelILInstruction,
    },
    Fadd {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    Fsub {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    Fmul {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    Fdiv {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    Fsqrt {
        src: LowLevelILInstruction,
    },
    Fneg {
        src: LowLevelILInstruction,
    },
    Fabs {
        src: LowLevelILInstruction,
    },
    FloatToInt {
        src: LowLevelILInstruction,
    },
    IntToFloat {
        src: LowLevelILInstruction,
    },
    FloatConv {
        src: LowLevelILInstruction,
    },
    RoundToInt {
        src: LowLevelILInstruction,
    },
    Floor {
        src: LowLevelILInstruction,
    },
    Ceil {
        src: LowLevelILInstruction,
    },
    Ftrunc {
        src: LowLevelILInstruction,
    },
    FcmpE {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    FcmpNe {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    FcmpLt {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    FcmpLe {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    FcmpGe {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    FcmpGt {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    FcmpO {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    FcmpUo {
        left: LowLevelILInstruction,
        right: LowLevelILInstruction,
    },
    SetRegSsa {
        dest: SSARegister,
        src: LowLevelILInstruction,
    },
    SetRegSsaPartial {
        full_reg: SSARegister,
        dest: Register,
        src: LowLevelILInstruction,
    },
    SetRegSplitSsa {
        hi: LowLevelILInstruction,
        lo: LowLevelILInstruction,
        src: LowLevelILInstruction,
    },
    SetRegStackRelSsa {
        stack: LowLevelILInstruction,
        dest: LowLevelILInstruction,
        top: LowLevelILInstruction,
        src: LowLevelILInstruction,
    },
    SetRegStackAbsSsa {
        stack: LowLevelILInstruction,
        dest: Register,
        src: LowLevelILInstruction,
    },
    RegSplitDestSsa {
        dest: SSARegister,
    },
    // RegStackDestSsa { src: reg_stack_ssa_dest_and_src, },
    RegSsa {
        src: SSARegister,
    },
    RegSsaPartial {
        full_reg: SSARegister,
        src: Register,
    },
    RegSplitSsa {
        hi: SSARegister,
        lo: SSARegister,
    },
    // RegStackRelSsa { stack: reg_stack_ssa, src: LowLevelILInstruction, top: LowLevelILInstruction, },
    // RegStackAbsSsa { stack: reg_stack_ssa, src: Register, },
    RegStackFreeRelSsa {
        stack: LowLevelILInstruction,
        dest: LowLevelILInstruction,
        top: LowLevelILInstruction,
    },
    RegStackFreeAbsSsa {
        stack: LowLevelILInstruction,
        dest: Register,
    },
    SetFlagSsa {
        dest: SSAFlag,
        src: LowLevelILInstruction,
    },
    FlagSsa {
        src: SSAFlag,
    },
    FlagBitSsa {
        src: SSAFlag,
        bit: u64,
    },
    CallSsa {
        output: LowLevelILInstruction,
        dest: LowLevelILInstruction,
        stack: LowLevelILInstruction,
        param: LowLevelILInstruction,
    },
    SyscallSsa {
        output: LowLevelILInstruction,
        stack: LowLevelILInstruction,
        param: LowLevelILInstruction,
    },
    TailcallSsa {
        output: LowLevelILInstruction,
        dest: LowLevelILInstruction,
        stack: LowLevelILInstruction,
        param: LowLevelILInstruction,
    },
    CallParam {
        src: Vec<LowLevelILInstruction>,
    },
    CallStackSsa {
        src: SSARegister,
        src_memory: u64,
    },
    CallOutputSsa {
        dest_memory: u64,
        dest: Vec<SSARegister>,
    },
    LoadSsa {
        src: LowLevelILInstruction,
        src_memory: u64,
    },
    StoreSsa {
        dest: LowLevelILInstruction,
        dest_memory: u64,
        src_memory: u64,
        src: LowLevelILInstruction,
    },
    // IntrinsicSsa { output: reg_or_flag_ssa_list, intrinsic: intrinsic, param: LowLevelILInstruction, },
    RegPhi {
        dest: SSARegister,
        src: Vec<SSARegister>,
    },
    // RegStackPhi { dest: reg_stack_ssa, src: reg_stack_ssa_list, },
    FlagPhi {
        dest: SSAFlag,
        src: Vec<SSAFlag>,
    },
    MemPhi {
        dest_memory: u64,
        src_memory: Vec<u64>,
    },
}
impl LowLevelILOperation {
    pub fn from_instr(
        instr: BNLowLevelILInstruction,
        func: &LowLevelILFunction,
        expr_index: usize,
    ) -> LowLevelILOperation {
        timeloop::scoped_timer!(crate::Timer::LowLevelILOperation__from_instr);

        let arch = func.arch().expect("Failed to get arch for LLIL").clone();
        let mut operand_index = 0;

        // Macros used to define each of the types of arguments in each operation
        macro_rules! expr {
            () => {{
                let res = LowLevelILInstruction::from_expr(
                    func.clone(),
                    instr.operands[operand_index].try_into().unwrap(),
                    None,
                );
                operand_index += 1;
                res
            }};
        }

        macro_rules! reg {
            () => {{
                let res = Register::new(arch.clone(), instr.operands[operand_index] as u32);
                operand_index += 1;
                res
            }};
        }

        macro_rules! float {
            () => {{
                // Extract the value from the operand
                let res = match instr.size {
                    4 => f32::from_bits(instr.operands[operand_index] as u32) as f64,
                    8 => f64::from_bits(instr.operands[operand_index]),
                    _ => unreachable!(),
                };
                operand_index += 1;
                res
            }};
        }

        macro_rules! int {
            () => {{
                let res = (instr.operands[operand_index] & ((1 << 63) - 1))
                    .wrapping_sub(instr.operands[operand_index] & (1 << 63));
                operand_index += 1;
                res
            }};
        }

        macro_rules! flag {
            () => {{
                let res = Flag::new(arch.clone(), instr.operands[operand_index] as u32);
                operand_index += 1;
                res
            }};
        }

        macro_rules! reg_ssa {
            () => {{
                let reg = instr.operands[operand_index] as u32;
                operand_index += 1;
                let version = instr.operands[operand_index] as u32;
                operand_index += 1;
                let reg = Register::new(arch.clone(), reg);
                SSARegister::new(reg, version)
            }};
        }

        macro_rules! reg_ssa_list {
            () => {{
                let mut count = 0;
                let mut regs = Vec::new();

                unsafe {
                    let operands = BNLowLevelILGetOperandList(
                        func.handle(),
                        expr_index,
                        operand_index,
                        &mut count,
                    );

                    operand_index += 1;

                    let operands_slice = slice::from_raw_parts(operands, count as usize);

                    for i in (0..count).step_by(2) {
                        let reg = operands_slice[i as usize];
                        let version = operands_slice[i as usize + 1] as u32;
                        let reg = Register::new(arch.clone(), reg as u32);
                        regs.push(SSARegister::new(reg, version));
                    }
                    BNLowLevelILFreeOperandList(operands);
                }

                regs
            }};
        }

        macro_rules! expr_list {
            () => {{
                // Initialize the resulting instructions vec
                let mut instrs = Vec::new();
                let mut count = 0;

                unsafe {
                    // Get the pointer to instruction indexes from binja core
                    let operands = BNLowLevelILGetOperandList(
                        func.handle(),
                        expr_index,
                        operand_index,
                        &mut count,
                    );

                    operand_index += 1;

                    // Get the slice from the found pointer
                    let operands_slice = slice::from_raw_parts(operands, count as usize);

                    // Create each instruction
                    for op in operands_slice {
                        let i = LowLevelILInstruction::from_expr(
                            func.clone(),
                            (*op).try_into().unwrap(),
                            None,
                        );
                        instrs.push(i);
                    }

                    // Free the binja core pointer
                    BNLowLevelILFreeOperandList(operands);
                }

                instrs
            }};
        }

        macro_rules! target_map {
            () => {{
                // Initialize the target map
                let mut target_map = HashMap::new();
                let mut count = 0;

                unsafe {
                    // Get the operands from the binja core
                    let operands = BNLowLevelILGetOperandList(
                        func.handle(),
                        expr_index,
                        operand_index,
                        &mut count,
                    );

                    operand_index += 1;

                    // Cast the result from binja core into the slice of operands
                    let operands_slice = slice::from_raw_parts(operands, count as usize);

                    // Extract the key, value pairs from the found slice
                    for i in (0..count).step_by(2) {
                        let key = operands_slice[i as usize];
                        let value = operands_slice[i as usize + 1];
                        target_map.insert(key, value);
                    }

                    // Free the operands
                    BNLowLevelILFreeOperandList(operands);
                }

                target_map
            }};
        }

        macro_rules! flag_ssa {
            () => {{
                let flag = instr.operands[operand_index] as u32;
                let version = instr.operands[operand_index + 1] as u32;
                operand_index += 2;
                let reg = Flag::new(arch.clone(), flag);
                SSAFlag::new(reg, version)
            }};
        }

        macro_rules! int_list {
            () => {{
                // Generate the int list from the binja core
                let mut count = 0;
                let mut int_list = Vec::new();

                unsafe {
                    let operands = BNLowLevelILGetOperandList(
                        func.handle(),
                        expr_index,
                        operand_index,
                        &mut count,
                    );

                    operand_index += 1;

                    let operands_slice = slice::from_raw_parts(operands, count as usize);

                    for i in 0..count {
                        int_list.push(operands_slice[i as usize]);
                    }

                    BNLowLevelILFreeOperandList(operands);
                }

                int_list
            }};
        }

        macro_rules! flag_ssa_list {
            () => {{
                // Generate the SSAFlag list from binja core
                let mut count = 0;
                let mut flags = Vec::new();

                unsafe {
                    let operands = BNLowLevelILGetOperandList(
                        func.handle(),
                        expr_index,
                        operand_index,
                        &mut count,
                    );

                    operand_index += 1;

                    let operands_slice = slice::from_raw_parts(operands, count as usize);

                    for i in (0..count).step_by(2) {
                        let flag = operands_slice[i as usize];
                        let version = operands_slice[i as usize + 1] as u32;
                        let flag = Flag::new(arch.clone(), flag as u32);
                        flags.push(SSAFlag::new(flag, version));
                    }

                    BNLowLevelILFreeOperandList(operands);
                }

                flags
            }};
        }

        macro_rules! reg_stack_adjust {
            () => {{
                let mut count = 0;
                let mut res = HashMap::new();

                unsafe {
                    let operands = BNLowLevelILGetOperandList(
                        func.handle(),
                        expr_index,
                        operand_index,
                        &mut count,
                    );

                    operand_index += 1;

                    let operands_slice = slice::from_raw_parts(operands, count as usize);

                    for i in (0..count).step_by(2) {
                        let reg_stack = operands_slice[i as usize];
                        let adjust = operands_slice[i as usize + 1] as u32;
                        res.insert(reg_stack, adjust);
                    }
                    BNLowLevelILFreeOperandList(operands);
                }

                res
            }};
        }

        match instr.operation {
            BNLowLevelILOperation_LLIL_NOP => LowLevelILOperation::Nop {},
            BNLowLevelILOperation_LLIL_SET_REG => {
                let dest = reg!();
                let src = expr!();
                LowLevelILOperation::SetReg { dest, src }
            }
            BNLowLevelILOperation_LLIL_SET_REG_SPLIT => {
                let hi = reg!();
                let lo = reg!();
                let src = expr!();
                LowLevelILOperation::SetRegSplit { hi, lo, src }
            }
            BNLowLevelILOperation_LLIL_SET_FLAG => {
                let dest = flag!();
                let src = expr!();
                LowLevelILOperation::SetFlag { dest, src }
            }
            BNLowLevelILOperation_LLIL_SET_REG_STACK_REL => {
                LowLevelILOperation::Unimpl {}
                /*
                let stack = reg_stack!();
                let dest = expr!();
                let src = expr!();
                LowLevelILOperation::SetRegStackRel {
                    stack, dest, src
                }
                */
            }
            BNLowLevelILOperation_LLIL_REG_STACK_PUSH => {
                LowLevelILOperation::Unimpl {}
                /*
                let stack = reg_stack!();
                let src = expr!();
                LowLevelILOperation::RegStackPush {
                    stack, src
                }
                */
            }
            BNLowLevelILOperation_LLIL_LOAD => {
                let src = expr!();
                LowLevelILOperation::Load { src }
            }
            BNLowLevelILOperation_LLIL_STORE => {
                let dest = expr!();
                let src = expr!();
                LowLevelILOperation::Store { dest, src }
            }
            BNLowLevelILOperation_LLIL_PUSH => {
                let src = expr!();
                LowLevelILOperation::Push { src }
            }
            BNLowLevelILOperation_LLIL_POP => LowLevelILOperation::Pop {},
            BNLowLevelILOperation_LLIL_REG => {
                let src = reg!();
                LowLevelILOperation::Reg { src }
            }
            BNLowLevelILOperation_LLIL_REG_SPLIT => {
                let hi = reg!();
                let lo = reg!();
                LowLevelILOperation::RegSplit { hi, lo }
            }
            BNLowLevelILOperation_LLIL_REG_STACK_REL => {
                LowLevelILOperation::Unimpl {}
                /*
                let stack = reg_stack!();
                let src = expr!();
                LowLevelILOperation::RegStackRel {
                    stack, src
                }
                */
            }
            BNLowLevelILOperation_LLIL_REG_STACK_POP => {
                LowLevelILOperation::Unimpl {}
                /*
                let stack = reg_stack!();
                LowLevelILOperation::RegStackPop {
                    stack
                }
                */
            }
            BNLowLevelILOperation_LLIL_REG_STACK_FREE_REG => {
                LowLevelILOperation::Unimpl {}
                /*
                let dest = reg!();
                LowLevelILOperation::RegStackFreeReg {
                    dest
                }
                */
            }
            BNLowLevelILOperation_LLIL_REG_STACK_FREE_REL => {
                LowLevelILOperation::Unimpl {}
                /*
                let stack = reg_stack!();
                let dest = expr!();
                LowLevelILOperation::RegStackFreeRel {
                    stack, dest
                }
                */
            }
            BNLowLevelILOperation_LLIL_CONST => {
                let constant = int!();
                LowLevelILOperation::Const { constant }
            }
            BNLowLevelILOperation_LLIL_CONST_PTR => {
                let constant = int!();
                LowLevelILOperation::ConstPtr { constant }
            }
            BNLowLevelILOperation_LLIL_EXTERN_PTR => {
                let constant = int!();
                let offset = int!();
                LowLevelILOperation::ExternPtr { constant, offset }
            }
            BNLowLevelILOperation_LLIL_FLOAT_CONST => {
                let constant = float!();
                LowLevelILOperation::FloatConst { constant }
            }
            BNLowLevelILOperation_LLIL_FLAG => {
                let src = flag!();
                LowLevelILOperation::Flag { src }
            }
            BNLowLevelILOperation_LLIL_FLAG_BIT => {
                let src = flag!();
                let bit = int!();
                LowLevelILOperation::FlagBit { src, bit }
            }
            BNLowLevelILOperation_LLIL_ADD => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::Add { left, right }
            }
            BNLowLevelILOperation_LLIL_ADC => {
                let left = expr!();
                let right = expr!();
                let carry = expr!();
                LowLevelILOperation::Adc { left, right, carry }
            }
            BNLowLevelILOperation_LLIL_SUB => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::Sub { left, right }
            }
            BNLowLevelILOperation_LLIL_SBB => {
                let left = expr!();
                let right = expr!();
                let carry = expr!();
                LowLevelILOperation::Sbb { left, right, carry }
            }
            BNLowLevelILOperation_LLIL_AND => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::And { left, right }
            }
            BNLowLevelILOperation_LLIL_OR => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::Or { left, right }
            }
            BNLowLevelILOperation_LLIL_XOR => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::Xor { left, right }
            }
            BNLowLevelILOperation_LLIL_LSL => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::Lsl { left, right }
            }
            BNLowLevelILOperation_LLIL_LSR => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::Lsr { left, right }
            }
            BNLowLevelILOperation_LLIL_ASR => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::Asr { left, right }
            }
            BNLowLevelILOperation_LLIL_ROL => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::Rol { left, right }
            }
            BNLowLevelILOperation_LLIL_RLC => {
                let left = expr!();
                let right = expr!();
                let carry = expr!();
                LowLevelILOperation::Rlc { left, right, carry }
            }
            BNLowLevelILOperation_LLIL_ROR => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::Ror { left, right }
            }
            BNLowLevelILOperation_LLIL_RRC => {
                let left = expr!();
                let right = expr!();
                let carry = expr!();
                LowLevelILOperation::Rrc { left, right, carry }
            }
            BNLowLevelILOperation_LLIL_MUL => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::Mul { left, right }
            }
            BNLowLevelILOperation_LLIL_MULU_DP => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::MuluDp { left, right }
            }
            BNLowLevelILOperation_LLIL_MULS_DP => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::MulsDp { left, right }
            }
            BNLowLevelILOperation_LLIL_DIVU => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::Divu { left, right }
            }
            BNLowLevelILOperation_LLIL_DIVU_DP => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::DivuDp { left, right }
            }
            BNLowLevelILOperation_LLIL_DIVS => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::Divs { left, right }
            }
            BNLowLevelILOperation_LLIL_DIVS_DP => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::DivsDp { left, right }
            }
            BNLowLevelILOperation_LLIL_MODU => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::Modu { left, right }
            }
            BNLowLevelILOperation_LLIL_MODU_DP => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::ModuDp { left, right }
            }
            BNLowLevelILOperation_LLIL_MODS => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::Mods { left, right }
            }
            BNLowLevelILOperation_LLIL_MODS_DP => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::ModsDp { left, right }
            }
            BNLowLevelILOperation_LLIL_NEG => {
                let src = expr!();
                LowLevelILOperation::Neg { src }
            }
            BNLowLevelILOperation_LLIL_NOT => {
                let src = expr!();
                LowLevelILOperation::Not { src }
            }
            BNLowLevelILOperation_LLIL_SX => {
                let src = expr!();
                LowLevelILOperation::SignExtend { src }
            }
            BNLowLevelILOperation_LLIL_ZX => {
                let src = expr!();
                LowLevelILOperation::ZeroExtend { src }
            }
            BNLowLevelILOperation_LLIL_LOW_PART => {
                let src = expr!();
                LowLevelILOperation::LowPart { src }
            }
            BNLowLevelILOperation_LLIL_JUMP => {
                let dest = expr!();
                LowLevelILOperation::Jump { dest }
            }
            BNLowLevelILOperation_LLIL_JUMP_TO => {
                let dest = expr!();
                let targets = target_map!();
                LowLevelILOperation::JumpTo { dest, targets }
            }
            BNLowLevelILOperation_LLIL_CALL => {
                let dest = expr!();
                LowLevelILOperation::Call { dest }
            }
            BNLowLevelILOperation_LLIL_CALL_STACK_ADJUST => {
                let dest = expr!();
                let stack_adjustment = int!();
                let reg_stack_adjustments = reg_stack_adjust!();
                LowLevelILOperation::CallStackAdjust {
                    dest,
                    stack_adjustment,
                    reg_stack_adjustments,
                }
            }
            BNLowLevelILOperation_LLIL_TAILCALL => {
                let dest = expr!();
                LowLevelILOperation::Tailcall { dest }
            }
            BNLowLevelILOperation_LLIL_RET => {
                let dest = expr!();
                LowLevelILOperation::Ret { dest }
            }
            BNLowLevelILOperation_LLIL_NORET => LowLevelILOperation::Noret {},
            BNLowLevelILOperation_LLIL_IF => {
                let condition = expr!();
                let true_ = int!();
                let false_ = int!();
                LowLevelILOperation::If {
                    condition,
                    true_,
                    false_,
                }
            }
            BNLowLevelILOperation_LLIL_GOTO => {
                let dest = int!();
                LowLevelILOperation::Goto { dest }
            }
            BNLowLevelILOperation_LLIL_FLAG_COND => {
                unimplemented!();
                /*
                let condition = cond!();
                let semantic_class = sem_class!();
                LowLevelILOperation::FlagCond {
                    condition, semantic_class
                }
                */
            }
            BNLowLevelILOperation_LLIL_FLAG_GROUP => {
                unimplemented!();
                /*
                let semantic_group = sem_group!();
                LowLevelILOperation::FlagGroup {
                    semantic_group
                }
                */
            }
            BNLowLevelILOperation_LLIL_CMP_E => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::Equals { left, right }
            }
            BNLowLevelILOperation_LLIL_CMP_NE => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::NotEquals { left, right }
            }
            BNLowLevelILOperation_LLIL_CMP_SLT => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::SignedLessThan { left, right }
            }
            BNLowLevelILOperation_LLIL_CMP_ULT => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::UnsignedLessThan { left, right }
            }
            BNLowLevelILOperation_LLIL_CMP_SLE => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::SignedLessThanEquals { left, right }
            }
            BNLowLevelILOperation_LLIL_CMP_ULE => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::UnsignedLessThanEquals { left, right }
            }
            BNLowLevelILOperation_LLIL_CMP_SGE => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::SignedGreaterThanEquals { left, right }
            }
            BNLowLevelILOperation_LLIL_CMP_UGE => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::UnsignedGreaterThanEquals { left, right }
            }
            BNLowLevelILOperation_LLIL_CMP_SGT => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::SsignedGreaterThan { left, right }
            }
            BNLowLevelILOperation_LLIL_CMP_UGT => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::UnsignedGreaterThan { left, right }
            }
            BNLowLevelILOperation_LLIL_TEST_BIT => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::TestBit { left, right }
            }
            BNLowLevelILOperation_LLIL_BOOL_TO_INT => {
                let src = expr!();
                LowLevelILOperation::BoolToInt { src }
            }
            BNLowLevelILOperation_LLIL_ADD_OVERFLOW => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::AddOverflow { left, right }
            }
            BNLowLevelILOperation_LLIL_SYSCALL => LowLevelILOperation::Syscall {},
            BNLowLevelILOperation_LLIL_BP => LowLevelILOperation::Bp {},
            BNLowLevelILOperation_LLIL_TRAP => {
                let vector = int!();
                LowLevelILOperation::Trap { vector }
            }
            BNLowLevelILOperation_LLIL_INTRINSIC => {
                // unimplemented!()
                LowLevelILOperation::Unimpl {}
                /*
                let output = reg_or_flag_list!();
                let intrinsic = intrinsic!();
                let param = expr!();
                LowLevelILOperation::Intrinsic {
                    output, intrinsic, param
                }
                */
            }
            BNLowLevelILOperation_LLIL_UNDEF => LowLevelILOperation::Undef {},
            BNLowLevelILOperation_LLIL_UNIMPL => LowLevelILOperation::Unimpl {},
            BNLowLevelILOperation_LLIL_UNIMPL_MEM => {
                let src = expr!();
                LowLevelILOperation::UnimplMem { src }
            }
            BNLowLevelILOperation_LLIL_FADD => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::Fadd { left, right }
            }
            BNLowLevelILOperation_LLIL_FSUB => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::Fsub { left, right }
            }
            BNLowLevelILOperation_LLIL_FMUL => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::Fmul { left, right }
            }
            BNLowLevelILOperation_LLIL_FDIV => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::Fdiv { left, right }
            }
            BNLowLevelILOperation_LLIL_FSQRT => {
                let src = expr!();
                LowLevelILOperation::Fsqrt { src }
            }
            BNLowLevelILOperation_LLIL_FNEG => {
                let src = expr!();
                LowLevelILOperation::Fneg { src }
            }
            BNLowLevelILOperation_LLIL_FABS => {
                let src = expr!();
                LowLevelILOperation::Fabs { src }
            }
            BNLowLevelILOperation_LLIL_FLOAT_TO_INT => {
                let src = expr!();
                LowLevelILOperation::FloatToInt { src }
            }
            BNLowLevelILOperation_LLIL_INT_TO_FLOAT => {
                let src = expr!();
                LowLevelILOperation::IntToFloat { src }
            }
            BNLowLevelILOperation_LLIL_FLOAT_CONV => {
                let src = expr!();
                LowLevelILOperation::FloatConv { src }
            }
            BNLowLevelILOperation_LLIL_ROUND_TO_INT => {
                let src = expr!();
                LowLevelILOperation::RoundToInt { src }
            }
            BNLowLevelILOperation_LLIL_FLOOR => {
                let src = expr!();
                LowLevelILOperation::Floor { src }
            }
            BNLowLevelILOperation_LLIL_CEIL => {
                let src = expr!();
                LowLevelILOperation::Ceil { src }
            }
            BNLowLevelILOperation_LLIL_FTRUNC => {
                let src = expr!();
                LowLevelILOperation::Ftrunc { src }
            }
            BNLowLevelILOperation_LLIL_FCMP_E => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::FcmpE { left, right }
            }
            BNLowLevelILOperation_LLIL_FCMP_NE => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::FcmpNe { left, right }
            }
            BNLowLevelILOperation_LLIL_FCMP_LT => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::FcmpLt { left, right }
            }
            BNLowLevelILOperation_LLIL_FCMP_LE => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::FcmpLe { left, right }
            }
            BNLowLevelILOperation_LLIL_FCMP_GE => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::FcmpGe { left, right }
            }
            BNLowLevelILOperation_LLIL_FCMP_GT => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::FcmpGt { left, right }
            }
            BNLowLevelILOperation_LLIL_FCMP_O => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::FcmpO { left, right }
            }
            BNLowLevelILOperation_LLIL_FCMP_UO => {
                let left = expr!();
                let right = expr!();
                LowLevelILOperation::FcmpUo { left, right }
            }
            BNLowLevelILOperation_LLIL_SET_REG_SSA => {
                let dest = reg_ssa!();
                let src = expr!();
                LowLevelILOperation::SetRegSsa { dest, src }
            }
            BNLowLevelILOperation_LLIL_SET_REG_SSA_PARTIAL => {
                let full_reg = reg_ssa!();
                let dest = reg!();
                let src = expr!();
                LowLevelILOperation::SetRegSsaPartial {
                    full_reg,
                    dest,
                    src,
                }
            }
            BNLowLevelILOperation_LLIL_SET_REG_SPLIT_SSA => {
                let hi = expr!();
                let lo = expr!();
                let src = expr!();
                LowLevelILOperation::SetRegSplitSsa { hi, lo, src }
            }
            BNLowLevelILOperation_LLIL_SET_REG_STACK_REL_SSA => {
                let stack = expr!();
                let dest = expr!();
                let top = expr!();
                let src = expr!();
                LowLevelILOperation::SetRegStackRelSsa {
                    stack,
                    dest,
                    top,
                    src,
                }
            }
            BNLowLevelILOperation_LLIL_SET_REG_STACK_ABS_SSA => {
                let stack = expr!();
                let dest = reg!();
                let src = expr!();
                LowLevelILOperation::SetRegStackAbsSsa { stack, dest, src }
            }
            BNLowLevelILOperation_LLIL_REG_SPLIT_DEST_SSA => {
                let dest = reg_ssa!();
                LowLevelILOperation::RegSplitDestSsa { dest }
            }
            BNLowLevelILOperation_LLIL_REG_STACK_DEST_SSA => {
                // unimplemented!();
                LowLevelILOperation::Unimpl {}
                /*
                let src = reg_stack_ssa_dest_and_src!();
                LowLevelILOperation::RegStackDestSsa {
                    src
                }
                */
            }
            BNLowLevelILOperation_LLIL_REG_SSA => {
                let src = reg_ssa!();
                LowLevelILOperation::RegSsa { src }
            }
            BNLowLevelILOperation_LLIL_REG_SSA_PARTIAL => {
                let full_reg = reg_ssa!();
                let src = reg!();
                LowLevelILOperation::RegSsaPartial { full_reg, src }
            }
            BNLowLevelILOperation_LLIL_REG_SPLIT_SSA => {
                let hi = reg_ssa!();
                let lo = reg_ssa!();
                LowLevelILOperation::RegSplitSsa { hi, lo }
            }
            BNLowLevelILOperation_LLIL_REG_STACK_REL_SSA => {
                LowLevelILOperation::Unimpl {}
                /*
                let stack = reg_stack_ssa!();
                let src = expr!();
                let top = expr!();
                LowLevelILOperation::RegStackRelSsa {
                    stack, src, top
                }
                */
            }
            BNLowLevelILOperation_LLIL_REG_STACK_ABS_SSA => {
                LowLevelILOperation::Unimpl {}
                /*
                let stack = reg_stack_ssa!();
                let src = reg!();
                LowLevelILOperation::RegStackAbsSsa {
                    stack, src
                }
                */
            }
            BNLowLevelILOperation_LLIL_REG_STACK_FREE_REL_SSA => {
                let stack = expr!();
                let dest = expr!();
                let top = expr!();
                LowLevelILOperation::RegStackFreeRelSsa { stack, dest, top }
            }
            BNLowLevelILOperation_LLIL_REG_STACK_FREE_ABS_SSA => {
                let stack = expr!();
                let dest = reg!();
                LowLevelILOperation::RegStackFreeAbsSsa { stack, dest }
            }
            BNLowLevelILOperation_LLIL_SET_FLAG_SSA => {
                let dest = flag_ssa!();
                let src = expr!();
                LowLevelILOperation::SetFlagSsa { dest, src }
            }
            BNLowLevelILOperation_LLIL_FLAG_SSA => {
                let src = flag_ssa!();
                LowLevelILOperation::FlagSsa { src }
            }
            BNLowLevelILOperation_LLIL_FLAG_BIT_SSA => {
                let src = flag_ssa!();
                let bit = int!();
                LowLevelILOperation::FlagBitSsa { src, bit }
            }
            BNLowLevelILOperation_LLIL_CALL_SSA => {
                let output = expr!();
                let dest = expr!();
                let stack = expr!();
                let param = expr!();
                LowLevelILOperation::CallSsa {
                    output,
                    dest,
                    stack,
                    param,
                }
            }
            BNLowLevelILOperation_LLIL_SYSCALL_SSA => {
                let output = expr!();
                let stack = expr!();
                let param = expr!();
                LowLevelILOperation::SyscallSsa {
                    output,
                    stack,
                    param,
                }
            }
            BNLowLevelILOperation_LLIL_TAILCALL_SSA => {
                let output = expr!();
                let dest = expr!();
                let stack = expr!();
                let param = expr!();
                LowLevelILOperation::TailcallSsa {
                    output,
                    dest,
                    stack,
                    param,
                }
            }
            BNLowLevelILOperation_LLIL_CALL_PARAM => {
                let src = expr_list!();
                LowLevelILOperation::CallParam { src }
            }
            BNLowLevelILOperation_LLIL_CALL_STACK_SSA => {
                let src = reg_ssa!();
                let src_memory = int!();
                LowLevelILOperation::CallStackSsa { src, src_memory }
            }
            BNLowLevelILOperation_LLIL_CALL_OUTPUT_SSA => {
                let dest_memory = int!();
                let dest = reg_ssa_list!();
                LowLevelILOperation::CallOutputSsa { dest_memory, dest }
            }
            BNLowLevelILOperation_LLIL_LOAD_SSA => {
                let src = expr!();
                let src_memory = int!();
                LowLevelILOperation::LoadSsa { src, src_memory }
            }
            BNLowLevelILOperation_LLIL_STORE_SSA => {
                let dest = expr!();
                let dest_memory = int!();
                let src_memory = int!();
                let src = expr!();
                LowLevelILOperation::StoreSsa {
                    dest,
                    dest_memory,
                    src_memory,
                    src,
                }
            }
            BNLowLevelILOperation_LLIL_INTRINSIC_SSA => {
                LowLevelILOperation::Unimpl {}
                /*
                let output = reg_or_flag_ssa_list!();
                let intrinsic = intrinsic!();
                let param = expr!();
                LowLevelILOperation::IntrinsicSsa {
                    output, intrinsic, param
                }
                */
            }
            BNLowLevelILOperation_LLIL_REG_PHI => {
                let dest = reg_ssa!();
                let src = reg_ssa_list!();
                LowLevelILOperation::RegPhi { dest, src }
            }
            BNLowLevelILOperation_LLIL_REG_STACK_PHI => {
                LowLevelILOperation::Unimpl {}
                /*
                let dest = reg_stack_ssa!();
                let src = reg_stack_ssa_list!();
                LowLevelILOperation::RegStackPhi {
                    dest, src
                }
                */
            }
            BNLowLevelILOperation_LLIL_FLAG_PHI => {
                let dest = flag_ssa!();
                let src = flag_ssa_list!();
                LowLevelILOperation::FlagPhi { dest, src }
            }
            BNLowLevelILOperation_LLIL_MEM_PHI => {
                let dest_memory = int!();
                let src_memory = int_list!();
                LowLevelILOperation::MemPhi {
                    dest_memory,
                    src_memory,
                }
            }
            _ => panic!("Unknown operation: {}\n", instr.operation),
        }
    }
}
