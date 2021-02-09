//! Provides High Level IL analysis
#![allow(non_upper_case_globals)]
#![allow(unused_assignments)]
#![allow(unused_variables)]

use core::*;

use anyhow::Result;

use std::slice;
use std::fmt;
use std::sync::Arc;

use crate::unsafe_try;
use crate::function::Function;
use crate::binjastr::BinjaStr;
use crate::architecture::CoreArchitecture;
use crate::basicblock::BasicBlock;
use crate::il::{SSAVariable, Variable, Intrinsic, GotoLabel};
use crate::instruction::InstructionTextToken;
use crate::traits::{FunctionTrait, BasicBlockTrait};
use crate::wrappers::{BinjaHighLevelILFunction, BinjaFunction, BinjaBasicBlock};

#[derive(Clone)]
pub struct HighLevelILFunction {
    handle: Arc<BinjaHighLevelILFunction>,
    func: Arc<BinjaFunction>
}

unsafe impl Send for HighLevelILFunction {}
unsafe impl Sync for HighLevelILFunction {}

impl HighLevelILFunction {
    pub fn new(func: Arc<BinjaFunction>) -> Result<HighLevelILFunction> {
        let handle = unsafe_try!(BNGetFunctionHighLevelIL(**func))?;
        Ok(HighLevelILFunction{ handle: Arc::new(BinjaHighLevelILFunction::new(handle)), func })
    }

    pub fn handle(&self) -> *mut BNHighLevelILFunction {
        **self.handle
    }

    /// Get the owner function for this HLIL function
    pub fn function(&self) -> Function {
        Function::from_arc(self.func.clone())
    }

    /// Get the start of this function
    pub fn start(&self) -> u64 {
        unsafe { BNGetFunctionStart(**self.func) }
    }

    /// Get the addrses of this function
    pub fn address(&self) -> u64 {
        unsafe { BNGetFunctionStart(**self.func) }
    }

    /// Get the number of instructions in this function
    pub fn len(&self) -> u64 {
        unsafe { BNGetHighLevelILInstructionCount(self.handle()) }
    }

    /// Get the architecture for this function
    pub fn arch(&self) -> Result<CoreArchitecture> {
        CoreArchitecture::new_from_func(**self.func)
    }

    /// Get the name of this function
    pub fn name(&self) -> Option<BinjaStr> {
        self.function().name()
    }

    fn get_index_for_instruction(&self, i: u64) -> u64 {
        unsafe { BNGetHighLevelILIndexForInstruction(self.handle(), i) }
    }

    pub fn ssa_form(&self) -> Result<Self> {
        let handle = unsafe_try!(BNGetHighLevelILSSAForm(self.handle()))?;
        Ok(HighLevelILFunction { 
            handle: Arc::new(BinjaHighLevelILFunction::new(handle)), 
            func: self.func.clone() 
        })
    }

    pub fn get_ssa_var_definition(&self, ssavar: SSAVariable) 
            -> Result<HighLevelILInstruction> {
        let expr_index = unsafe {
            BNGetHighLevelILSSAVarDefinition(self.handle(), &ssavar.var.var, 
                ssavar.version as u64)
        };

        print!("Expr index in func: {:#x}\n", expr_index);

        HighLevelILInstruction::from_expr(self.clone(), expr_index, None)
    }

    /// Get the total number of HLILSSA expressions in this function
    pub fn expr_count(&self) -> u64 {
        unsafe {
            BNGetHighLevelILExprCount(self.handle())
        }
    }
}

impl fmt::Display for HighLevelILFunction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl fmt::Debug for HighLevelILFunction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut result = f.debug_struct("HLILFunction");
        result.field("start", &format_args!("{:#x}", self.start()));
        result.field("name", &self.function().name());
        result.finish()
    }
}

impl FunctionTrait for HighLevelILFunction {
    type Ins   = HighLevelILInstruction;
    type Block = HighLevelILBasicBlock;
    type Func  = HighLevelILFunction;

    /// Retrieve a HighLevelILInstruction for a given index
    fn instruction(&self, i: u64) -> Result<Self::Ins> {
        HighLevelILInstruction::from_func_index(self.clone(), i)
    }

    fn blocks(&self) -> Vec<Self::Block> {
        let mut count = 0;

        unsafe { 
            let blocks = BNGetHighLevelILBasicBlockList(self.handle(), &mut count);
            trace!("Function {} returned {} blocks", self, count);

            let blocks_slice = slice::from_raw_parts(blocks, count as usize);

            let result = blocks_slice.iter()
                                     .map(|&b| HighLevelILBasicBlock::new(b, self.clone()).unwrap())
                                     .collect();

            BNFreeBasicBlockList(blocks, count);
            result
        }
    }

    fn ssa_form(&self) -> Result<HighLevelILFunction> {
        self.ssa_form()
    }

    /// Construct the text for a given HighLevelILInstruction index
    fn text(&self, i: u64) -> Result<String> {
        let mut count = 0;

        // Get the raw index for the given instruction index
        let expr_index = self.get_index_for_instruction(i);

        // The resulting line texts
        let mut res_lines: Vec<String> = Vec::new();

        unsafe { 
            let lines = BNGetHighLevelILExprText(self.handle(), expr_index, /* as_ast */ true, 
                                                 &mut count);

            if lines.is_null() || count == 0{
                return Err(anyhow!("Failed to retrieve HLILInstruction tokens"));
            }

            let lines_slice = slice::from_raw_parts(lines, count as usize);

            for line in lines_slice.iter() {
                assert!(line.count < 10000, 
                    "More than 10000 tokens?! Probably wrong struct used with BNDisassemblyTextLine");

                let tokens = slice::from_raw_parts(line.tokens, line.count as usize).to_vec();

                // Collect all of the token elements into one string
                let curr_line = tokens.iter()
                    .fold(String::new(), 
                        |mut acc, &l| { 
                            acc.push_str(&InstructionTextToken::new_from_token(l).text); acc 
                        });

                res_lines.push(curr_line);
            }
            
            BNFreeDisassemblyTextLines(lines, count);
        }

        Ok(res_lines.join("\n"))
    }

    fn expr_text(&self, expr_index: u64) -> Result<String> {
        let mut count = 0;

        // The resulting line texts
        let mut res_lines: Vec<String> = Vec::new();

        unsafe { 
            let lines = BNGetHighLevelILExprText(self.handle(), expr_index, 
                /* as_ast */ true, &mut count);

            if lines.is_null() || count == 0 {
                return Err(anyhow!("Failed to retrieve HLILInstruction tokens"));
            }

            let lines_slice = slice::from_raw_parts(lines, count as usize);

            for line in lines_slice.iter() {
                assert!(line.count < 10000, 
                    "More than 10000 tokens?! Probably wrong struct used with BNDisassemblyTextLine");

                let tokens = slice::from_raw_parts(line.tokens, line.count as usize).to_vec();

                // Collect all of the token elements into one string
                let curr_line = tokens.iter()
                    .fold(String::new(), 
                        |mut acc, &l| { 
                            acc.push_str(&InstructionTextToken::new_from_token(l).text); acc 
                        });

                res_lines.push(curr_line);
            }
            
            BNFreeDisassemblyTextLines(lines, count);
        }

        Ok(res_lines.join("\n"))
    }
}

#[derive(Clone)]
pub struct HighLevelILBasicBlock {
    handle: Arc<BinjaBasicBlock>,
    func: HighLevelILFunction
}

impl HighLevelILBasicBlock {
    pub fn new(handle: *mut BNBasicBlock, func: HighLevelILFunction) -> Result<HighLevelILBasicBlock> {
        let handle = unsafe_try!(BNNewBasicBlockReference(handle))?;
        Ok(HighLevelILBasicBlock{ handle: Arc::new(BinjaBasicBlock::new(handle)), func })
    }

    /// Get the raw basic block for this basic block
    pub fn basic_block(&self) -> BasicBlock {
        BasicBlock::from_arc(self.handle.clone(), self.func.function().clone())
    }
}

impl BasicBlockTrait for HighLevelILBasicBlock {
    type Ins = HighLevelILInstruction;
    type Func = HighLevelILFunction;

    fn handle(&self) -> *mut BNBasicBlock {
        **self.handle
    }

    fn func(&self) -> Option<&Self::Func> {
        Some(&self.func)
    }

    fn raw_function(&self) -> Function {
        self.func.function()
    }
}

unsafe impl Send for HighLevelILBasicBlock {}
unsafe impl Sync for HighLevelILBasicBlock {}

#[derive(Clone)]
pub struct HighLevelILInstruction {
    pub operation: Box<HighLevelILOperation>,
    pub source_operand: u32,
    pub size: u64,
    pub operands: [u64; 5usize],
    pub address: u64,
    pub function: HighLevelILFunction,
    pub expr_index: u64,
    pub instr_index: u64,
    pub text: String,
    pub as_ast: bool
}

unsafe impl Send for HighLevelILInstruction {}
unsafe impl Sync for HighLevelILInstruction {}

impl HighLevelILInstruction {
    /// Get the HLIL basic block containing this instruction
    pub fn hlil_basic_block(&self) -> Option<HighLevelILBasicBlock> {
        for bb in self.function.blocks() {
            if bb.start() <= self.instr_index && bb.end() >= self.instr_index {
                return Some(bb.clone());
            }
        }

        None
    }

    /// Get the `BasicBlock` containing this instruction
    pub fn basic_block(&self) -> Option<BasicBlock> {
        Some(self.hlil_basic_block()?.basic_block())
    }

    /// Get the HLIL instruction from the given `func` at the given `instr_index`
    pub fn from_func_index(func: HighLevelILFunction, instr_index: u64) -> Result<HighLevelILInstruction> { 
        // Get the raw index for the given instruction index
        let expr_index = func.get_index_for_instruction(instr_index);

        HighLevelILInstruction::from_expr(func, expr_index, Some(instr_index))
    }

    /// Get the HLIL instruction from the given internal `expr` at the given `instr_index`
    pub fn from_expr(func: HighLevelILFunction, expr_index: u64, 
            mut instr_index: Option<u64>) -> Result<HighLevelILInstruction> { 
        let as_ast = true;

        // Get the IL for the given index
        let instr = unsafe { BNGetHighLevelILByIndex(func.handle(), expr_index, as_ast) };

        // Check if the expr_index is for a valid instruction
        if unsafe { BNGetHighLevelILInstructionForExpr(func.handle(), expr_index) } == !0 
            && instr.address == 0 && instr.size == 0 && instr.operation == 0 {
            return Err(anyhow!("Invalid instruction expr_index: {}", expr_index));
        }

        // If we have the instruction index, grab the text for that instruction
        let text = func.expr_text(expr_index)?;
        
        // Get the instruction for an expression that has hasn't provided an instruction
        if instr_index.is_none() {
            instr_index = Some(unsafe {
                BNGetHighLevelILInstructionForExpr(func.handle(), expr_index)
            });
        }


        Ok(HighLevelILInstruction {
            operation: Box::new(HighLevelILOperation::from_instr(instr, &func, expr_index)?),
            source_operand: instr.sourceOperand,
            size: instr.size,
            operands: instr.operands,
            address: instr.address,
            function: func,
            expr_index,
            instr_index: instr_index.unwrap(),
            text,
            as_ast
        })
    }

    /// Convert HLIL instruction into HLIL SSA (Alias for ssa_form)
    pub fn ssa(&self) -> Result<HighLevelILInstruction> {
        self.ssa_form()
    }

    /// Convert HLIL instruction into HLIL SSA
    pub fn ssa_form(&self) -> Result<HighLevelILInstruction> {
        let func_ssa = self.function.ssa_form()?;
        unsafe {
            let expr_index = BNGetHighLevelILSSAExprIndex(self.function.handle(), self.expr_index);
            HighLevelILInstruction::from_expr(func_ssa, expr_index, Some(self.instr_index))
        }
    }

    /// Returns all of the `SSAVariable`s that are in this HLIL instruction
    pub fn ssa_vars(&self) -> Option<Vec<SSAVariable>> {
        self.operation.ssa_vars()
    }

    pub fn operation_name(&self) -> String {
        self.operation.name()
    }
}

impl fmt::Display for HighLevelILInstruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // If any operation contains a `ConstPtr` attempt to write the symbol in its place
        /*
        loop {
            let mut instr = self;
            match *instr.operation {
                HighLevelILOperation::AssignUnpack { src, .. } => { instr = src; }
                HighLevelILOperation::CallSsa { dest, .. } => { instr = src; }
                HighLevelILOperation::CallSsa { dest, .. } => { instr = src; }
                _ => break
            }
        }
        */

        // write!(f, "<HLIL::{}>", self.operation_name())
        write!(f, "{}", self.text)
    }
}

impl fmt::Debug for HighLevelILInstruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("HighLevelILInstruction")
            .field("Operation", &self.operation)
            .finish()
    }
}

impl std::cmp::PartialEq for HighLevelILInstruction {
    fn eq(&self, other: &Self) -> bool {
        (self.function.handle(), self.expr_index) == (other.function.handle(), other.expr_index)
    }
}


#[derive(Debug, Clone)]
pub enum HighLevelILOperation {
    Nop { },
    Block { body: Vec<HighLevelILInstruction>, },
    If { condition: HighLevelILInstruction, true_: HighLevelILInstruction, false_: HighLevelILInstruction, },
    While { condition: HighLevelILInstruction, body: HighLevelILInstruction, },
    DoWhile { body: HighLevelILInstruction, condition: HighLevelILInstruction, },
    For { init: HighLevelILInstruction, condition: HighLevelILInstruction, update: HighLevelILInstruction, body: HighLevelILInstruction, },
    Switch { condition: HighLevelILInstruction, default: HighLevelILInstruction, cases: Vec<HighLevelILInstruction>, },
    Case { values: Vec<HighLevelILInstruction>, body: HighLevelILInstruction, },
    Break { },
    Continue { },
    Jump { dest: HighLevelILInstruction, },
    Ret { src: Vec<HighLevelILInstruction>, },
    Noret { },
    Goto { target: GotoLabel, },
    Label { target: GotoLabel, },
    VarDeclare { var: Variable, },
    VarInit { dest: Variable, src: HighLevelILInstruction, },
    Assign { dest: HighLevelILInstruction, src: HighLevelILInstruction, },
    AssignUnpack { dest: Vec<HighLevelILInstruction>, src: HighLevelILInstruction, },
    Var { var: Variable, },
    StructField { src: HighLevelILInstruction, offset: u64, member_index: Option<u64>, },
    ArrayIndex { src: HighLevelILInstruction, index: HighLevelILInstruction, },
    Split { high: HighLevelILInstruction, low: HighLevelILInstruction, },
    Deref { src: HighLevelILInstruction, },
    DerefField { src: HighLevelILInstruction, offset: u64, member_index: Option<u64>, },
    AddressOf { src: HighLevelILInstruction, },
    Const { constant: u64, },
    ConstPtr { constant: u64, },
    ExternPtr { constant: u64, offset: u64, },
    FloatConst { constant: f64 },
    Import { constant: u64, },
    Add { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    Adc { left: HighLevelILInstruction, right: HighLevelILInstruction, carry: HighLevelILInstruction, },
    Sub { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    Sbb { left: HighLevelILInstruction, right: HighLevelILInstruction, carry: HighLevelILInstruction, },
    And { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    Or { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    Xor { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    Lsl { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    Lsr { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    Asr { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    Rol { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    Rlc { left: HighLevelILInstruction, right: HighLevelILInstruction, carry: HighLevelILInstruction, },
    Ror { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    Rrc { left: HighLevelILInstruction, right: HighLevelILInstruction, carry: HighLevelILInstruction, },
    Mul { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    MuluDp { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    MulsDp { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    Divu { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    DivuDp { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    Divs { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    DivsDp { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    Modu { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    ModuDp { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    Mods { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    ModsDp { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    Neg { src: HighLevelILInstruction, },
    Not { src: HighLevelILInstruction, },
    SignExtend { src: HighLevelILInstruction, },
    ZeroExtend { src: HighLevelILInstruction, },
    LowPart { src: HighLevelILInstruction, },
    Call { dest: HighLevelILInstruction, params: Vec<HighLevelILInstruction>, },
    Equals { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    NotEquals { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    SignedLessThan { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    UnsignedLessThan { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    SignedLessThanEquals { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    UnsignedLessThanEquals { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    SignedGreaterThanEquals { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    UnsignedGreaterThanEquals { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    SsignedGreaterThan { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    UnsignedGreaterThan { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    TestBit { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    BoolToInt { src: HighLevelILInstruction, },
    AddOverflow { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    Syscall { params: Vec<HighLevelILInstruction>, },
    Tailcall { dest: HighLevelILInstruction, params: Vec<HighLevelILInstruction>, },
    Intrinsic { intrinsic: Intrinsic, params: Vec<HighLevelILInstruction>, },
    Bp { },
    Trap { vector: u64, },
    Undef { },
    Unimpl { },
    UnimplMem { src: HighLevelILInstruction, },
    Fadd { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    Fsub { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    Fmul { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    Fdiv { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    Fsqrt { src: HighLevelILInstruction, },
    Fneg { src: HighLevelILInstruction, },
    Fabs { src: HighLevelILInstruction, },
    FloatToInt { src: HighLevelILInstruction, },
    IntToFloat { src: HighLevelILInstruction, },
    FloatConv { src: HighLevelILInstruction, },
    RoundToInt { src: HighLevelILInstruction, },
    Floor { src: HighLevelILInstruction, },
    Ceil { src: HighLevelILInstruction, },
    Ftrunc { src: HighLevelILInstruction, },
    FcmpE { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    FcmpNe { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    FcmpLt { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    FcmpLe { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    FcmpGe { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    FcmpGt { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    FcmpO { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    FcmpUo { left: HighLevelILInstruction, right: HighLevelILInstruction, },
    WhileSsa { condition_phi: HighLevelILInstruction, condition: HighLevelILInstruction, body: HighLevelILInstruction, },
    DoWhileSsa { body: HighLevelILInstruction, condition_phi: HighLevelILInstruction, condition: HighLevelILInstruction, },
    ForSsa { init: HighLevelILInstruction, condition_phi: HighLevelILInstruction, condition: HighLevelILInstruction, update: HighLevelILInstruction, body: HighLevelILInstruction, },
    VarInitSsa { dest: SSAVariable, src: HighLevelILInstruction, },
    AssignMemSsa { dest: HighLevelILInstruction, dest_memory: u64, src: HighLevelILInstruction, src_memory: u64, },
    AssignUnpackMemSsa { dest: Vec<HighLevelILInstruction>, dest_memory: u64, src: HighLevelILInstruction, src_memory: u64, },
    VarSsa { var: SSAVariable, },
    ArrayIndexSsa { src: HighLevelILInstruction, src_memory: u64, index: HighLevelILInstruction, },
    DerefSsa { src: HighLevelILInstruction, src_memory: u64, },
    DerefFieldSsa { src: HighLevelILInstruction, src_memory: u64, offset: u64, member_index: Option<u64>, },
    CallSsa { dest: HighLevelILInstruction, params: Vec<HighLevelILInstruction>, dest_memory: u64, src_memory: u64, },
    SyscallSsa { params: Vec<HighLevelILInstruction>, dest_memory: u64, src_memory: u64, },
    IntrinsicSsa { intrinsic: Intrinsic, params: Vec<HighLevelILInstruction>, dest_memory: u64, src_memory: u64, },
    VarPhi { dest: SSAVariable, src: Vec<SSAVariable>, },
    MemPhi { dest: u64, src: Vec<u64>, },
}
impl HighLevelILOperation {
    pub fn from_instr(instr: BNHighLevelILInstruction, func: &HighLevelILFunction, expr_index: u64)
            -> Result<HighLevelILOperation> {
        let arch = func.arch().expect("Failed to get arch for HLIL").clone();
        let mut operand_index = 0;

        // Macros used to define each of the types of arguments in each operation
        macro_rules! expr {
            () => {{
                let res = HighLevelILInstruction::from_expr(func.clone(), instr.operands[operand_index], None)?;
                operand_index += 1;
                res
            }}
        }

        macro_rules! float {
            () => {{
                // Extract the value from the operand
                let res = match instr.size {
                    4 => f32::from_bits(instr.operands[operand_index] as u32) as f64,
                    8 => f64::from_bits(instr.operands[operand_index]),
                    _ => unreachable!()
                };
                operand_index += 1;
                res
            }}
        }

        macro_rules! int {
            () => {{
                /*
                print!("{:#x} {:#x}\n", 
                    instr.operands[operand_index] & ((1 << 63) - 1),
                    instr.operands[operand_index] & (1 << 63));
                */

                let value = instr.operands[operand_index] ;
                let res = (value & ((1 << 63) - 1)).wrapping_sub(value & (1 << 63));
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
                    let operands = BNHighLevelILGetOperandList(func.handle(), expr_index, 
                                                               operand_index as u64, &mut count);

                    operand_index += 2;

                    // Get the slice from the found pointer
                    let operands_slice = slice::from_raw_parts(operands, count as usize);

                    // Create each instruction
                    for op in operands_slice {
                        let i = HighLevelILInstruction::from_expr(func.clone(), *op, None)?;
                        instrs.push(i);
                    }

                    // Free the binja core pointer
                    BNHighLevelILFreeOperandList(operands);
                }

                instrs
            }}
        }

        macro_rules! int_list {
            () => {{
                // Generate the int list from the binja core
                let mut count = 0;
                let mut int_list = Vec::new();
            
                unsafe { 
                    let operands = BNHighLevelILGetOperandList(func.handle(), expr_index, 
                                                            operand_index as u64, &mut count);

                    operand_index += 1;

                    let operands_slice = slice::from_raw_parts(operands, count as usize);

                    for i in 0..count {
                        int_list.push(operands_slice[i as usize]);
                    }

                    BNHighLevelILFreeOperandList(operands);
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

        macro_rules! var_ssa_list {
            () => {{
                let mut count = 0;
                let mut vars = Vec::new();
            
                unsafe { 
                    let operands = BNHighLevelILGetOperandList(func.handle(), expr_index, 
                                                                 operand_index as u64, &mut count);

                    operand_index += 2;

                    let operands_slice = slice::from_raw_parts(operands, count as usize);

                    for i in (0..count).step_by(2) {
                        let id      = operands_slice[i as usize];
                        let version = operands_slice[i as usize + 1] as u32;
                        let bnvar = BNFromVariableIdentifier(id);
                        let var = Variable::new(func.function(), bnvar);
                        vars.push(SSAVariable::new(var, version));
                    }

                    BNHighLevelILFreeOperandList(operands);
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

        macro_rules! member_index {
            () => {{
                let value = instr.operands[operand_index];
                let mut res = Some(value);
                if (value & (1 << 63)) != 0 {
                    res = None;
                }
                operand_index += 1;
                res
            }}
        }

        macro_rules! label {
            () => {{
                let res = GotoLabel::new(func.clone(), instr.operands[operand_index]);
                operand_index += 1;
                res
            }}
        }

        Ok(match instr.operation {
            BNHighLevelILOperation_HLIL_NOP => {
                HighLevelILOperation::Nop {
                    
                }
            }
            BNHighLevelILOperation_HLIL_BLOCK => {
                let body = expr_list!();
                HighLevelILOperation::Block {
                    body
                }
            }
            BNHighLevelILOperation_HLIL_IF => {
                let condition = expr!();
                let true_ = expr!();
                let false_ = expr!();
                HighLevelILOperation::If {
                    condition, true_, false_
                }
            }
            BNHighLevelILOperation_HLIL_WHILE => {
                let condition = expr!();
                let body = expr!();
                HighLevelILOperation::While {
                    condition, body
                }
            }
            BNHighLevelILOperation_HLIL_DO_WHILE => {
                let body = expr!();
                let condition = expr!();
                HighLevelILOperation::DoWhile {
                    body, condition
                }
            }
            BNHighLevelILOperation_HLIL_FOR => {
                let init = expr!();
                let condition = expr!();
                let update = expr!();
                let body = expr!();
                HighLevelILOperation::For {
                    init, condition, update, body
                }
            }
            BNHighLevelILOperation_HLIL_SWITCH => {
                let condition = expr!();
                let default = expr!();
                let cases = expr_list!();
                HighLevelILOperation::Switch {
                    condition, default, cases
                }
            }
            BNHighLevelILOperation_HLIL_CASE => {
                let values = expr_list!();
                let body = expr!();
                HighLevelILOperation::Case {
                    values, body
                }
            }
            BNHighLevelILOperation_HLIL_BREAK => {
                HighLevelILOperation::Break {
                    
                }
            }
            BNHighLevelILOperation_HLIL_CONTINUE => {
                HighLevelILOperation::Continue {
                    
                }
            }
            BNHighLevelILOperation_HLIL_JUMP => {
                let dest = expr!();
                HighLevelILOperation::Jump {
                    dest
                }
            }
            BNHighLevelILOperation_HLIL_RET => {
                let src = expr_list!();
                HighLevelILOperation::Ret {
                    src
                }
            }
            BNHighLevelILOperation_HLIL_NORET => {
                HighLevelILOperation::Noret {
                    
                }
            }
            BNHighLevelILOperation_HLIL_GOTO => {
                let target = label!();
                HighLevelILOperation::Goto {
                    target
                }
            }
            BNHighLevelILOperation_HLIL_LABEL => {
                let target = label!();
                HighLevelILOperation::Label {
                    target
                }
            }
            BNHighLevelILOperation_HLIL_VAR_DECLARE => {
                let var = var!();
                HighLevelILOperation::VarDeclare {
                    var
                }
            }
            BNHighLevelILOperation_HLIL_VAR_INIT => {
                let dest = var!();
                let src = expr!();
                HighLevelILOperation::VarInit {
                    dest, src
                }
            }
            BNHighLevelILOperation_HLIL_ASSIGN => {
                let dest = expr!();
                let src = expr!();
                HighLevelILOperation::Assign {
                    dest, src
                }
            }
            BNHighLevelILOperation_HLIL_ASSIGN_UNPACK => {
                let dest = expr_list!();
                let src = expr!();
                HighLevelILOperation::AssignUnpack {
                    dest, src
                }
            }
            BNHighLevelILOperation_HLIL_VAR => {
                let var = var!();
                HighLevelILOperation::Var {
                    var
                }
            }
            BNHighLevelILOperation_HLIL_STRUCT_FIELD => {
                let src = expr!();
                let offset = int!();
                let member_index = member_index!();
                HighLevelILOperation::StructField {
                    src, offset, member_index
                }
            }
            BNHighLevelILOperation_HLIL_ARRAY_INDEX => {
                let src = expr!();
                let index = expr!();
                HighLevelILOperation::ArrayIndex {
                    src, index
                }
            }
            BNHighLevelILOperation_HLIL_SPLIT => {
                let high = expr!();
                let low = expr!();
                HighLevelILOperation::Split {
                    high, low
                }
            }
            BNHighLevelILOperation_HLIL_DEREF => {
                let src = expr!();
                HighLevelILOperation::Deref {
                    src
                }
            }
            BNHighLevelILOperation_HLIL_DEREF_FIELD => {
                let src = expr!();
                let offset = int!();
                let member_index = member_index!();
                HighLevelILOperation::DerefField {
                    src, offset, member_index
                }
            }
            BNHighLevelILOperation_HLIL_ADDRESS_OF => {
                let src = expr!();
                HighLevelILOperation::AddressOf {
                    src
                }
            }
            BNHighLevelILOperation_HLIL_CONST => {
                let constant = int!();
                HighLevelILOperation::Const {
                    constant
                }
            }
            BNHighLevelILOperation_HLIL_CONST_PTR => {
                let constant = int!();
                HighLevelILOperation::ConstPtr {
                    constant
                }
            }
            BNHighLevelILOperation_HLIL_EXTERN_PTR => {
                let constant = int!();
                let offset = int!();
                HighLevelILOperation::ExternPtr {
                    constant, offset
                }
            }
            BNHighLevelILOperation_HLIL_FLOAT_CONST => {
                let constant = float!();
                HighLevelILOperation::FloatConst {
                    constant
                }
            }
            BNHighLevelILOperation_HLIL_IMPORT => {
                let constant = int!();
                HighLevelILOperation::Import {
                    constant
                }
            }
            BNHighLevelILOperation_HLIL_ADD => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::Add {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_ADC => {
                let left = expr!();
                let right = expr!();
                let carry = expr!();
                HighLevelILOperation::Adc {
                    left, right, carry
                }
            }
            BNHighLevelILOperation_HLIL_SUB => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::Sub {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_SBB => {
                let left = expr!();
                let right = expr!();
                let carry = expr!();
                HighLevelILOperation::Sbb {
                    left, right, carry
                }
            }
            BNHighLevelILOperation_HLIL_AND => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::And {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_OR => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::Or {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_XOR => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::Xor {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_LSL => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::Lsl {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_LSR => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::Lsr {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_ASR => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::Asr {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_ROL => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::Rol {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_RLC => {
                let left = expr!();
                let right = expr!();
                let carry = expr!();
                HighLevelILOperation::Rlc {
                    left, right, carry
                }
            }
            BNHighLevelILOperation_HLIL_ROR => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::Ror {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_RRC => {
                let left = expr!();
                let right = expr!();
                let carry = expr!();
                HighLevelILOperation::Rrc {
                    left, right, carry
                }
            }
            BNHighLevelILOperation_HLIL_MUL => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::Mul {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_MULU_DP => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::MuluDp {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_MULS_DP => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::MulsDp {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_DIVU => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::Divu {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_DIVU_DP => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::DivuDp {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_DIVS => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::Divs {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_DIVS_DP => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::DivsDp {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_MODU => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::Modu {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_MODU_DP => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::ModuDp {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_MODS => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::Mods {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_MODS_DP => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::ModsDp {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_NEG => {
                let src = expr!();
                HighLevelILOperation::Neg {
                    src
                }
            }
            BNHighLevelILOperation_HLIL_NOT => {
                let src = expr!();
                HighLevelILOperation::Not {
                    src
                }
            }
            BNHighLevelILOperation_HLIL_SX => {
                let src = expr!();
                HighLevelILOperation::SignExtend {
                    src
                }
            }
            BNHighLevelILOperation_HLIL_ZX => {
                let src = expr!();
                HighLevelILOperation::ZeroExtend {
                    src
                }
            }
            BNHighLevelILOperation_HLIL_LOW_PART => {
                let src = expr!();
                HighLevelILOperation::LowPart {
                    src
                }
            }
            BNHighLevelILOperation_HLIL_CALL => {
                let dest = expr!();
                let params = expr_list!();
                HighLevelILOperation::Call {
                    dest, params
                }
            }
            BNHighLevelILOperation_HLIL_CMP_E => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::Equals {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_CMP_NE => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::NotEquals {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_CMP_SLT => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::SignedLessThan {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_CMP_ULT => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::UnsignedLessThan {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_CMP_SLE => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::SignedLessThanEquals {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_CMP_ULE => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::UnsignedLessThanEquals {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_CMP_SGE => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::SignedGreaterThanEquals {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_CMP_UGE => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::UnsignedGreaterThanEquals {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_CMP_SGT => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::SsignedGreaterThan {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_CMP_UGT => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::UnsignedGreaterThan {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_TEST_BIT => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::TestBit {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_BOOL_TO_INT => {
                let src = expr!();
                HighLevelILOperation::BoolToInt {
                    src
                }
            }
            BNHighLevelILOperation_HLIL_ADD_OVERFLOW => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::AddOverflow {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_SYSCALL => {
                let params = expr_list!();
                HighLevelILOperation::Syscall {
                    params
                }
            }
            BNHighLevelILOperation_HLIL_TAILCALL => {
                let dest = expr!();
                let params = expr_list!();
                HighLevelILOperation::Tailcall {
                    dest, params
                }
            }
            BNHighLevelILOperation_HLIL_INTRINSIC => {
                let intrinsic = intrinsic!();
                let params = expr_list!();
                HighLevelILOperation::Intrinsic {
                    intrinsic, params
                }
            }
            BNHighLevelILOperation_HLIL_BP => {
                HighLevelILOperation::Bp {
                    
                }
            }
            BNHighLevelILOperation_HLIL_TRAP => {
                let vector = int!();
                HighLevelILOperation::Trap {
                    vector
                }
            }
            BNHighLevelILOperation_HLIL_UNDEF => {
                HighLevelILOperation::Undef {
                    
                }
            }
            BNHighLevelILOperation_HLIL_UNIMPL => {
                HighLevelILOperation::Unimpl {
                    
                }
            }
            BNHighLevelILOperation_HLIL_UNIMPL_MEM => {
                let src = expr!();
                HighLevelILOperation::UnimplMem {
                    src
                }
            }
            BNHighLevelILOperation_HLIL_FADD => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::Fadd {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_FSUB => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::Fsub {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_FMUL => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::Fmul {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_FDIV => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::Fdiv {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_FSQRT => {
                let src = expr!();
                HighLevelILOperation::Fsqrt {
                    src
                }
            }
            BNHighLevelILOperation_HLIL_FNEG => {
                let src = expr!();
                HighLevelILOperation::Fneg {
                    src
                }
            }
            BNHighLevelILOperation_HLIL_FABS => {
                let src = expr!();
                HighLevelILOperation::Fabs {
                    src
                }
            }
            BNHighLevelILOperation_HLIL_FLOAT_TO_INT => {
                let src = expr!();
                HighLevelILOperation::FloatToInt {
                    src
                }
            }
            BNHighLevelILOperation_HLIL_INT_TO_FLOAT => {
                let src = expr!();
                HighLevelILOperation::IntToFloat {
                    src
                }
            }
            BNHighLevelILOperation_HLIL_FLOAT_CONV => {
                let src = expr!();
                HighLevelILOperation::FloatConv {
                    src
                }
            }
            BNHighLevelILOperation_HLIL_ROUND_TO_INT => {
                let src = expr!();
                HighLevelILOperation::RoundToInt {
                    src
                }
            }
            BNHighLevelILOperation_HLIL_FLOOR => {
                let src = expr!();
                HighLevelILOperation::Floor {
                    src
                }
            }
            BNHighLevelILOperation_HLIL_CEIL => {
                let src = expr!();
                HighLevelILOperation::Ceil {
                    src
                }
            }
            BNHighLevelILOperation_HLIL_FTRUNC => {
                let src = expr!();
                HighLevelILOperation::Ftrunc {
                    src
                }
            }
            BNHighLevelILOperation_HLIL_FCMP_E => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::FcmpE {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_FCMP_NE => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::FcmpNe {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_FCMP_LT => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::FcmpLt {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_FCMP_LE => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::FcmpLe {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_FCMP_GE => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::FcmpGe {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_FCMP_GT => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::FcmpGt {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_FCMP_O => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::FcmpO {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_FCMP_UO => {
                let left = expr!();
                let right = expr!();
                HighLevelILOperation::FcmpUo {
                    left, right
                }
            }
            BNHighLevelILOperation_HLIL_WHILE_SSA => {
                let condition_phi = expr!();
                let condition = expr!();
                let body = expr!();
                HighLevelILOperation::WhileSsa {
                    condition_phi, condition, body
                }
            }
            BNHighLevelILOperation_HLIL_DO_WHILE_SSA => {
                let body = expr!();
                let condition_phi = expr!();
                let condition = expr!();
                HighLevelILOperation::DoWhileSsa {
                    body, condition_phi, condition
                }
            }
            BNHighLevelILOperation_HLIL_FOR_SSA => {
                let init = expr!();
                let condition_phi = expr!();
                let condition = expr!();
                let update = expr!();
                let body = expr!();
                HighLevelILOperation::ForSsa {
                    init, condition_phi, condition, update, body
                }
            }
            BNHighLevelILOperation_HLIL_VAR_INIT_SSA => {
                let dest = var_ssa!();
                let src = expr!();
                HighLevelILOperation::VarInitSsa {
                    dest, src
                }
            }
            BNHighLevelILOperation_HLIL_ASSIGN_MEM_SSA => {
                let dest = expr!();
                let dest_memory = int!();
                let src = expr!();
                let src_memory = int!();
                HighLevelILOperation::AssignMemSsa {
                    dest, dest_memory, src, src_memory
                }
            }
            BNHighLevelILOperation_HLIL_ASSIGN_UNPACK_MEM_SSA => {
                let dest = expr_list!();
                let dest_memory = int!();
                let src = expr!();
                let src_memory = int!();
                HighLevelILOperation::AssignUnpackMemSsa {
                    dest, dest_memory, src, src_memory
                }
            }
            BNHighLevelILOperation_HLIL_VAR_SSA => {
                let var = var_ssa!();
                HighLevelILOperation::VarSsa {
                    var
                }
            }
            BNHighLevelILOperation_HLIL_ARRAY_INDEX_SSA => {
                let src = expr!();
                let src_memory = int!();
                let index = expr!();
                HighLevelILOperation::ArrayIndexSsa {
                    src, src_memory, index
                }
            }
            BNHighLevelILOperation_HLIL_DEREF_SSA => {
                let src = expr!();
                let src_memory = int!();
                HighLevelILOperation::DerefSsa {
                    src, src_memory
                }
            }
            BNHighLevelILOperation_HLIL_DEREF_FIELD_SSA => {
                let src = expr!();
                let src_memory = int!();
                let offset = int!();
                let member_index = member_index!();
                HighLevelILOperation::DerefFieldSsa {
                    src, src_memory, offset, member_index
                }
            }
            BNHighLevelILOperation_HLIL_CALL_SSA => {
                let dest = expr!();
                let params = expr_list!();
                let dest_memory = int!();
                let src_memory = int!();
                HighLevelILOperation::CallSsa {
                    dest, params, dest_memory, src_memory
                }
            }
            BNHighLevelILOperation_HLIL_SYSCALL_SSA => {
                let params = expr_list!();
                let dest_memory = int!();
                let src_memory = int!();
                HighLevelILOperation::SyscallSsa {
                    params, dest_memory, src_memory
                }
            }
            BNHighLevelILOperation_HLIL_INTRINSIC_SSA => {
                let intrinsic = intrinsic!();
                let params = expr_list!();
                let dest_memory = int!();
                let src_memory = int!();
                HighLevelILOperation::IntrinsicSsa {
                    intrinsic, params, dest_memory, src_memory
                }
            }
            BNHighLevelILOperation_HLIL_VAR_PHI => {
                let dest = var_ssa!();
                let src = var_ssa_list!();
                HighLevelILOperation::VarPhi {
                    dest, src
                }
            }
            BNHighLevelILOperation_HLIL_MEM_PHI => {
                let dest = int!();
                let src = int_list!();
                HighLevelILOperation::MemPhi {
                    dest, src
                }
            }
            _ => unreachable!()
        })
    }

    /// Return all the SSAVarible in this current HighLevelILOperation
    pub fn ssa_vars(&self) -> Option<Vec<SSAVariable>> {
        let res = match self {
        HighLevelILOperation::Nop {  }=> {
            Vec::new()
        }
        HighLevelILOperation::Block { /* Vec<HighLevelILInstruction> */ body }=> {
            let mut res = Vec::new();
            for instr in body {
                if let Some(vars) = instr.ssa_vars() {
                    for var in vars {
                        res.push(var.clone());
                    }
                }
            }
            res
        }
        HighLevelILOperation::If { /* HighLevelILInstruction */ condition, /* HighLevelILInstruction */ true_, /* HighLevelILInstruction */ false_ }=> {
            let mut res = Vec::new();
            if let Some(vars) = condition.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = true_.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = false_.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::While { /* HighLevelILInstruction */ condition, /* HighLevelILInstruction */ body }=> {
            let mut res = Vec::new();
            if let Some(vars) = condition.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = body.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::WhileSsa { /* HighLevelILInstruction */ condition_phi, /* HighLevelILInstruction */ condition, /* HighLevelILInstruction */ body }=> {
            let mut res = Vec::new();
            if let Some(vars) = condition_phi.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = condition.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = body.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::DoWhile { /* HighLevelILInstruction */ body, /* HighLevelILInstruction */ condition }=> {
            let mut res = Vec::new();
            if let Some(vars) = body.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = condition.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::DoWhileSsa { /* HighLevelILInstruction */ body, /* HighLevelILInstruction */ condition_phi, /* HighLevelILInstruction */ condition }=> {
            let mut res = Vec::new();
            if let Some(vars) = body.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = condition_phi.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = condition.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::For { /* HighLevelILInstruction */ init, /* HighLevelILInstruction */ condition, /* HighLevelILInstruction */ update, /* HighLevelILInstruction */ body }=> {
            let mut res = Vec::new();
            if let Some(vars) = init.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = condition.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = update.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = body.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::ForSsa { /* HighLevelILInstruction */ init, /* HighLevelILInstruction */ condition_phi, /* HighLevelILInstruction */ condition, /* HighLevelILInstruction */ update, /* HighLevelILInstruction */ body }=> {
            let mut res = Vec::new();
            if let Some(vars) = init.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = condition_phi.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = condition.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = update.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = body.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Switch { /* HighLevelILInstruction */ condition, /* HighLevelILInstruction */ default, /* Vec<HighLevelILInstruction> */ cases }=> {
            let mut res = Vec::new();
            if let Some(vars) = condition.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = default.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            for instr in cases {
                if let Some(vars) = instr.ssa_vars() {
                    for var in vars {
                        res.push(var.clone());
                    }
                }
            }
            res
        }
        HighLevelILOperation::Case { /* Vec<HighLevelILInstruction> */ values, /* HighLevelILInstruction */ body }=> {
            let mut res = Vec::new();
            for instr in values {
                if let Some(vars) = instr.ssa_vars() {
                    for var in vars {
                        res.push(var.clone());
                    }
                }
            }
            if let Some(vars) = body.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Break {  }=> {
            Vec::new()
        }
        HighLevelILOperation::Continue {  }=> {
            Vec::new()
        }
        HighLevelILOperation::Jump { /* HighLevelILInstruction */ dest }=> {
            let mut res = Vec::new();
            if let Some(vars) = dest.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Ret { /* Vec<HighLevelILInstruction> */ src }=> {
            let mut res = Vec::new();
            for instr in src {
                if let Some(vars) = instr.ssa_vars() {
                    for var in vars {
                        res.push(var.clone());
                    }
                }
            }
            res
        }
        HighLevelILOperation::Noret {  }=> {
            Vec::new()
        }
        HighLevelILOperation::Goto { /* GotoLabel */ target }=> {
            Vec::new()
        }
        HighLevelILOperation::Label { /* GotoLabel */ target }=> {
            Vec::new()
        }
        HighLevelILOperation::VarDeclare { /* var */ var }=> {
            Vec::new()
        }
        HighLevelILOperation::VarInit { /* var */ dest, /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::VarInitSsa { /* var_ssa */ dest, /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            res.push(dest.clone());
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Assign { /* HighLevelILInstruction */ dest, /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            if let Some(vars) = dest.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::AssignUnpack { /* Vec<HighLevelILInstruction> */ dest, /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            for instr in dest {
                if let Some(vars) = instr.ssa_vars() {
                    for var in vars {
                        res.push(var.clone());
                    }
                }
            }
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::AssignMemSsa { /* HighLevelILInstruction */ dest, /* u64 */ dest_memory, /* HighLevelILInstruction */ src, /* u64 */ src_memory }=> {
            let mut res = Vec::new();
            if let Some(vars) = dest.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::AssignUnpackMemSsa { /* Vec<HighLevelILInstruction> */ dest, /* u64 */ dest_memory, /* HighLevelILInstruction */ src, /* u64 */ src_memory }=> {
            let mut res = Vec::new();
            for instr in dest {
                if let Some(vars) = instr.ssa_vars() {
                    for var in vars {
                        res.push(var.clone());
                    }
                }
            }
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Var { /* var */ var }=> {
            Vec::new()
        }
        HighLevelILOperation::VarSsa { /* var_ssa */ var }=> {
            let mut res = Vec::new();
            res.push(var.clone());
            res
        }
        HighLevelILOperation::VarPhi { /* var_ssa */ dest, /* var_ssa_list */ src }=> {
            let mut res = Vec::new();
            res.push(dest.clone());
            for var in src {
                res.push(var.clone());
            }
            res
        }
        HighLevelILOperation::MemPhi { /* u64 */ dest, /* Vec<u64> */ src }=> {
            Vec::new()
        }
        HighLevelILOperation::StructField { /* HighLevelILInstruction */ src, /* u64 */ offset, /* u64 */ member_index }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::ArrayIndex { /* HighLevelILInstruction */ src, /* HighLevelILInstruction */ index }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = index.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::ArrayIndexSsa { /* HighLevelILInstruction */ src, /* u64 */ src_memory, /* HighLevelILInstruction */ index }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = index.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Split { /* HighLevelILInstruction */ high, /* HighLevelILInstruction */ low }=> {
            let mut res = Vec::new();
            if let Some(vars) = high.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = low.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Deref { /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::DerefField { /* HighLevelILInstruction */ src, /* u64 */ offset, /* u64 */ member_index }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::DerefSsa { /* HighLevelILInstruction */ src, /* u64 */ src_memory }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::DerefFieldSsa { /* HighLevelILInstruction */ src, /* u64 */ src_memory, /* u64 */ offset, /* u64 */ member_index }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::AddressOf { /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Const { /* u64 */ constant }=> {
            Vec::new()
        }
        HighLevelILOperation::ConstPtr { /* u64 */ constant }=> {
            Vec::new()
        }
        HighLevelILOperation::ExternPtr { /* u64 */ constant, /* u64 */ offset }=> {
            Vec::new()
        }
        HighLevelILOperation::FloatConst { /* float */ constant }=> {
            Vec::new()
        }
        HighLevelILOperation::Import { /* u64 */ constant }=> {
            Vec::new()
        }
        HighLevelILOperation::Add { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Adc { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right, /* HighLevelILInstruction */ carry }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = carry.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Sub { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Sbb { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right, /* HighLevelILInstruction */ carry }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = carry.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::And { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Or { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Xor { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Lsl { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Lsr { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Asr { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Rol { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Rlc { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right, /* HighLevelILInstruction */ carry }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = carry.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Ror { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Rrc { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right, /* HighLevelILInstruction */ carry }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = carry.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Mul { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::MuluDp { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::MulsDp { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Divu { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::DivuDp { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Divs { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::DivsDp { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Modu { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::ModuDp { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Mods { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::ModsDp { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Neg { /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Not { /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::SignExtend { /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::ZeroExtend { /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::LowPart { /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Call { /* HighLevelILInstruction */ dest, /* Vec<HighLevelILInstruction> */ params }=> {
            let mut res = Vec::new();
            if let Some(vars) = dest.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            for instr in params {
                if let Some(vars) = instr.ssa_vars() {
                    for var in vars {
                        res.push(var.clone());
                    }
                }
            }
            res
        }
        HighLevelILOperation::CallSsa { /* HighLevelILInstruction */ dest, /* Vec<HighLevelILInstruction> */ params, /* u64 */ dest_memory, /* u64 */ src_memory }=> {
            let mut res = Vec::new();
            if let Some(vars) = dest.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            for instr in params {
                if let Some(vars) = instr.ssa_vars() {
                    for var in vars {
                        res.push(var.clone());
                    }
                }
            }
            res
        }
        HighLevelILOperation::Equals { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::NotEquals { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::SignedLessThan { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::UnsignedLessThan { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::SignedLessThanEquals { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::UnsignedLessThanEquals { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::SignedGreaterThanEquals { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::UnsignedGreaterThanEquals { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::SsignedGreaterThan { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::UnsignedGreaterThan { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::TestBit { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::BoolToInt { /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::AddOverflow { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Syscall { /* Vec<HighLevelILInstruction> */ params }=> {
            let mut res = Vec::new();
            for instr in params {
                if let Some(vars) = instr.ssa_vars() {
                    for var in vars {
                        res.push(var.clone());
                    }
                }
            }
            res
        }
        HighLevelILOperation::SyscallSsa { /* Vec<HighLevelILInstruction> */ params, /* u64 */ dest_memory, /* u64 */ src_memory }=> {
            let mut res = Vec::new();
            for instr in params {
                if let Some(vars) = instr.ssa_vars() {
                    for var in vars {
                        res.push(var.clone());
                    }
                }
            }
            res
        }
        HighLevelILOperation::Tailcall { /* HighLevelILInstruction */ dest, /* Vec<HighLevelILInstruction> */ params }=> {
            let mut res = Vec::new();
            if let Some(vars) = dest.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            for instr in params {
                if let Some(vars) = instr.ssa_vars() {
                    for var in vars {
                        res.push(var.clone());
                    }
                }
            }
            res
        }
        HighLevelILOperation::Bp {  }=> {
            Vec::new()
        }
        HighLevelILOperation::Trap { /* u64 */ vector }=> {
            Vec::new()
        }
        HighLevelILOperation::Intrinsic { /* intrinsic */ intrinsic, /* Vec<HighLevelILInstruction> */ params }=> {
            let mut res = Vec::new();
            for instr in params {
                if let Some(vars) = instr.ssa_vars() {
                    for var in vars {
                        res.push(var.clone());
                    }
                }
            }
            res
        }
        HighLevelILOperation::IntrinsicSsa { /* intrinsic */ intrinsic, /* Vec<HighLevelILInstruction> */ params, /* u64 */ dest_memory, /* u64 */ src_memory }=> {
            let mut res = Vec::new();
            for instr in params {
                if let Some(vars) = instr.ssa_vars() {
                    for var in vars {
                        res.push(var.clone());
                    }
                }
            }
            res
        }
        HighLevelILOperation::Undef {  }=> {
            Vec::new()
        }
        HighLevelILOperation::Unimpl {  }=> {
            Vec::new()
        }
        HighLevelILOperation::UnimplMem { /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Fadd { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Fsub { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Fmul { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Fdiv { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Fsqrt { /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Fneg { /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Fabs { /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::FloatToInt { /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::IntToFloat { /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::FloatConv { /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::RoundToInt { /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Floor { /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Ceil { /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::Ftrunc { /* HighLevelILInstruction */ src }=> {
            let mut res = Vec::new();
            if let Some(vars) = src.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::FcmpE { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::FcmpNe { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::FcmpLt { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::FcmpLe { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::FcmpGe { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::FcmpGt { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::FcmpO { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }
        HighLevelILOperation::FcmpUo { /* HighLevelILInstruction */ left, /* HighLevelILInstruction */ right }=> {
            let mut res = Vec::new();
            if let Some(vars) = left.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            if let Some(vars) = right.ssa_vars() {
                for var in vars {
                    res.push(var.clone());
                }
            }
            res
        }

        };

        if res.is_empty() { None } else { Some(res) }
    }

    pub fn name(&self) -> String {
        match self {
            HighLevelILOperation::Nop { } => "Nop".to_string(),
            HighLevelILOperation::Block {..} => "Block".to_string(),
            HighLevelILOperation::If {..} => "If".to_string(),
            HighLevelILOperation::While {..} => "While".to_string(),
            HighLevelILOperation::DoWhile {..} => "DoWhile".to_string(),
            HighLevelILOperation::For {..} => "For".to_string(),
            HighLevelILOperation::Switch {..} => "Switch".to_string(),
            HighLevelILOperation::Case {..} => "Case".to_string(),
            HighLevelILOperation::Break {..} => "Break".to_string(),
            HighLevelILOperation::Continue {..} => "Continue".to_string(),
            HighLevelILOperation::Jump {..} => "Jump".to_string(),
            HighLevelILOperation::Ret {..} => "Ret".to_string(),
            HighLevelILOperation::Noret {..} => "Noret".to_string(),
            HighLevelILOperation::Goto {..} => "Goto".to_string(),
            HighLevelILOperation::Label {..} => "Label".to_string(),
            HighLevelILOperation::VarDeclare {..} => "VarDeclare".to_string(),
            HighLevelILOperation::VarInit {..} => "VarInit".to_string(),
            HighLevelILOperation::Assign {..} => "Assign".to_string(),
            HighLevelILOperation::AssignUnpack {..} => "AssignUnpack".to_string(),
            HighLevelILOperation::Var {..} => "Var".to_string(),
            HighLevelILOperation::StructField {..} => "StructField".to_string(),
            HighLevelILOperation::ArrayIndex {..} => "ArrayIndex".to_string(),
            HighLevelILOperation::Split {..} => "Split".to_string(),
            HighLevelILOperation::Deref {..} => "Deref".to_string(),
            HighLevelILOperation::DerefField {..} => "DerefField".to_string(),
            HighLevelILOperation::AddressOf {..} => "AddressOf".to_string(),
            HighLevelILOperation::Const {..} => "Const".to_string(),
            HighLevelILOperation::ConstPtr {..} => "ConstPtr".to_string(),
            HighLevelILOperation::ExternPtr {..} => "ExternPtr".to_string(),
            HighLevelILOperation::FloatConst {..} => "FloatConst".to_string(),
            HighLevelILOperation::Import {..} => "Import".to_string(),
            HighLevelILOperation::Add {..} => "Add".to_string(),
            HighLevelILOperation::Adc {..} => "Adc".to_string(),
            HighLevelILOperation::Sub {..} => "Sub".to_string(),
            HighLevelILOperation::Sbb {..} => "Sbb".to_string(),
            HighLevelILOperation::And {..} => "And".to_string(),
            HighLevelILOperation::Or {..} => "Or".to_string(),
            HighLevelILOperation::Xor {..} => "Xor".to_string(),
            HighLevelILOperation::Lsl {..} => "Lsl".to_string(),
            HighLevelILOperation::Lsr {..} => "Lsr".to_string(),
            HighLevelILOperation::Asr {..} => "Asr".to_string(),
            HighLevelILOperation::Rol {..} => "Rol".to_string(),
            HighLevelILOperation::Rlc {..} => "Rlc".to_string(),
            HighLevelILOperation::Ror {..} => "Ror".to_string(),
            HighLevelILOperation::Rrc {..} => "Rrc".to_string(),
            HighLevelILOperation::Mul {..} => "Mul".to_string(),
            HighLevelILOperation::MuluDp {..} => "MuluDp".to_string(),
            HighLevelILOperation::MulsDp {..} => "MulsDp".to_string(),
            HighLevelILOperation::Divu {..} => "Divu".to_string(),
            HighLevelILOperation::DivuDp {..} => "DivuDp".to_string(),
            HighLevelILOperation::Divs {..} => "Divs".to_string(),
            HighLevelILOperation::DivsDp {..} => "DivsDp".to_string(),
            HighLevelILOperation::Modu {..} => "Modu".to_string(),
            HighLevelILOperation::ModuDp {..} => "ModuDp".to_string(),
            HighLevelILOperation::Mods {..} => "Mods".to_string(),
            HighLevelILOperation::ModsDp {..} => "ModsDp".to_string(),
            HighLevelILOperation::Neg {..} => "Neg".to_string(),
            HighLevelILOperation::Not {..} => "Not".to_string(),
            HighLevelILOperation::SignExtend {..} => "SignExtend".to_string(),
            HighLevelILOperation::ZeroExtend {..} => "ZeroExtend".to_string(),
            HighLevelILOperation::LowPart {..} => "LowPart".to_string(),
            HighLevelILOperation::Call {..} => "Call".to_string(),
            HighLevelILOperation::Equals {..} => "Equals".to_string(),
            HighLevelILOperation::NotEquals {..} => "NotEquals".to_string(),
            HighLevelILOperation::SignedLessThan {..} => "SignedLessThan".to_string(),
            HighLevelILOperation::UnsignedLessThan {..} => "UnsignedLessThan".to_string(),
            HighLevelILOperation::SignedLessThanEquals {..} => "SignedLessThanEquals".to_string(),
            HighLevelILOperation::UnsignedLessThanEquals {..} => "UnsignedLessThanEquals".to_string(),
            HighLevelILOperation::SignedGreaterThanEquals {..} => "SignedGreaterThanEquals".to_string(),
            HighLevelILOperation::UnsignedGreaterThanEquals {..} => "UnsignedGreaterThanEquals".to_string(),
            HighLevelILOperation::SsignedGreaterThan {..} => "SsignedGreaterThan".to_string(),
            HighLevelILOperation::UnsignedGreaterThan {..} => "UnsignedGreaterThan".to_string(),
            HighLevelILOperation::TestBit {..} => "TestBit".to_string(),
            HighLevelILOperation::BoolToInt {..} => "BoolToInt".to_string(),
            HighLevelILOperation::AddOverflow {..} => "AddOverflow".to_string(),
            HighLevelILOperation::Syscall {..} => "Syscall".to_string(),
            HighLevelILOperation::Tailcall {..} => "Tailcall".to_string(),
            HighLevelILOperation::Intrinsic {..} => "Intrinsic".to_string(),
            HighLevelILOperation::Bp {..} => "Bp".to_string(),
            HighLevelILOperation::Trap {..} => "Trap".to_string(),
            HighLevelILOperation::Undef {..} => "Undef".to_string(),
            HighLevelILOperation::Unimpl {..} => "Unimpl".to_string(),
            HighLevelILOperation::UnimplMem {..} => "UnimplMem".to_string(),
            HighLevelILOperation::Fadd {..} => "Fadd".to_string(),
            HighLevelILOperation::Fsub {..} => "Fsub".to_string(),
            HighLevelILOperation::Fmul {..} => "Fmul".to_string(),
            HighLevelILOperation::Fdiv {..} => "Fdiv".to_string(),
            HighLevelILOperation::Fsqrt {..} => "Fsqrt".to_string(),
            HighLevelILOperation::Fneg {..} => "Fneg".to_string(),
            HighLevelILOperation::Fabs {..} => "Fabs".to_string(),
            HighLevelILOperation::FloatToInt {..} => "FloatToInt".to_string(),
            HighLevelILOperation::IntToFloat {..} => "IntToFloat".to_string(),
            HighLevelILOperation::FloatConv {..} => "FloatConv".to_string(),
            HighLevelILOperation::RoundToInt {..} => "RoundToInt".to_string(),
            HighLevelILOperation::Floor {..} => "Floor".to_string(),
            HighLevelILOperation::Ceil {..} => "Ceil".to_string(),
            HighLevelILOperation::Ftrunc {..} => "Ftrunc".to_string(),
            HighLevelILOperation::FcmpE {..} => "FcmpE".to_string(),
            HighLevelILOperation::FcmpNe {..} => "FcmpNe".to_string(),
            HighLevelILOperation::FcmpLt {..} => "FcmpLt".to_string(),
            HighLevelILOperation::FcmpLe {..} => "FcmpLe".to_string(),
            HighLevelILOperation::FcmpGe {..} => "FcmpGe".to_string(),
            HighLevelILOperation::FcmpGt {..} => "FcmpGt".to_string(),
            HighLevelILOperation::FcmpO {..} => "FcmpO".to_string(),
            HighLevelILOperation::FcmpUo {..} => "FcmpUo".to_string(),
            HighLevelILOperation::WhileSsa {..} => "WhileSsa".to_string(),
            HighLevelILOperation::DoWhileSsa {..} => "DoWhileSsa".to_string(),
            HighLevelILOperation::ForSsa {..} => "ForSsa".to_string(),
            HighLevelILOperation::VarInitSsa {..} => "VarInitSsa".to_string(),
            HighLevelILOperation::AssignMemSsa {..} => "AssignMemSsa".to_string(),
            HighLevelILOperation::AssignUnpackMemSsa {..} => "AssignUnpackMemSsa".to_string(),
            HighLevelILOperation::VarSsa {..} => "VarSsa".to_string(),
            HighLevelILOperation::ArrayIndexSsa {..} => "ArrayIndexSsa".to_string(),
            HighLevelILOperation::DerefSsa {..} => "DerefSsa".to_string(),
            HighLevelILOperation::DerefFieldSsa {..} => "DerefFieldSsa".to_string(),
            HighLevelILOperation::CallSsa {..} => "CallSsa".to_string(),
            HighLevelILOperation::SyscallSsa {..} => "SyscallSsa".to_string(),
            HighLevelILOperation::IntrinsicSsa {..} => "IntrinsicSsa".to_string(),
            HighLevelILOperation::VarPhi {..} => "VarPhi".to_string(),
            HighLevelILOperation::MemPhi {..} => "MemPhi".to_string(),
        }
    }
}

