//! Provides BasicBlock and analysis functions for Basic Blocks
use core::*;

use anyhow::Result;

use std::convert::TryInto;
use std::sync::Arc;

use crate::function::Function;
use crate::lowlevelil::LowLevelILBasicBlock;
use crate::mediumlevelil::MediumLevelILBasicBlock;
use crate::traits::*;
use crate::unsafe_try;
use crate::wrappers::BinjaBasicBlock;

#[derive(Clone)]
pub struct BasicBlock {
    handle: Arc<BinjaBasicBlock>,
    pub func: Function,
}

impl BasicBlock {
    pub fn new(handle: *mut BNBasicBlock, func: Function) -> BasicBlock {
        BasicBlock {
            handle: Arc::new(BinjaBasicBlock::new(handle)),
            func,
        }
    }

    pub fn from_arc(handle: Arc<BinjaBasicBlock>, func: Function) -> BasicBlock {
        BasicBlock { handle, func }
    }

    pub fn handle(&self) -> *mut BNBasicBlock {
        **self.handle
    }

    pub fn low_level_il(&self) -> LowLevelILBasicBlock {
        unimplemented!()
    }

    /// Create a MLIL basic block
    pub fn medium_level_il(&self) -> Result<MediumLevelILBasicBlock> {
        let mlil_func = self.func.medium_level_il()?;
        MediumLevelILBasicBlock::new(self.handle(), mlil_func.clone())
    }

    /// Get the MLIL form of this basic block (Alias for self.medium_level_il)
    pub fn mlil(&self) -> Result<MediumLevelILBasicBlock> {
        self.medium_level_il()
    }

    /// Get the start index of a basic block
    pub fn start(&self) -> u64 {
        unsafe { BNGetBasicBlockStart(self.handle()) }
    }

    /// Get the end index of a basic block
    pub fn end(&self) -> u64 {
        unsafe { BNGetBasicBlockEnd(self.handle()) }
    }

    /// Get the length of the basic block
    pub fn len(&self) -> u64 {
        self.end() - self.start()
    }

    pub fn print_mlil(&self) {
        if let Ok(mlilbb) = self.mlil() {
            for instr in mlilbb.il() {
                print!("{:#x}: {}\n", instr.address, instr);
            }
        }
    }

    pub fn print_mlilssa(&self) {
        if let Ok(mlilbb) = self.mlil() {
            for instr in mlilbb.il() {
                print!(
                    "{:#x}: {}\n",
                    instr.ssa_form().unwrap().address,
                    instr.ssa_form().unwrap()
                );
            }
        }
    }

    /// Get the total number of edges for this basic block
    pub fn total_edges(&self) -> u64 {
        let mut res = 0;

        unsafe {
            let mut count = 0;

            // Get the outgoing edges from the core
            let _ = BNGetBasicBlockOutgoingEdges(self.handle(), &mut count);

            // Add the outgoing edges to the count
            res += count;

            // Get the outgoing edges from the core
            let _ = BNGetBasicBlockIncomingEdges(self.handle(), &mut count);

            // Add the incoming edges to the count
            res += count;
        }

        res.try_into().unwrap()
    }

    /// Get the list of outgoing edges from this basic block
    pub fn outgoing_edges(&self) -> Vec<BasicBlockEdge> {
        let mut count = 0;

        let mut res = Vec::new();

        unsafe {
            // Get the outgoing edges from the core
            let edges = BNGetBasicBlockOutgoingEdges(self.handle(), &mut count);

            // Get a workable slice to the edges
            let edges_slice = std::slice::from_raw_parts(edges, count as usize);

            // Get the functions from the starting addresses passed back from binja core
            for edge in edges_slice {
                res.push(BasicBlockEdge::new(*edge, self.func.clone()));
            }

            // Free the list from core
            BNFreeBasicBlockEdgeList(edges, count);
        }

        res
    }

    /// Get the list of incoming edges from this basic block
    pub fn incoming_edges(&self) -> Vec<BasicBlockEdge> {
        let mut count = 0;

        let mut res = Vec::new();

        unsafe {
            // Get the outgoing edges from the core
            let edges = BNGetBasicBlockIncomingEdges(self.handle(), &mut count);

            // Get a workable slice to the edges
            let edges_slice = std::slice::from_raw_parts(edges, count as usize);

            // Get the functions from the starting addresses passed back from binja core
            for edge in edges_slice {
                res.push(BasicBlockEdge::new(*edge, self.func.clone()));
            }

            // Free the list from core
            BNFreeBasicBlockEdgeList(edges, count);
        }

        res
    }

    /// List of dominators for this basic block
    pub fn dominators(&self) -> Vec<BasicBlock> {
        let mut count = 0;

        let mut res = Vec::new();

        unsafe {
            let blocks = BNGetBasicBlockDominators(self.handle(), &mut count, false);
            let blocks_slice = std::slice::from_raw_parts(blocks, count as usize);
            for block in blocks_slice {
                res.push(BasicBlock::new(*block, self.func.clone()));
            }
        }

        res
    }

    /// List of post dominators for this basic block
    pub fn post_dominators(&self) -> Vec<BasicBlock> {
        let mut count = 0;

        let mut res = Vec::new();

        unsafe {
            let blocks = BNGetBasicBlockDominators(self.handle(), &mut count, true);
            let blocks_slice = std::slice::from_raw_parts(blocks, count as usize);

            for block in blocks_slice {
                res.push(BasicBlock::new(*block, self.func.clone()));
            }
        }

        res
    }

    /// List of strict dominators for this basic block
    pub fn strict_dominators(&self) -> Vec<BasicBlock> {
        let mut count = 0;

        let mut res = Vec::new();

        unsafe {
            let blocks = BNGetBasicBlockStrictDominators(self.handle(), &mut count, false);
            let blocks_slice = std::slice::from_raw_parts(blocks, count as usize);
            for block in blocks_slice {
                res.push(BasicBlock::new(*block, self.func.clone()));
            }
        }

        res
    }

    /// Get the immediate dominator for this basic block
    pub fn immediate_dominator(&self) -> Result<BasicBlock> {
        let handle = unsafe_try!(BNGetBasicBlockImmediateDominator(self.handle(), false))?;
        Ok(BasicBlock::new(handle, self.func.clone()))
    }

    /// Get the immediate post dominator for this basic block
    pub fn immediate_post_dominator(&self) -> Result<BasicBlock> {
        let handle = unsafe_try!(BNGetBasicBlockImmediateDominator(self.handle(), true))?;
        Ok(BasicBlock::new(handle, self.func.clone()))
    }

    /// List of child blocks in the dominator tree for this basic block
    pub fn dominator_tree_children(&self) -> Vec<BasicBlock> {
        let mut count = 0;

        let mut res = Vec::new();

        unsafe {
            let blocks = BNGetBasicBlockDominatorTreeChildren(self.handle(), &mut count, false);
            let blocks_slice = std::slice::from_raw_parts(blocks, count as usize);
            for block in blocks_slice {
                res.push(BasicBlock::new(*block, self.func.clone()));
            }
        }

        res
    }

    /// List of child blocks in the post dominator tree for this basic block
    pub fn post_dominator_tree_children(&self) -> Vec<BasicBlock> {
        let mut count = 0;

        let mut res = Vec::new();

        unsafe {
            let blocks = BNGetBasicBlockDominatorTreeChildren(self.handle(), &mut count, true);
            let blocks_slice = std::slice::from_raw_parts(blocks, count as usize);
            for block in blocks_slice {
                res.push(BasicBlock::new(*block, self.func.clone()));
            }
        }

        res
    }

    /// List of child blocks in the dominator tree for this basic block
    pub fn dominance_frontier(&self) -> Vec<BasicBlock> {
        let mut count = 0;

        let mut res = Vec::new();

        unsafe {
            let blocks = BNGetBasicBlockDominanceFrontier(self.handle(), &mut count, false);
            let blocks_slice = std::slice::from_raw_parts(blocks, count as usize);
            for block in blocks_slice {
                res.push(BasicBlock::new(*block, self.func.clone()));
            }
        }

        res
    }

    /// List of child blocks in the post dominator tree for this basic block
    pub fn post_dominance_frontier(&self) -> Vec<BasicBlock> {
        let mut count = 0;

        let mut res = Vec::new();

        unsafe {
            let blocks = BNGetBasicBlockDominanceFrontier(self.handle(), &mut count, true);
            let blocks_slice = std::slice::from_raw_parts(blocks, count as usize);
            for block in blocks_slice {
                res.push(BasicBlock::new(*block, self.func.clone()));
            }
        }

        res
    }
}

impl std::cmp::PartialEq for BasicBlock {
    fn eq(&self, other: &Self) -> bool {
        (self.start(), self.end(), self.func.start())
            == (other.start(), other.end(), other.func.start())
    }
}

impl std::fmt::Debug for BasicBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "<BasicBlock func: {:?}, start: {:#x}, len: {}>",
            self.func.name(),
            self.start(),
            self.len()
        )
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u32)]
enum BasicBlockEdgeType {
    UnconditionalBranch = 0,
    FalseBranch = 1,
    TrueBranch = 2,
    CallDestination = 3,
    FunctionReturn = 4,
    SystemCall = 5,
    IndirectBranch = 6,
    ExceptionBranch = 7,
    UnresolvedBranch = 127,
    UserDefinedBranch = 128,
}

impl BasicBlockEdgeType {
    pub fn from_u32(val: u32) -> Self {
        match val {
            0 => BasicBlockEdgeType::UnconditionalBranch,
            1 => BasicBlockEdgeType::FalseBranch,
            2 => BasicBlockEdgeType::TrueBranch,
            3 => BasicBlockEdgeType::CallDestination,
            4 => BasicBlockEdgeType::FunctionReturn,
            5 => BasicBlockEdgeType::SystemCall,
            6 => BasicBlockEdgeType::IndirectBranch,
            7 => BasicBlockEdgeType::ExceptionBranch,
            127 => BasicBlockEdgeType::UnresolvedBranch,
            128 => BasicBlockEdgeType::UserDefinedBranch,
            _ => panic!("Unknown BasicBlockEdge: {}\n", val),
        }
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct BasicBlockEdge {
    type_: BasicBlockEdgeType,
    target: BasicBlock,
    back_edge: bool,
    fall_through: bool,
}

impl BasicBlockEdge {
    pub fn new(edge: BNBasicBlockEdge, func: Function) -> Self {
        BasicBlockEdge {
            type_: BasicBlockEdgeType::from_u32(edge.type_.try_into().unwrap()),
            target: BasicBlock::new(edge.target, func),
            back_edge: edge.backEdge,
            fall_through: edge.fallThrough,
        }
    }
}
