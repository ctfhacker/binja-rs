//! Provides various traits for abstracting over the various ILs Binary Ninja provides
use core::*;
use std::convert::TryInto;

use anyhow::Result;

use crate::basicblock::BasicBlock;
use crate::function::Function;

pub trait FunctionTrait {
    type Ins;
    type Block: BasicBlockTrait<Ins = Self::Ins>;
    type Func;

    fn instruction(&self, i: usize) -> Result<Self::Ins>;
    fn blocks(&self) -> Vec<Self::Block>;

    fn ssa_form(&self) -> Result<Self::Func>;
    fn ssa(&self) -> Result<Self::Func> {
        self.ssa_form()
    }

    fn text(&self, i: usize) -> Result<String>;
    fn expr_text(&self, expr_index: usize) -> Result<String>;

    /*
    /// Retrieve all of the IL for the current function
    fn il(&self) -> Vec<Self::Ins> {
        self.blocks().iter().fold(Vec::new(), |mut v, bb| {
            v.extend(bb.il()); v
        })
    }
    */
}

pub trait BasicBlockTrait {
    type Ins;
    type Func: FunctionTrait<Ins = Self::Ins>;

    /// Getter for handle in order to default implement start() and end()
    fn handle(&self) -> *mut BNBasicBlock;

    /// Getter for func in order to default implement il()
    fn func(&self) -> Option<&Self::Func>;

    fn raw_function(&self) -> Function;

    /// Get the start index of a basic block
    fn start(&self) -> usize {
        let res = unsafe { BNGetBasicBlockStart(self.handle()) };
        trace!("BB Start: {}\n", res);
        res.try_into().unwrap()
    }

    /// Get the end index of a basic block
    fn end(&self) -> usize {
        let res = unsafe { BNGetBasicBlockEnd(self.handle()) };
        trace!("BB End: {}\n", res);
        res.try_into().unwrap()
    }

    /// Get the length of the basic block
    fn len(&self) -> usize {
        self.end() - self.start()
    }

    /// Default implementation on getting the IL for a particular basic block
    fn il(&self) -> Vec<Self::Ins> {
        let result = (self.start()..self.end())
            .filter_map(|i| self.func().unwrap().instruction(i).ok())
            .collect();
        result
    }

    /// List of dominators for this basic block
    fn dominators(&self) -> Vec<BasicBlock> {
        let mut count = 0;

        let mut res = Vec::new();

        unsafe {
            let blocks = BNGetBasicBlockDominators(self.handle(), &mut count, false);
            let blocks_slice = std::slice::from_raw_parts(blocks, count as usize);
            for block in blocks_slice {
                res.push(BasicBlock::new(*block, self.raw_function().clone()));
            }
        }

        res
    }

    /// List of post dominators for this basic block
    fn post_dominators(&self) -> Vec<BasicBlock> {
        let mut count = 0;

        let mut res = Vec::new();

        unsafe {
            let blocks = BNGetBasicBlockDominators(self.handle(), &mut count, true);
            let blocks_slice = std::slice::from_raw_parts(blocks, count as usize);

            for block in blocks_slice {
                res.push(BasicBlock::new(*block, self.raw_function().clone()));
            }
        }

        res
    }
}

/*
pub trait InstrumentTrait {
    type Params;
    fn params(&self) -> Option<Self::Params>;

    fn expr_list<T: InstrumentTrait>(&self, op_index: usize) -> Vec<T>;
    fn expr<T: InstrumentTrait>(&self) -> T;
}
*/
