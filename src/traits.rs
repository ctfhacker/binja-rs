//! Provides various traits for abstracting over the various ILs Binary Ninja provides
use core::*;

use anyhow::Result;

pub trait FunctionTrait {
    type Ins;
    type Block: BasicBlockTrait<Ins=Self::Ins>;
    type Func;

    fn instruction(&self, i: u64) -> Result<Self::Ins>;
    fn blocks(&self) -> Vec<Self::Block>;

    fn ssa_form(&self) -> Result<Self::Func>;
    fn ssa(&self) -> Result<Self::Func> { 
        self.ssa_form() 
    }

    fn text(&self, i: u64) -> Result<String>;
    // fn expr_text(&self, expr_index: usize) -> Vec<InstructionTextToken>;

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
    type Func: FunctionTrait<Ins=Self::Ins>;

    /// Getter for handle in order to default implement start() and end()
    fn handle(&self) -> *mut BNBasicBlock;

    /// Getter for func in order to default implement il()
    fn func(&self) -> Option<&Self::Func>;

    /// Get the start index of a basic block
    fn start(&self) -> u64 {
        let res = unsafe { BNGetBasicBlockStart(self.handle()) };
        trace!("BB Start: {}\n", res);
        res
    }

    /// Get the end index of a basic block
    fn end(&self) -> u64 {
        let res = unsafe { BNGetBasicBlockEnd(self.handle()) };
        trace!("BB End: {}\n", res);
        res
    }

    /// Get the length of the basic block
    fn len(&self) -> u64 {
        self.end() - self.start()
    }

    /// Default implementation on getting the IL for a particular basic block
    fn il(&self) -> Vec<Self::Ins> {
        let result = (self.start()..self.end()).filter_map(|i| self.func().unwrap().instruction(i as u64).ok())
                                               .collect();
        result
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
