//! Provides `InstructionTextToken`
use binja_sys::*;

use std::ffi::CStr;
use std::fmt;

pub struct InstructionTextToken {
    pub type_: BNInstructionTextTokenType,
    pub text: String,
    pub value: u64,
    pub size: usize,
    pub operand: usize,
    pub context: BNInstructionTextTokenContext,
    pub confidence: u8,
    pub address: u64,
}

impl InstructionTextToken {
    pub fn new_from_token(token: BNInstructionTextToken) -> InstructionTextToken {
        InstructionTextToken {
            type_: token.type_,
            text: unsafe {
                CStr::from_ptr(token.text)
                    .to_string_lossy()
                    .into_owned()
                    .into()
            },
            value: token.value,
            size: token.size,
            operand: token.operand,
            context: token.context,
            confidence: token.confidence,
            address: token.address,
        }
    }
}

impl fmt::Debug for InstructionTextToken {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.text)
    }
}
