//! Provides miscellaneous types for analysis
use core::*;

use std::sync::Arc;

use crate::binjastr::BinjaStr;
use crate::wrappers::BinjaSymbol;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum SymbolType {
    FunctionSymbol,
    ImportAddressSymbol,
    ImportedFunctionSymbol,
    DataSymbol,
    ImportedDataSymbol,
}

// Send for rayon
unsafe impl Send for SymbolType {}

impl SymbolType {
    pub fn from_bnsymboltype(n: BNSymbolType) -> Option<SymbolType> {
        match n as u64 {
            0 => Some(SymbolType::FunctionSymbol),
            1 => Some(SymbolType::ImportAddressSymbol),
            2 => Some(SymbolType::ImportedFunctionSymbol),
            3 => Some(SymbolType::DataSymbol),
            4 => Some(SymbolType::ImportedDataSymbol),
            _ => panic!("Unknown Symbol type: {:?}\n", n),
        }
    }
}

#[derive(Clone)]
pub struct Symbol {
    handle: Arc<BinjaSymbol>,
}

impl Symbol {
    pub fn new_from_symbol(ptr: *mut BNSymbol) -> Symbol {
        Symbol {
            handle: Arc::new(BinjaSymbol::new(ptr)),
        }
    }

    fn handle(&self) -> *mut BNSymbol {
        **self.handle
    }

    pub fn name(&self) -> BinjaStr {
        unsafe { BinjaStr::new(BNGetSymbolRawName(self.handle())) }
    }

    pub fn short_name(&self) -> BinjaStr {
        unsafe { BinjaStr::new(BNGetSymbolShortName(self.handle())) }
    }

    pub fn long_name(&self) -> BinjaStr {
        unsafe { BinjaStr::new(BNGetSymbolFullName(self.handle())) }
    }

    pub fn address(&self) -> u64 {
        unsafe { BNGetSymbolAddress(self.handle()) }
    }

    pub fn symbol_type(&self) -> SymbolType {
        let symbol_type = unsafe { BNGetSymbolType(self.handle()) };
        SymbolType::from_bnsymboltype(symbol_type)
            .expect(format!("Found unknown SymbolType: {:?}", symbol_type).as_str())
    }
}

impl std::fmt::Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Symbol")
            .field("name", &self.name())
            .field("address", &format!("{:#x}", self.address()))
            .field("short_name", &self.short_name())
            .field("long_name", &self.long_name())
            .finish()
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum VariableSourceType {
    StackVariableSourceType,
    RegisterVariableSourceType,
    FlagVariableSourceType,
    Unknown,
}

impl VariableSourceType {
    pub fn from_bn(n: BNVariableSourceType) -> Option<VariableSourceType> {
        match n as u64 {
            0 => Some(VariableSourceType::StackVariableSourceType),
            1 => Some(VariableSourceType::RegisterVariableSourceType),
            2 => Some(VariableSourceType::FlagVariableSourceType),
            _ => Some(VariableSourceType::Unknown),
        }
    }
}
