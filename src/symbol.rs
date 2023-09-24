//! Provides `Symbol`
use core::*;

use std::sync::Arc;

use crate::binjastr::BinjaStr;
use crate::wrappers::BinjaSymbol;

/// Symbol of an address from binary ninja
#[derive(Clone)]
pub struct Symbol {
    handle: Arc<BinjaSymbol>,
}

impl Symbol {
    pub fn new(handle: *mut BNSymbol) -> Self {
        Symbol {
            handle: Arc::new(BinjaSymbol::new(handle)),
        }
    }

    fn handle(&self) -> *mut BNSymbol {
        **self.handle
    }

    /// Name of the symbol as provided by the core
    pub fn name(&self) -> BinjaStr {
        let binjastr = unsafe { BNGetSymbolRawName(self.handle()) };
        BinjaStr::new(binjastr)
    }

    /// Short name of the symbol as provided by the core
    pub fn short_name(&self) -> BinjaStr {
        let binjastr = unsafe { BNGetSymbolShortName(self.handle()) };
        BinjaStr::new(binjastr)
    }

    /// Full name of the symbol as provided by the core
    pub fn full_name(&self) -> BinjaStr {
        let binjastr = unsafe { BNGetSymbolFullName(self.handle()) };
        BinjaStr::new(binjastr)
    }

    /// Address of the symbol as provided by the core
    pub fn address(&self) -> u64 {
        unsafe { BNGetSymbolAddress(self.handle()) }
    }

    /// Ordinal of the symbol as provided by the core
    pub fn ordinal(&self) -> u64 {
        unsafe { BNGetSymbolOrdinal(self.handle()) }
    }

    pub fn symbol_type(&self) -> SymbolType {
        let symbol_type = unsafe { BNGetSymbolType(self.handle()) };
        SymbolType::from_bnsymboltype(symbol_type)
            .expect(format!("Found unknown SymbolType: {:?}", symbol_type).as_str())
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "<symbol: {}>", self.name())
    }
}

impl std::fmt::Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("Symbol")
            .field("address", &format!("{:#x}", self.address()))
            .field("name", &self.name())
            .field("short_name", &self.short_name())
            .field("full_name", &self.full_name())
            .field("ordinal", &self.ordinal())
            .finish()
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum SymbolType {
    FunctionSymbol,
    ImportAddressSymbol,
    ImportedFunctionSymbol,
    DataSymbol,
    ImportedDataSymbol,
    ExternalSymbol,
    LibraryFunctionSymbol,
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
            5 => Some(SymbolType::ExternalSymbol),
            6 => Some(SymbolType::LibraryFunctionSymbol),
            _ => panic!("Unknown Symbol type: {:?}\n", n),
        }
    }
}
