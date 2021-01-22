//! Provides `Symbol` 
use core::*;

use std::sync::Arc;

use crate::wrappers::BinjaSymbol;
use crate::binjastr::BinjaStr;

/// Symbol of an address from binary ninja
#[derive(Clone)]
pub struct Symbol {
    handle: Arc<BinjaSymbol>
}

impl Symbol {
    pub fn new(handle: *mut BNSymbol) -> Self {
        Symbol { handle: Arc::new(BinjaSymbol::new(handle)) }
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
