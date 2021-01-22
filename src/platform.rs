//! Provides bare minimum `Platform` support
use core::*;

use std::sync::Arc;
use binaryview::BinaryView;
use wrappers::BinjaPlatform;

#[derive(Clone)]
pub struct Platform {
    pub handle: Arc<BinjaPlatform>
}

impl Platform {
    pub fn new(bv: &BinaryView) -> Option<Platform> {
        /*
        unsafe {
            let plat = BNGetDefaultPlatform(bv);
            if plat.is_null() {
                None
            } else {
                Some( Platform{ handle: plat } )
            }
        }
        */
        let handle = unsafe {
            BNGetDefaultPlatform(bv.handle()).as_mut().expect("BNGetDefaultPlatform failed")
        };

        Some( Platform{ handle: Arc::new(BinjaPlatform::new(handle)) } )
    }

    pub fn handle(&self) -> *mut BNPlatform {
        **self.handle
    }
}
