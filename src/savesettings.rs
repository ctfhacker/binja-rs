use std::sync::Arc;

use anyhow::{Context, Result};

use binja_sys::*;

use crate::unsafe_try;
use crate::wrappers::BinjaSaveSettings;

pub struct SaveSettings {
    handle: Arc<BinjaSaveSettings>,
}

impl SaveSettings {
    pub fn new() -> Result<Self> {
        let handle =
            unsafe_try!(BNCreateSaveSettings()).context("Failed to create save setings")?;

        Ok(Self {
            handle: Arc::new(BinjaSaveSettings::new(handle)),
        })
    }

    pub fn handle(&self) -> *mut BNSaveSettings {
        **self.handle
    }
}
