use std::sync::Arc;

use anyhow::{Result, Context};

use core::*;
use crate::unsafe_try;
use wrappers::BinjaSaveSettings;

pub struct SaveSettings {
    handle: Arc<BinjaSaveSettings>
}

impl SaveSettings {
    pub fn new() -> Result<Self> {
        let handle = unsafe_try!(BNCreateSaveSettings())
            .context("Failed to create save setings")?;

        Ok(Self {
            handle: Arc::new(BinjaSaveSettings::new(handle))
        })
    }

    pub fn handle(&self) -> *mut BNSaveSettings {
        **self.handle
    }
}
