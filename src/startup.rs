//! Provides functions dealing with initialization of the Binary Ninja Core
use core::*;
use std::path::PathBuf;

use crate::binjastr::BinjaStr;

static mut _INIT_PLUGINS: bool = false;

/// Returns a string containing the current plugin path inside the install path
pub fn get_bundled_plugin_path() -> BinjaStr {
    unsafe { BinjaStr::from(BNGetBundledPluginDirectory()) }
}

#[cfg(target_os = "macos")]
pub fn get_os_plugin_path() -> PathBuf {
    PathBuf::from("/Applications/Binary Ninja.app/Contents/MacOS/plugins")
}

#[cfg(target_os = "linux")]
pub fn get_os_plugin_path() -> PathBuf {
    let home = std::env::var("HOME").expect("No HOME key in env");
    std::path::PathBuf::from(&home)
        .join("binaryninja")
        .join("plugins")
}

#[cfg(target_os = "windows")]
pub fn get_os_plugin_path() -> PathBuf {
    let home = std::env::var("ProgramFiles").expect("ProgramFiles");
    std::path::PathBuf::from(&home)
        .join("Vector35")
        .join("BinaryNinja")
        .join("plugins")
}

/// Initialize plugins necessary to begin analysis
pub fn init_plugins() {
    unsafe {
        if _INIT_PLUGINS {
            return;
        }

        // Init the env logger as well
        env_logger::init();

        let plugin_dir = get_os_plugin_path();
        print!("Using plugin dir: {:?}\n", get_os_plugin_path());

        BNSetBundledPluginDirectory(
            plugin_dir.as_os_str().to_str().unwrap().as_ptr() as usize as *const i8
        );

        BNInitPlugins(
            !std::env::var("BN_DISABLE_USER_PLUGINS")
                .unwrap_or_else(|_| "true".to_string())
                .parse::<bool>()
                .unwrap(),
        );
        BNInitRepoPlugins();

        // print!("BNInitCorePlugins: {}\n", BNInitCorePlugins());
        // BNInitUserPlugins();
        // BNInitRepoPlugins();
        // print!("BNInitPlugins: {}\n", BNInitPlugins(true));

        if !BNIsLicenseValidated() {
            panic!("License is not valid! Please supply a valid license.");
        }

        _INIT_PLUGINS = true;
    }

    trace!("Plugins initialized");
}
