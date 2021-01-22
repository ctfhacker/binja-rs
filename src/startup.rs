//! Provides functions dealing with initialization of the Binary Ninja Core
use core::*;

use binjastr::BinjaStr;

static mut _INIT_PLUGINS: bool = false;

/// Returns a string containing the current plugin path inside the install path
pub fn get_bundled_plugin_path() -> BinjaStr {
    unsafe { 
        BinjaStr::from(BNGetBundledPluginDirectory())
    }
}

#[cfg(target_os = "macos")]
pub fn get_os_plugin_path() -> String {
    String::from("/Applications/Binary Ninja.app/Contents/MacOS/plugins")
}

#[cfg(target_os = "linux")]
pub fn get_os_plugin_path() -> String {
    let home = std::env::var("HOME").expect("No HOME key in env");
    std::path::Path::new(&home).join("binaryninja").join("plugins")
        .to_string_lossy().into_owned()
}

#[cfg(target_os = "windows")]
pub fn get_os_plugin_path() -> String {
    let home = std::env::var("ProgramFiles").expect("ProgramFiles");
    std::path::Path::new(&home).join("Vector35").join("BinaryNinja").join("plugins")
        .to_string_lossy().into_owned()
}

/// Initialize plugins necessary to begin analysis
pub fn init_plugins() {
    unsafe {
        if _INIT_PLUGINS { 
            return; 
        }

        _INIT_PLUGINS = true;

        let plugin_dir = get_os_plugin_path().as_ptr() as usize as *const i8;
        print!("Using plugin dir: {}\n", get_os_plugin_path());

        BNSetBundledPluginDirectory(plugin_dir);

        print!("BNInitCorePlugins: {}\n", BNInitCorePlugins());
        BNInitUserPlugins();
        BNInitRepoPlugins();
        // print!("BNInitPlugins: {}\n", BNInitPlugins(true));

        if !BNIsLicenseValidated() {
            panic!("License is not valid! Please supply a valid license.");
        }
    }

    trace!("Plugins initialized");
}
