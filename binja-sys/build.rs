#[cfg(feature = "build_time_bindings")]
// extern crate bindgen;
#[cfg(feature = "build_time_bindings")]
use std::path::Path;

#[cfg(feature = "build_time_bindings")]
use std::path::PathBuf;

#[cfg(feature = "build_time_bindings")]
use std::process::Command;

#[cfg(target_os = "macos")]
pub fn link_path() -> String {
    String::from("/Applications/Binary Ninja.app/Contents/MacOS")
}

#[cfg(target_os = "linux")]
pub fn link_path() -> String {
    let home = std::env::var("HOME").expect("No HOME key in env");
    std::path::Path::new(&home)
        .join("binaryninja")
        .to_string_lossy()
        .into_owned()
}

#[cfg(target_os = "windows")]
pub fn link_path() -> String {
    let home = std::env::var("ProgramFiles").expect("ProgramFiles");
    std::path::Path::new(&home)
        .join("Vector35")
        .join("BinaryNinja")
        .to_string_lossy()
        .into_owned()
}

#[cfg(feature = "build_time_bindings")]
fn build() {
    if !Path::new("binaryninja-api/.git").exists() {
        let _ = Command::new("git")
            .args(&["submodule", "update", "--init"])
            .status();
    }

    let dst = PathBuf::from("src");

    let bindings = bindgen::Builder::default()
        .header("wrapper.hpp")
        .generate()
        .expect("Unable to generate bindings");

    let mut headings = String::from(
        r#"#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(improper_ctypes)]
"#,
    );

    headings.push_str(&bindings.to_string());

    // headings = headings.replace("::std::os::raw::c_longlong", "REPLACEMEBACK");
    // headings = headings.replace("::std::os::raw::c_long", "u64");
    // headings = headings.replace( "REPLACEMEBACK", "::std::os::raw::c_longlong");

    std::fs::write(dst.join("lib.rs"), headings).expect("Failed to write lib.rs bindings");
}

fn main() {
    #[cfg(feature = "build_time_bindings")]
    build();

    println!("cargo:rustc-link-search={}", link_path());
    println!("cargo:rustc-link-lib=binaryninjacore");
}
