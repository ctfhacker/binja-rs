// #[cfg(feature="build_time_bindings")]
// extern crate bindgen;

use std::path::PathBuf;

#[cfg(feature="build_time_bindings")]
use std::path::Path;

#[cfg(feature="build_time_bindings")]
use std::process::Command;

#[cfg(target_os = "macos")]
fn link_path() -> PathBuf {
    PathBuf::from("/applications/binary ninja.app/contents/macos")
}

#[cfg(target_os = "linux")]
fn link_path() -> PathBuf {
    let binja_path = PathBuf::from(std::env::var("HOME").unwrap()).join("binaryninja");
    let lib = binja_path.join("libbinaryninjacore.so");

    if !lib.exists() {
        if !binja_path.join("libbinaryninjacore.so.1").exists() {
            panic!("Cannot find libbinaryninjacore.so.1 in {}", binja_path.to_str().unwrap());
        }

        use std::os::unix::fs;

        fs::symlink(binja_path.join("libbinaryninjacore.so.1"), lib)
            .expect("failed to create required symlink");
    }

    binja_path
}

#[cfg(target_os = "windows")]
fn link_path() -> PathBuf {
    let localappdata = std::env::var("LOCALAPPDATA").expect("No LOCALAPPDATA in env");
    PathBuf::from(localappdata).join("Vector35").join("BinaryNinja")
}

#[cfg(feature="build_time_bindings")]
fn build() {
    if !Path::new("binaryninja-api/.git").exists() {
        let _ = Command::new("git").args(&["submodule", "update", "--init"])
                                   .status();
    }

    let dst = PathBuf::from("src");

    let bindings = bindgen::Builder::default()
        .header("wrapper.hpp")
        .generate()
        .expect("Unable to generate bindings");

    let mut headings = String::from(r#"#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(improper_ctypes)]
"#);

    headings.push_str(&bindings.to_string());

    headings = headings.replace("::std::os::raw::c_longlong", "REPLACEMEBACK");
    headings = headings.replace("::std::os::raw::c_long", "u64");
    headings = headings.replace( "REPLACEMEBACK", "::std::os::raw::c_longlong");

    std::fs::write(dst.join("lib.rs"), headings);
}

fn main() {
    #[cfg(feature="build_time_bindings")]
    build();

    println!("cargo:rustc-link-search={}", link_path().to_str().unwrap());
    println!("cargo:rustc-link-lib=binaryninjacore");
}
