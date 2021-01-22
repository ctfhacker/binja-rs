#[macro_use]
extern crate clap;
extern crate binja_rs;

use binja_rs::*;
use clap::{App, Arg};

fn main() {
    let matches = App::new("Binja bininfo")
                    .version("0.1")
                    .author("@ctfhacker")
                    .about("Prints binary information using Binary Ninja")
                    .arg(Arg::with_name("INPUT")
                        .help("Binary file to analyze")
                        .required(true)
                        .index(1))
                    .get_matches();

    let filename = matches.value_of("INPUT").unwrap();
    let bv = binaryview::BinaryView::new_from_filename(filename).expect("BinaryView failed");

    println!("Binary View: {}", bv);
    println!("Target: {}", filename);
    println!("TYPE: {}", bv.type_name());
    println!("START: {:#x}", bv.start());
    println!("ENTRY: {:#x}", bv.entry_point());

    println!("Function Count: {}", bv.functions().len());

    println!("---------- 10 Functions with xrefs ----------");
    for func in bv.functions().iter().take(10) {
        println!("{:#x}: {}", func.start(), func.name().unwrap());

        let refs: Vec<u64> = bv.get_code_refs(func.start()).iter().map(|x| x.addr).collect();
        println!("xrefs: {:x?}", refs);
    }

    println!("---------- 10 Strings ----------");
    println!("Address [len]: Data");
    for s in bv.strings().iter().take(10) {
        println!("{:#x} [{:2}]: {}", s.start(), s.len(), s);
    }
}
