extern crate clap;
extern crate binja_rs;
extern crate rayon;
extern crate anyhow;

use clap::{App, Arg};
use anyhow::Result;

use std::time::Instant;

use binja_rs::*;
use binja_rs::lowlevelil::LowLevelILOperation;
use binja_rs::mediumlevelil::MediumLevelILOperation;
use binja_rs::highlevelil::HighLevelILOperation;

fn main() -> Result<()> {
    let matches = App::new("Binja count-ins")
                    .version("0.1")
                    .author("@ctfhacker")
                    .about("Count the number of two types of instructions NOT in parallel")
                    .arg(Arg::with_name("INPUT")
                        .help("Binary file to analyze")
                        .required(true)
                        .index(1))
                    .get_matches();

    let filename = matches.value_of("INPUT").unwrap();
    let now = Instant::now();
    let bv = binaryview::BinaryView::new_from_filename(filename).unwrap();
    let done = now.elapsed();
    println!("Analysis took: {}.{}", done.as_secs(), done.subsec_nanos());

    let now = Instant::now();

    let llil_ins = bv.llil_instructions();
    let call_count = llil_ins.iter()
                       .filter(|ref i| matches!(*i.operation, LowLevelILOperation::Call { .. }))
                       .collect::<Vec<_>>();
    
    let done = now.elapsed();
    print!("[{:?}] LLIL_CALL: {:?}\n", done, call_count.len());
    for instr in call_count.iter().take(10) {
        print!("[{:#x}] {}\n", instr.address, instr);
    }
    print!("\n");

    let now = Instant::now();

    let mlil_ins = bv.mlil_instructions();
    let call_count = mlil_ins.iter()
                       .filter(|ref i| matches!(*i.operation, MediumLevelILOperation::Call { .. }))
                       .collect::<Vec<_>>();
    
    let done = now.elapsed();
    print!("[{:?}] MLIL_CALL: {:?}\n", done, call_count.len());
    for instr in call_count.iter().take(10) {
        print!("[{:#x}] {}\n", instr.address, instr);
    }
    print!("\n");

    let hlil_ins = bv.hlil_instructions();
    let call_count = hlil_ins.iter()
                       .filter(|ref i| matches!(*i.operation, HighLevelILOperation::Call { .. }))
                       .collect::<Vec<_>>();
    
    let done = now.elapsed();
    print!("[{:?}] HLIL_CALL: {:?}\n", done, call_count.len());
    for instr in call_count.iter().take(10) {
        print!("[{:#x}] {}\n", instr.address, instr);
    }

    Ok(())
}
