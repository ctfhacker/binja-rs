extern crate anyhow;
extern crate binja_rs;
extern crate clap;
extern crate rayon;

use anyhow::Result;
use clap::Parser;

use std::time::Instant;

use binja_rs::highlevelil::HighLevelILOperation;
use binja_rs::lowlevelil::LowLevelILOperation;
use binja_rs::mediumlevelil::MediumLevelILOperation;
use binja_rs::*;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    input: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let now = Instant::now();
    let bv = binaryview::BinaryView::new_from_filename(&args.input).unwrap();
    let done = now.elapsed();
    println!("Analysis took: {}.{}", done.as_secs(), done.subsec_nanos());

    let now = Instant::now();

    let llil_ins = bv.llil_instructions();
    let call_count = llil_ins
        .iter()
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
    println!("MLIL Instrs: {}", mlil_ins.len());
    let call_count = mlil_ins
        .iter()
        .filter(|ref i| matches!(*i.operation, MediumLevelILOperation::Call { .. }))
        .collect::<Vec<_>>();

    let done = now.elapsed();
    print!("[{:?}] MLIL_CALL: {:?}\n", done, call_count.len());
    for instr in call_count.iter().take(10) {
        print!("[{:#x}] {}\n", instr.address, instr);
    }
    print!("\n");

    let hlil_ins = bv.hlil_instructions();
    let call_count = hlil_ins
        .iter()
        .filter(|ref i| matches!(*i.operation, HighLevelILOperation::Call { .. }))
        .collect::<Vec<_>>();

    let done = now.elapsed();
    print!("[{:?}] HLIL_CALL: {:?}\n", done, call_count.len());
    for instr in call_count.iter().take(10) {
        print!("[{:#x}] {}\n", instr.address, instr);
    }

    Ok(())
}
