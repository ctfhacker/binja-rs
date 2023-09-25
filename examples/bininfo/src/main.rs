use binja_rs::*;

use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    input: String,
}

fn main() {
    let args = Args::parse();

    let bv = binaryview::BinaryView::new_from_filename(&args.input)
        .build()
        .unwrap();

    println!("Binary View: {}", bv);
    println!("Target: {}", args.input);
    println!("TYPE: {}", bv.type_name());
    println!("START: {:#x}", bv.start());
    println!("ENTRY: {:#x}", bv.entry_point());

    println!("Function Count: {}", bv.functions().len());

    println!("---------- 10 Functions with xrefs ----------");
    for func in bv.functions().iter().take(10) {
        println!("{:#x}: {}", func.start(), func.name().unwrap());

        let refs: Vec<u64> = bv
            .get_code_refs(func.start())
            .iter()
            .map(|x| x.addr)
            .collect();
        println!("xrefs: {:x?}", refs);
    }

    println!("---------- 10 Strings ----------");
    println!("Address [len]: Data");
    for s in bv.strings().iter().take(10) {
        println!("{:#x} [{:2}]: {}", s.start(), s.len(), s);
    }
}
