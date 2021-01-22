#[macro_use]
extern crate clap;
extern crate binja_rs;
extern crate anyhow;
extern crate rayon;

use anyhow::Result;
use rayon::prelude::*;
use clap::{App, Arg};

use binja_rs::binaryview::BinaryView;
use binja_rs::traits::*;

fn main() -> Result<()> {
    let matches = App::new("Binja hlil_parse")
                    .version("0.1")
                    .author("@ctfhacker")
                    .about("Example using some of the HLIL functionality")
                    .arg(Arg::with_name("INPUT")
                        .help("Binary file to analyze")
                        .required(true)
                        .index(1))
                    .get_matches();

    let filename = matches.value_of("INPUT").unwrap();
    let bv = BinaryView::new_from_filename(filename).unwrap();

    for func in bv.functions().iter().take(2) {
        print!("{:?}\n", func.name());
    }

    let num_funcs = 3;
    println!("---------- HLIL FIRST {} FUNCS ----------", num_funcs);
    for func in bv.functions().iter().take(num_funcs) {
        for bb in func.hlil()?.blocks().iter().take(num_funcs) {
            print!("Func: {:?}\n", func.name());
            for il in bb.il().iter() {
                print!("{:#x}: {}\n", il.address, il);
            }
        }
    };

    /*
    println!("---------- HLIL SSA ----------");
    for func in bv.functions().iter() {
        for bb in func.hlil()?.ssa_form()?.blocks() {
            for il in bb.il().iter().take(10) {
                println!("[{:?}] {:?} - {}", il.instr_index, il.operation, il);
            }
        }
    }
    */

    Ok(())
}
