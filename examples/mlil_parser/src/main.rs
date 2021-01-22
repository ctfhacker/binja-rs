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
    let matches = App::new("Binja mlil_parse")
                    .version("0.1")
                    .author("@ctfhacker")
                    .about("Example using some of the MLIL functionality")
                    .arg(Arg::with_name("INPUT")
                        .help("Binary file to analyze")
                        .required(true)
                        .index(1))
                    .get_matches();

    let filename = matches.value_of("INPUT").unwrap();
    let bv = BinaryView::new_from_filename(filename).unwrap();

    println!("---------- MLIL ----------");
    bv.functions().par_iter().for_each(|func| {
        for bb in func.mlil().unwrap().blocks() {
            for il in bb.il().iter() {
                println!("[{:?}:{}] {:?} - {}", func.name(), il.instr_index.unwrap_or(!0), il.operation, il);
            }
        }
    });

    /*
    println!("---------- MLIL ----------");
    for func in bv.functions().iter().filter(|x| x.name().unwrap() == "_init".to_string()) {
        println!("func: {:?}", func.name().unwrap());
        for bb in func.medium_level_il().blocks() {
            for il in bb.il().iter().take(10) {
                println!("[{}] {:?} - {}", il.instr_index, il.operation, il);
            }
        }
    }
    */

    println!("---------- MLIL SSA ----------");
    for func in bv.functions().iter() {
        println!("SSA Func: {:?}", func.name().unwrap());
        for bb in func.medium_level_il()?.ssa_form()?.blocks() {
            for il in bb.il().iter().take(10) {
                println!("[{:?}] {:?} - {}", il.instr_index, il.operation, il);
            }
        }
    }

    Ok(())
}
