extern crate clap;
extern crate binja_rs;
extern crate anyhow;

extern crate rayon;

use clap::{App, Arg};
use anyhow::Result;
use rayon::prelude::*;

use binja_rs::*;
use binja_rs::traits::*;

fn main() -> Result<()> {
    let matches = App::new("Binja llil_parse")
                    .version("0.1")
                    .author("@ctfhacker")
                    .about("Example using some of the LLIL functionality")
                    .arg(Arg::with_name("INPUT")
                        .help("Binary file to analyze")
                        .required(true)
                        .index(1))
                    .get_matches();

    let filename = matches.value_of("INPUT").unwrap();
    let bv = binaryview::BinaryView::new_from_filename(filename).unwrap();

    /*
     OLD more verbose methods for getting LLIL and LLILSSA

    println!("---------- LLIL ----------");
    bv.functions().par_iter().for_each(|func| {
        for bb in func.llil().unwrap().blocks() {
            for il in bb.il().iter().take(3) {
                println!("[{:?}:{}] {:?} - {}", func.name(), il.instr_index.unwrap_or(!0), il.operation, il);
            }
        }
    });

    println!("---------- LLIL SSA ----------");
    bv.functions().par_iter().for_each(|func| {
        println!("func: {:?}", func.name().unwrap());
        for bb in func.low_level_il().expect("No LLIL")
                      .ssa_form().expect("No LLILSSA")
                      .blocks() {
            for il in bb.il().iter() {
                print!("{}\n", il);
            }
        }
    });
    */

    print!("LLIL instructions\n");

    let now = std::time::Instant::now();
    for instr in bv.llil_instructions().iter().take(10) {
        print!("{}\n", instr);
    }
    print!("Took {:?}\n", now.elapsed());

    print!("LLIL instructions gathered in parallel\n");

    let now = std::time::Instant::now();
    for instr in bv.par_llil_instructions().iter().take(10) {
        print!("{}\n", instr);
    }
    print!("Took {:?}\n", now.elapsed());

    Ok(())
}
