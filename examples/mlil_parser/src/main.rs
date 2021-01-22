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

    print!("MLIL instructions\n");

    let now = std::time::Instant::now();
    for instr in bv.mlil_instructions().iter().take(10) {
        print!("{}\n", instr);
    }
    print!("Took {:?}\n\n", now.elapsed());

    print!("MLIL instructions gathered in parallel\n");

    let now = std::time::Instant::now();
    for instr in bv.par_mlil_instructions().iter().take(10) {
        print!("{}\n", instr);
    }
    print!("Took {:?}\n", now.elapsed());


    Ok(())
}
