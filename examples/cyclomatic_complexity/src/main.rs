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
    let matches = App::new("Binja cyclomatic_complexity")
                    .version("0.1")
                    .author("@ctfhacker")
                    .about("Dumping the cyclomatic complexity of all functions")
                    .arg(Arg::with_name("INPUT")
                        .help("Binary file to analyze")
                        .required(true)
                        .index(1))
                    .get_matches();

    let filename = matches.value_of("INPUT").unwrap();
    let bv = BinaryView::new_from_filename(filename).unwrap();

    let mut function_connections = Vec::new();
    let mut bb_connections = Vec::new();

    for func in bv.functions() {
        let callees = func.callees(&bv)?.len();
        let callers = func.callers(&bv).len();

        // print!("Func: {:?} callees: {} callers: {}\n", func.name(), callees, callers);

        function_connections.push((callees + callers, func.clone()));

        let blocks = func.blocks();
        // print!("Blocks: {}\n", blocks.len());
        for bb in blocks {
            let edges = bb.total_edges();
            // print!("    edges: {}\n", edges);
            bb_connections.push((edges, bb.clone()));
        }
    }

    // Sort the functions by the first item: callees + callers
    // Note: the b.cmp(a) vs a.cmp(b). b.cmp(a) will be descending order
    function_connections.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    // Sort the basic blocks by the first item: total edges
    // Note: the b.cmp(a) vs a.cmp(b). b.cmp(a) will be descending order
    bb_connections.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    print!("CONNECTED FUNCTIONS\n");
    for (count, f) in function_connections.iter().take(5) {
        print!("{}: {:?}\n", count, f);
    }

    print!("CONNECTED BASIC BLOCKS\n");
    for (count, b) in bb_connections.iter().take(5) {
        print!("{}: {:?}\n", count, b);
    }

    print!("DONE\n");

    Ok(())
}
