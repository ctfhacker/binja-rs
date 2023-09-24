use anyhow::Result;
use clap::Parser;
use rayon::prelude::*;

use binja_rs::binaryview::BinaryView;
use binja_rs::traits::*;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    input: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let bv = BinaryView::new_from_filename(&args.input).unwrap();

    let mut function_connections = Vec::new();
    let mut bb_connections = Vec::new();

    for func in bv.functions() {
        let callees = func.callees(&bv)?.len();
        let callers = func.callers(&bv).len();

        print!(
            "Func: {:?} callees: {} callers: {}\n",
            func.name(),
            callees,
            callers
        );

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
