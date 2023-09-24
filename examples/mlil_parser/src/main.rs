use anyhow::Result;
use clap::Parser;

use binja_rs::binaryview::BinaryView;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    input: String,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let bv = BinaryView::new_from_filename(&args.input).unwrap();

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
