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

    print!("HLIL instructions\n");

    let now = std::time::Instant::now();
    for instr in bv.hlil_instructions().iter().take(10) {
        print!("{}\n", instr);
    }
    print!("Took {:?}\n", now.elapsed());

    print!("HLIL instructions gathered in parallel\n");

    let now = std::time::Instant::now();
    for instr in bv.par_hlil_instructions().iter().take(10) {
        print!("{}\n", instr);
    }
    print!("Took {:?}\n", now.elapsed());

    Ok(())
}
