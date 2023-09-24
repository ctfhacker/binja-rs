use std::collections::VecDeque;
use std::path::Path;

use clap::Parser;

use binja_rs::binaryview::BinaryView;
use binja_rs::highlevelil::HighLevelILOperation;
use binja_rs::highlevelil::HighLevelILOperation::*;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    input: String,
}

fn main() {
    let args = Args::parse();

    let bv = BinaryView::new_from_filename(&args.input).unwrap();

    // Get all the HLILSSA expressions in the binary
    let exprs = bv.hlilssa_expressions();
    print!("Exprs: {}\n", exprs.len());

    let mut instrs = VecDeque::new();

    // Filter all the HLILSSA instructions
    for instr in exprs {
        /*
        if matches!(*instr.operation, HighLevelILOperation::Tailcall { .. }) {
            instrs.push(vec![instr.clone()]);
        }
        */
        if matches!(*instr.operation, HighLevelILOperation::Call { .. }) {
            instrs.push_back(vec![instr.clone()]);
        }
        /*
        if matches!(*instr.operation, HighLevelILOperation::CallSsa {..}) {
            instrs.push(vec![instr.clone()]);
        }
        */
    }

    // Initialize the vec to store all the results that we find
    let mut result = Vec::new();

    // Start the timer for metrics once we are done
    let start = std::time::Instant::now();

    // Start the queue loop processing each slice one at a time, looking for the
    // end condition
    'top: loop {
        println!("Instrs left: {}", instrs.len());

        // Get the next slice to process
        let curr_slice = instrs.pop_front();

        // If there are no more slices to process, we are done!
        if curr_slice.is_none() {
            break;
        }

        let curr_slice = curr_slice.unwrap();
        let curr_instr = curr_slice.last().unwrap();

        // Check the last instruction of the current slice to see if it matches
        // our "stop" condition. In this case, we want to stop slicing if we
        // find an `Add` or `Sub` instruction
        let mut check = curr_instr;
        let mut count = 0;

        // This loop will walk the HLIL Operation tree looking for the Operation filter
        // that we care about. The `check` variable is reassigned each iteration, going
        // deeper into the tree. This loop is just an example and would be further
        // fleshed out in more useful plugins.
        loop {
            match &*check.operation {
                Assign { src, .. } => {
                    check = src;
                }
                AssignMemSsa { dest, .. } => {
                    check = dest;
                }
                DerefSsa { src, .. } => {
                    check = src;
                }
                Add { left, right } => {
                    if let Const { constant } = &*left.operation {
                        if *constant == 8 {
                            result.push(curr_slice);
                            continue 'top;
                        }
                    }

                    if let Const { constant } = &*right.operation {
                        if *constant == 8 {
                            result.push(curr_slice);
                            continue 'top;
                        }
                    }
                }
                Sub { left, right } => {
                    if let Const { constant } = &*left.operation {
                        if *constant == 8 {
                            result.push(curr_slice);
                            continue 'top;
                        }
                    }

                    if let Const { constant } = &*right.operation {
                        if *constant == 8 {
                            result.push(curr_slice);
                            continue 'top;
                        }
                    }
                }
                _ => break,
            }

            // Sanity limit on how far to recurse into the tree in case we are in a loop
            count += 1;
            if count > 8 {
                break;
            }
        }

        if curr_slice.len() > 16 {
            println!("Curr slice too long.. next one");
            result.push(curr_slice);
            continue;
        }

        // Last instruction didn't match our filter, so continue slicing backwards
        if let Some(vars) = curr_instr.ssa_vars() {
            for ssa_var in vars {
                // Get the list of xref_slices. This holds a vector for each
                // definition. This also will attempt to slice backwards through
                // a function using the `arg#` as an indicator of which call
                // parameter to slice with.
                let def = ssa_var.hlil_definition(&bv);

                /*
                // Debug print to check when we hit an operation that we haven't
                // yet handled in the bindings
                if let Err(unknown_op) = &def {
                    for (i, instr) in curr_slice.iter().enumerate() {
                        print!("[{}/{}][{:#x}] {}\n", i+1, curr_slice.len(), instr.address, instr);
                    }

                    print!("{}\n", unknown_op);
                    continue;
                }
                */

                // For each variable definition, add it to the slice if it isn't
                // already in the slice to mitigate infinite recursion
                for src in def.unwrap() {
                    if src == *curr_instr {
                        continue;
                    }

                    if !curr_slice.contains(&src) {
                        let mut new_slice = curr_slice.clone();
                        new_slice.push(src);
                        instrs.push_front(new_slice);
                    }
                }
            }
        }
    }

    // For each resulting slice, dump it to the screen
    for slice in result.iter() {
        print!("\n");
        for (i, instr) in slice.iter().enumerate() {
            let func_name = if let Some(name) = instr.function.function().name() {
                name.to_string()
            } else {
                "???".to_string()
            };

            print!(
                "[{}/{}][{:#x}:{}] {}\n",
                i + 1,
                slice.len(),
                instr.address,
                func_name,
                instr
            );
        }
    }

    print!("Time taken: {:?}\n", start.elapsed());
}
