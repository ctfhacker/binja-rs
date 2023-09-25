use anyhow::Result;
use clap::Parser;

use binja_rs::binaryview::BinaryView;
use binja_rs::lowlevelil::{LowLevelILInstruction, LowLevelILOperation};
use binja_rs::*;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    input: String,

    #[arg(long)]
    base_addr: Option<String>,
}

timeloop::impl_enum!(
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum Timer {
        BinaryNinja,
        ParseExpressions,
    }
);

timeloop::create_profiler!(Timer);

fn main() -> Result<()> {
    let args = Args::parse();

    timeloop::start_profiler!();

    let mut bv = timeloop::time_work!(Timer::BinaryNinja, {
        binaryview::BinaryView::new_from_filename(&args.input)
            .base_addr(
                args.base_addr
                    .map(|x| u64::from_str_radix(&x.replace("0x", ""), 16).unwrap()),
            )
            .build()
            .unwrap()
    });

    let mut lines = Vec::new();
    timeloop::time_work!(Timer::ParseExpressions, {
        let now = std::time::Instant::now();
        for instr in bv.llil_expressions().iter() {
            match &*instr.operation {
                LowLevelILOperation::Equals { left, right }
                | LowLevelILOperation::NotEquals { left, right }
                | LowLevelILOperation::NotEquals { left, right } => {
                    let operation = match &*instr.operation {
                        LowLevelILOperation::Equals { left, right } => "CMP_E",
                        LowLevelILOperation::NotEquals { left, right } => "CMP_NE",
                        _ => panic!("Invalid operation: {:?}", instr.operation),
                    };

                    let mut left_str = String::new();
                    let mut right_str = String::new();
                    get_cmp_operand(&mut bv, left, &mut left_str)?;
                    get_cmp_operand(&mut bv, right, &mut right_str)?;

                    let res = format!(
                        "{:#x},{:#x},{left_str},{operation},{right_str}",
                        instr.address, instr.size,
                    );

                    lines.push(res);
                }
                _ => {}
            }
        }
    });

    for line in lines {
        println!("{line}");
    }
    timeloop::print!();

    Ok(())
}

pub fn get_cmp_operand(
    bv: &mut BinaryView,
    instr: &LowLevelILInstruction,
    output: &mut String,
) -> Result<()> {
    match &*instr.operation {
        LowLevelILOperation::Reg { src } => {
            // If this register name is a temp register, go to the ssa reg definition
            // and use that instead
            //
            // Example:
            //
            // temp0.q -> rax
            //
            // 86 @ 00016f89  temp0.q = rax
            // 89 @ 00016f92  if (temp0.q != temp1.q) then 106 @ 0x170bd else 108 @ 0x16f98

            let name = format!("{src:?}");
            if name.contains(&"temp") {
                let func_ssa = instr.function.ssa_form()?;

                let LowLevelILOperation::RegSsa { src } = &*instr.ssa_form()?.operation else {
                    panic!("Failed to get SSA");
                };

                let LowLevelILOperation::SetRegSsa { dest, src } =
                    &*func_ssa.get_ssa_reg_definition(src).operation
                else {
                    panic!("Failed to get SetRegSsa");
                };

                get_cmp_operand(bv, &src.non_ssa_form()?, output);
            } else {
                output.push_str(&format!("reg {name}"));
            }
        }
        LowLevelILOperation::Load { src } => {
            output.push_str("load ");
            get_cmp_operand(bv, src, output)?;
        }
        LowLevelILOperation::ConstPtr { constant } => {
            output.push_str(&format!("{constant:#x}"));
        }
        LowLevelILOperation::Const { constant } => {
            output.push_str(&format!("{constant:#x}"));
        }
        LowLevelILOperation::Add { left, right }
        | LowLevelILOperation::Sub { left, right }
        | LowLevelILOperation::Or { left, right }
        | LowLevelILOperation::Xor { left, right }
        | LowLevelILOperation::And { left, right }
        | LowLevelILOperation::Lsr { left, right }
        | LowLevelILOperation::Lsl { left, right }
        | LowLevelILOperation::MuluDp { left, right } => {
            let name = match *instr.operation {
                LowLevelILOperation::Add { .. } => "add",
                LowLevelILOperation::Sub { .. } => "sub",
                LowLevelILOperation::And { .. } => "and",
                LowLevelILOperation::Or { .. } => "or",
                LowLevelILOperation::Xor { .. } => "xor",
                LowLevelILOperation::Lsr { .. } => "logical_shift_right",
                LowLevelILOperation::Lsl { .. } => "logical_shift_left",
                LowLevelILOperation::MuluDp { .. } => "mul",
                _ => panic!("Invalid arithmetic operation"),
            };

            output.push_str(&format!("{name} "));
            get_cmp_operand(bv, left, output)?;
            output.push(' ');
            get_cmp_operand(bv, right, output)?;
        }
        LowLevelILOperation::Neg { src } => {
            output.push_str("neg ");
            get_cmp_operand(bv, src, output)?;
        }
        _ => panic!("Unknown: {instr:?}"),
    }

    Ok(())
}
