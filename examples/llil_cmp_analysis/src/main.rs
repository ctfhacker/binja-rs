use anyhow::Result;
use clap::Parser;

use binja_rs::binaryview::BinaryView;
use binja_rs::lowlevelil::{LowLevelILInstruction, LowLevelILOperation};
use binja_rs::mediumlevelil::{MediumLevelILInstruction, MediumLevelILOperation};
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
        WriteRules,
    }
);

const STRCMPS: &[&'static str] = &[
    "strcmp",
    "strncasecmp",
    "strcmp",
    "xmlStrcmp",
    "xmlStrEqual",
    "g_strcmp0",
    "curl_strequal",
    "strcsequal",
];

const MEMCMPS: &[&'static str] = &[
    "memcmp",
    "bcmp",
    "CRYPTO_memcmp",
    "OPENSSL_memcmp",
    "memcmp_const_time",
    "memcmpct",
];

const STRNCMPS: &[&'static str] = &["strncmp", "xmlStrncmp", "curl_strnequal"];

const STRCASECMPS: &[&'static str] = &[
    "strcasecmp",
    "stricmp",
    "ap_cstr_casecmp",
    "OPENSSL_strcasecmp",
    "xmlStrcasecmp",
    "g_strcasecmp",
    "g_ascii_strcasecmp",
    "Curl_strcasecompare",
    "Curl_safe_strcasecompare",
    "cmsstrcasecmp",
];

const STRNCASECMPS: &[&'static str] = &[
    "strncasecmp",
    "strnicmp",
    "ap_cstr_casecmpn",
    "OPENSSL_strncasecmp",
    "xmlStrncasecmp",
    "g_ascii_strncasecmp",
    "Curl_strncasecompare",
    "g_strncasecmp",
];

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
                LowLevelILOperation::Equal { left, right }
                | LowLevelILOperation::NotEqual { left, right }
                | LowLevelILOperation::UnsignedLessThan { left, right }
                | LowLevelILOperation::UnsignedLessThanEqual { left, right }
                | LowLevelILOperation::UnsignedGreaterThan { left, right }
                | LowLevelILOperation::UnsignedGreaterThanEqual { left, right }
                | LowLevelILOperation::SignedLessThan { left, right }
                | LowLevelILOperation::SignedLessThanEqual { left, right }
                | LowLevelILOperation::SignedGreaterThan { left, right }
                | LowLevelILOperation::SignedGreaterThanEqual { left, right }
                | LowLevelILOperation::FcmpE { left, right }
                | LowLevelILOperation::FcmpNe { left, right }
                | LowLevelILOperation::FcmpLt { left, right }
                | LowLevelILOperation::FcmpLe { left, right }
                | LowLevelILOperation::FcmpGe { left, right }
                | LowLevelILOperation::FcmpGt { left, right } => {
                    let mut size = instr.size;
                    let address = instr.address;

                    let operation = match &*instr.operation {
                        LowLevelILOperation::Equal { .. } => "CMP_E",
                        LowLevelILOperation::NotEqual { .. } => "CMP_NE",
                        LowLevelILOperation::UnsignedLessThan { .. } => "CMP_ULT",
                        LowLevelILOperation::UnsignedLessThanEqual { .. } => "CMP_ULE",
                        LowLevelILOperation::UnsignedGreaterThan { .. } => "CMP_UGT",
                        LowLevelILOperation::UnsignedGreaterThanEqual { .. } => "CMP_UGE",
                        LowLevelILOperation::SignedLessThan { .. } => "CMP_SLT",
                        LowLevelILOperation::SignedLessThanEqual { .. } => "CMP_SLE",
                        LowLevelILOperation::SignedGreaterThan { .. } => "CMP_SGT",
                        LowLevelILOperation::SignedGreaterThanEqual { .. } => "CMP_SGE",
                        LowLevelILOperation::FcmpE { .. } => "FCMP_E",
                        LowLevelILOperation::FcmpNe { .. } => "FCMP_NE",
                        LowLevelILOperation::FcmpLt { .. } => "FCMP_LT",
                        LowLevelILOperation::FcmpLe { .. } => "FCMP_LE",
                        LowLevelILOperation::FcmpGt { .. } => "FCMP_GT",
                        LowLevelILOperation::FcmpGe { .. } => "FCMP_GE",
                        _ => panic!("Invalid operation: {:?}", instr.operation),
                    };

                    let mut left_str = String::new();
                    let mut right_str = String::new();
                    get_cmp_operand(&mut bv, left, &mut left_str)?;
                    get_cmp_operand(&mut bv, right, &mut right_str)?;

                    let res = format!("{address:#x},{size:#x},{left_str},{operation},{right_str}",);

                    lines.push(res);
                }
                LowLevelILOperation::Call {
                    dest: LowLevelILInstruction { operation, .. },
                } => match **operation {
                    LowLevelILOperation::ConstPtr { constant } => {
                        let name = bv.get_symbol_at(constant)?.name();

                        // Get the registers for this function's calling convention
                        let mut function = instr.function.function();
                        let calling_convention = function.calling_convention()?;
                        let arg1 = &calling_convention.int_arg_regs[0];
                        let arg2 = &calling_convention.int_arg_regs[1];
                        let arg3 = &calling_convention.int_arg_regs[2];

                        let res = if STRCMPS.contains(&name.as_str())
                            || STRCASECMPS.contains(&name.as_str())
                            || STRNCMPS.contains(&name.as_str())
                            || STRNCASECMPS.contains(&name.as_str())
                        {
                            // A two argument function call
                            format!(
                                "{:#x},{:#x},reg {arg1},strcmp,reg {arg2}",
                                instr.address, instr.size
                            )
                        } else if MEMCMPS.contains(&name.as_str()) {
                            // A three argument function call with a dynamic size
                            format!(
                                "{:#x},reg {arg3},reg {arg1},memcmp,reg {arg2}",
                                instr.address
                            )
                        } else {
                            eprintln!("Ignoring function call {name:?}");
                            continue;
                        };

                        // Add this function call to the results
                        lines.push(res);
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    });

    timeloop::time_work!(Timer::BinaryNinja, {
        let out_file = format!("{}.cmps", args.input);
        std::fs::write(&out_file, lines.join("\n"));
        println!("Output rules written to {out_file}");
    });

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
            output.push_str("load_from ");
            get_cmp_operand(bv, src, output)?;
        }
        LowLevelILOperation::ConstPtr { constant } => {
            output.push_str(&format!("{constant:#x}"));
        }
        LowLevelILOperation::Const { constant } => {
            let sign = if constant.is_negative() { "-" } else { "" };
            output.push_str(&format!("{sign}{:#x}", constant.abs()));
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
