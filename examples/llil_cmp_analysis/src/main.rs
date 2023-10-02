use anyhow::Result;
use clap::Parser;

use std::collections::HashSet;
use std::fs::File;
use std::io::Write;

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

    #[arg(long)]
    max_analysis_time: Option<u64>,

    #[arg(long)]
    max_function_size: Option<u64>,

    #[arg(long, default_value = "full")]
    mode: Option<String>,
}

timeloop::impl_enum!(
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum Timer {
        BinaryNinja,
        ParseExpressions,
        WriteRules,
    }
);

/// strcmp-like symbols to treat as if it's a strcmp
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

/// memcmp-like symbols to treat as if it's a memcmp
const MEMCMPS: &[&'static str] = &[
    "memcmp",
    "bcmp",
    "CRYPTO_memcmp",
    "OPENSSL_memcmp",
    "memcmp_const_time",
    "memcmpct",
];

/// strncmp-like symbols to treat as if it's a strncmp
const STRNCMPS: &[&'static str] = &["strncmp", "xmlStrncmp", "curl_strnequal"];

/// strcasecmp-like symbols to treat as if it's a strcasecmp
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

/// strncasecmp-like symbols to treat as if it's a strncasecmp
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
    let start = std::time::Instant::now();

    timeloop::start_profiler!();

    let out_file = format!("{}.cmps", args.input);
    println!("Writing cmp rules to {out_file}");
    let mut out_file = File::create(out_file)?;

    // Log Info to stdout
    binjalog::to_stdout(binjalog::LogLevel::Info);

    let mut bv = timeloop::time_work!(Timer::BinaryNinja, {
        let base_addr = args
            .base_addr
            .map(|x| u64::from_str_radix(&x.replace("0x", ""), 16).unwrap());

        let mut bv = binaryview::BinaryView::new_from_filename(&args.input);
        bv = bv.base_addr(base_addr);

        if let Some(max_analysis_time) = args.max_analysis_time {
            bv = bv.option("analysis.limits.maxFunctionAnalysisTime", max_analysis_time);
        }

        if let Some(max_function_size) = args.max_function_size {
            bv = bv.option("analysis.limits.maxFunctionSize", max_function_size);
        }

        bv = bv.option("analysis.mode", "basic");
        bv = bv.option("analysis.linearSweep.detailedLogInfo", true);

        // Build the BinaryView after adding the options
        bv.build().unwrap()
    });

    /// Returns `true` if the given function contains a substring related to ASAN
    fn filter_non_asan(func: &binja_rs::function::Function) -> bool {
        const DEFAULT_IGNORE: &[&'static str] =
            &["asan", "ubsan", "msan", "lcov", "sanitizer", "interceptor"];

        // Check if the given function contains any of the blacklisted functions
        let res = DEFAULT_IGNORE.iter().any(|bad_func| {
            func.name()
                .map(|f| f.to_string().contains(bad_func))
                .unwrap_or(false)
        });

        // Return `true` if it is NOT an asan function, and `false` otherwise
        !res
    }

    use LowLevelILOperation::*;
    timeloop::time_work!(Timer::ParseExpressions, {
        let now = std::time::Instant::now();
        for instrs in bv.llil_expressions_filtered(filter_non_asan)
        // .par_llil_expressions_filtered(filter_non_asan)
        // .into_iter()
        {
            let instrs = [instrs];

            for instr in instrs {
                match &*instr.operation {
                    Equal { left, right }
                    | NotEqual { left, right }
                    | UnsignedLessThan { left, right }
                    | UnsignedLessThanEqual { left, right }
                    | UnsignedGreaterThan { left, right }
                    | UnsignedGreaterThanEqual { left, right }
                    | SignedLessThan { left, right }
                    | SignedLessThanEqual { left, right }
                    | SignedGreaterThan { left, right }
                    | SignedGreaterThanEqual { left, right }
                    | FcmpE { left, right }
                    | FcmpNe { left, right }
                    | FcmpLt { left, right }
                    | FcmpLe { left, right }
                    | FcmpGe { left, right }
                    | FcmpGt { left, right } => {
                        let mut size = instr.size;
                        let address = instr.address;

                        let operation = match &*instr.operation {
                            Equal { .. } => "CMP_E",
                            NotEqual { .. } => "CMP_NE",
                            UnsignedLessThan { .. } => "CMP_ULT",
                            UnsignedLessThanEqual { .. } => "CMP_ULE",
                            UnsignedGreaterThan { .. } => "CMP_UGT",
                            UnsignedGreaterThanEqual { .. } => "CMP_UGE",
                            SignedLessThan { .. } => "CMP_SLT",
                            SignedLessThanEqual { .. } => "CMP_SLE",
                            SignedGreaterThan { .. } => "CMP_SGT",
                            SignedGreaterThanEqual { .. } => "CMP_SGE",
                            FcmpE { .. } => "FCMP_E",
                            FcmpNe { .. } => "FCMP_NE",
                            FcmpLt { .. } => "FCMP_LT",
                            FcmpLe { .. } => "FCMP_LE",
                            FcmpGt { .. } => "FCMP_GT",
                            FcmpGe { .. } => "FCMP_GE",
                            _ => panic!("Invalid operation: {:?}", instr.operation),
                        };

                        // Check if this rule should collapse it's definition register
                        if let Some(collapsed_rules) =
                            check_collapse_rule(&mut bv, left.clone(), operation, right)?
                        {
                            for rule in collapsed_rules {
                                let new_rule = format!("{address:#x},{size:#x},{rule}\n");
                                out_file.write(new_rule.as_bytes());
                            }
                            continue;
                        }

                        // Parse and add this rule
                        let mut left_str = String::new();
                        let mut right_str = String::new();
                        get_cmp_operand(&mut bv, &left, &mut left_str)?;
                        get_cmp_operand(&mut bv, &right, &mut right_str)?;

                        let res =
                            format!("{address:#x},{size:#x},{left_str},{operation},{right_str}\n");
                        out_file.write(res.as_bytes());
                    }
                    LowLevelILOperation::Call {
                        dest: LowLevelILInstruction { operation, .. },
                    } => match **operation {
                        LowLevelILOperation::ConstPtr { constant } => {
                            let name = match bv.get_symbol_at(constant) {
                                Ok(symbol) => symbol.name().to_string(),
                                Err(_) => format!("UnknownSymbol: {constant:#x}"),
                            };

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
                                    "{:#x},{:#x},reg {arg1},strcmp,reg {arg2}\n",
                                    instr.address, instr.size
                                )
                            } else if MEMCMPS.contains(&name.as_str()) {
                                // A three argument function call with a dynamic size
                                format!(
                                    "{:#x},reg {arg3},reg {arg1},memcmp,reg {arg2}\n",
                                    instr.address
                                )
                            } else {
                                // eprintln!("Ignoring function call {name:?}");
                                continue;
                            };

                            // Add this function call to the results
                            out_file.write(res.as_bytes());
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
    });

    timeloop::print!();

    println!("Total time: {:?}", start.elapsed());

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

                let Some(def) = func_ssa.get_ssa_reg_definition(src) else {
                    panic!(
                        "No SSA reg for this variable? {:#x} {instr:?}",
                        instr.address
                    );
                };

                let LowLevelILOperation::SetRegSsa { dest, src } = &*def.operation else {
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
        LowLevelILOperation::SignExtend { src } => {
            output.push_str("sign_extend ");
            get_cmp_operand(bv, src, output)?;
        }
        LowLevelILOperation::ZeroExtend { src } => {
            output.push_str("zero_extend ");
            get_cmp_operand(bv, src, output)?;
        }
        LowLevelILOperation::LowPart { src } => {
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
        | LowLevelILOperation::Asr { left, right }
        | LowLevelILOperation::Equal { left, right }
        | LowLevelILOperation::NotEqual { left, right }
        | LowLevelILOperation::UnsignedLessThan { left, right }
        | LowLevelILOperation::UnsignedLessThanEqual { left, right }
        | LowLevelILOperation::UnsignedGreaterThan { left, right }
        | LowLevelILOperation::UnsignedGreaterThanEqual { left, right }
        | LowLevelILOperation::SignedLessThan { left, right }
        | LowLevelILOperation::SignedLessThanEqual { left, right }
        | LowLevelILOperation::SignedGreaterThan { left, right }
        | LowLevelILOperation::SignedGreaterThanEqual { left, right }
        | LowLevelILOperation::MuluDp { left, right } => {
            let name = match *instr.operation {
                LowLevelILOperation::Add { .. } => "add",
                LowLevelILOperation::Sub { .. } => "sub",
                LowLevelILOperation::And { .. } => "and",
                LowLevelILOperation::Or { .. } => "or",
                LowLevelILOperation::Xor { .. } => "xor",
                LowLevelILOperation::Lsl { .. } => "lsl",
                LowLevelILOperation::Lsr { .. } => "lsr",
                LowLevelILOperation::Asr { .. } => "asr",
                LowLevelILOperation::MuluDp { .. } => "mul",
                LowLevelILOperation::Equal { .. } => "eq",
                LowLevelILOperation::NotEqual { .. } => "not_eq",
                LowLevelILOperation::UnsignedLessThan { .. } => "u<",
                LowLevelILOperation::UnsignedLessThanEqual { .. } => "u<=",
                LowLevelILOperation::UnsignedGreaterThan { .. } => "u>",
                LowLevelILOperation::UnsignedGreaterThanEqual { .. } => "u>=",
                LowLevelILOperation::SignedLessThan { .. } => "s<",
                LowLevelILOperation::SignedLessThanEqual { .. } => "s<=",
                LowLevelILOperation::SignedGreaterThan { .. } => "s>",
                LowLevelILOperation::SignedGreaterThanEqual { .. } => "s>=",
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
        LowLevelILOperation::Not { src } => {
            output.push_str("not ");
            get_cmp_operand(bv, src, output)?;
        }
        _ => println!("Unknown instr.operation: {:?}", instr.operation),
    }

    Ok(())
}

/// This common pattern compares the result of an operation. We
/// want to create the rule that would actually produce the result.
///
/// eax = eax & ecx
/// al = eax != 0
///
/// Would produce the rule: reg eax,CMP_NE,0x0
/// Want to produce:        and reg eax reg ecx,CMP_NE,0x0
///
/// In this case, we want to check eax & ecx == 0 and not just the result
fn check_collapse_rule(
    bv: &mut BinaryView,
    left: LowLevelILInstruction,
    operation: &str,
    right: &LowLevelILInstruction,
) -> Result<Option<HashSet<String>>> {
    use LowLevelILOperation::*;

    let left_defs = left.definition()?;
    let mut lines = HashSet::new();

    if !left_defs.is_empty()
        && (matches!(*left.operation, Reg { .. }) && matches!(*right.operation, Const { constant }))
    {
        let Const { constant } = *right.operation else {
            unreachable!();
        };

        for (i, left_def) in left_defs.iter().enumerate() {
            match &*left_def.operation {
                Add { left, right, .. }
                | Adc { left, right, .. }
                | Sub { left, right, .. }
                | Sbb { left, right, .. }
                | And { left, right, .. }
                | Or { left, right, .. }
                | Xor { left, right, .. }
                | Lsl { left, right, .. }
                | Lsr { left, right, .. }
                | Asr { left, right, .. }
                | Rol { left, right, .. }
                | Rlc { left, right, .. }
                | Ror { left, right, .. }
                | Rrc { left, right, .. }
                | Mul { left, right, .. }
                | MuluDp { left, right, .. }
                | MulsDp { left, right, .. }
                | Divu { left, right, .. }
                | DivuDp { left, right, .. }
                | Divs { left, right, .. }
                | DivsDp { left, right, .. }
                | Modu { left, right, .. }
                | ModuDp { left, right, .. }
                | Mods { left, right, .. }
                | ModsDp { left, right, .. }
                | Equal { left, right, .. }
                | NotEqual { left, right, .. }
                | SignedLessThan { left, right, .. }
                | UnsignedLessThan { left, right, .. }
                | SignedLessThanEqual { left, right, .. }
                | UnsignedLessThanEqual { left, right, .. }
                | SignedGreaterThanEqual { left, right, .. }
                | UnsignedGreaterThanEqual { left, right, .. }
                | SignedGreaterThan { left, right, .. }
                | UnsignedGreaterThan { left, right, .. }
                | TestBit { left, right, .. }
                | AddOverflow { left, right, .. }
                | Fadd { left, right, .. }
                | Fsub { left, right, .. }
                | Fmul { left, right, .. }
                | Fdiv { left, right, .. }
                | FcmpE { left, right, .. }
                | FcmpNe { left, right, .. }
                | FcmpLt { left, right, .. }
                | FcmpLe { left, right, .. }
                | FcmpGe { left, right, .. }
                | FcmpGt { left, right, .. }
                | FcmpO { left, right, .. }
                | FcmpUo { left, right, .. } => {
                    let mut left_str = String::new();
                    let mut right_str = String::new();
                    get_cmp_operand(bv, &left, &mut left_str)?;
                    get_cmp_operand(bv, &right, &mut right_str)?;

                    // Use the operation of a conditional if the conditional
                    // is in the definition
                    let operation = match &*left_def.operation {
                        Equal { .. } => "CMP_E",
                        NotEqual { .. } => "CMP_NE",
                        UnsignedLessThan { .. } => "CMP_ULT",
                        UnsignedLessThanEqual { .. } => "CMP_ULE",
                        UnsignedGreaterThan { .. } => "CMP_UGT",
                        UnsignedGreaterThanEqual { .. } => "CMP_UGE",
                        SignedLessThan { .. } => "CMP_SLT",
                        SignedLessThanEqual { .. } => "CMP_SLE",
                        SignedGreaterThan { .. } => "CMP_SGT",
                        SignedGreaterThanEqual { .. } => "CMP_SGE",
                        FcmpE { .. } => "FCMP_E",
                        FcmpNe { .. } => "FCMP_NE",
                        FcmpLt { .. } => "FCMP_LT",
                        FcmpLe { .. } => "FCMP_LE",
                        FcmpGt { .. } => "FCMP_GT",
                        FcmpGe { .. } => "FCMP_GE",
                        op => operation,
                    };

                    let res = format!("{left_str},{operation},{right_str}");
                    lines.insert(res);

                    if constant != 0 {
                        for adjustment in ["add", "sub"] {
                            let res = format!(
                                "{left_str},{operation},{adjustment} {constant:#x} {right_str}",
                            );
                            lines.insert(res);

                            let res = format!(
                                "{adjustment} {constant} {left_str},{operation},{right_str}",
                            );
                            lines.insert(res);
                        }
                    }

                    continue;
                }
                Call { .. }
                | Load { .. }
                | Reg { .. }
                | Const { .. }
                | Flag { .. }
                | SignExtend { .. }
                | LowPart { .. }
                | Neg { .. }
                | ConstPtr { .. }
                | FloatToInt { .. }
                | ZeroExtend { .. } => {
                    // Valid operation, keep the original left operand
                }
                x => {
                    println!("ERROR Is this an SSA def: {:#x} {x:?}", left.address);
                    continue;

                    /*
                    println!(
                        "{:#x} -> {:#x}",
                        left.address,
                        left_def.unwrap().address
                    );
                    println!("{:x?} -> {:x?}", left, left_def);
                    */
                }
            }
        }
    }

    if lines.is_empty() {
        Ok(None)
    } else {
        Ok(Some(lines))
    }
}
