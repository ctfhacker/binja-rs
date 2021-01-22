import binaryninja

def make_camel(s):
    return ''.join([x.title() for x in s.lower().split("_")[1:]])


'''
More verbose names to use for operations
'''
name_translation = {
    'Sx': 'SignExtend',
    'Zx': 'ZeroExtend',
    'CmpE': 'Equals',
    'CmpNe': 'NotEquals',
    'CmpSlt': 'SignedLessThan',
    'CmpUlt': 'UnsignedLessThan',
    'CmpSle': 'SignedLessThanEquals',
    'CmpUle': 'UnsignedLessThanEquals',
    'CmpSge': 'SignedGreaterThanEquals',
    'CmpUge': 'UnsignedGreaterThanEquals',
    'CmpSgt': 'SsignedGreaterThan',
    'CmpUgt': 'UnsignedGreaterThan',
}

'''
First, gather all possible types so that the macros for each type are listed above the enum
'''
llil_types = []
for (k, v) in binaryninja.lowlevelil.LowLevelILInstruction.ILOperations.items():
    for (name, typ) in v:
        llil_types.append(typ)

llil_types = set(llil_types)

''' 
Initialize the enum for the Operations
'''
print('enum LowLevelILOperation {')

for (k, v) in binaryninja.lowlevelil.LowLevelILInstruction.ILOperations.items():
    name = make_camel(str(k))

    # Translate the name into a more verbose name, or use the original if not found
    name = name_translation.get(name, name)

    translation = {
        'expr': 'LowLevelILInstruction',
        'int': 'u64',
        'reg_ssa': 'SSARegister',
        'reg': 'Register',
        'flag': 'Flag',
        'flag_ssa': 'SSAFlag',
        'reg_ssa_list': 'Vec<SSARegister>',
        'expr_list': 'Vec<LowLevelILInstruction>',
        'int_list': 'Vec<u64>',
        'target_map': 'HashMap<u64, u64>',
        'flag_ssa_list': 'Vec<SSAFlag>',
    }

    keys = '{ '
    for (n, typ) in v:
        keys += n
        keys += ': '
        keys += translation.get(typ, typ)
        keys += ', '

    keys += '}'

    print('    {0} {1},'.format(name, keys))

print('}')

''' 
Implement the switch statement to create the Operation
'''
print('impl LowLevelILOperation {')

print('    pub fn from_instr(instr: BNLowLevelILInstruction, func: &LowLevelILFunction, expr_index: u64)')
print('            -> LowLevelILOperation {')

print('        let arch = func.arch().expect("Failed to get arch for LLIL").clone();')
print('        let mut operand_index = 0;')

'''
Dump the generic `unimplemented` version of each macro in the function itself 
'''
for t in llil_types:
    print('''
            macro_rules! {0} {{
                () => {{{{
                    unimplemented!("{0}");
                }}}}
            }}'''.format(t))

print('        match instr.operation {')

for (k, v) in binaryninja.lowlevelil.LowLevelILInstruction.ILOperations.items():
    name = make_camel(str(k))

    # Translate the name into a more verbose name, or use the original if not found
    name = name_translation.get(name, name)

    print('            BN{} => {{'.format(str(k).replace(".", "_")))
    for (kind, typ) in v:
        print('                let {} = {}!();'.format(kind, typ))
    print('                LowLevelILOperation::{} {{'.format(name))
    print('                    {}'.format(', '.join([str(name) for (name, typ) in v])))
    print('                }')
    print('            }')

print('        }')
print('    }')


print('}')
