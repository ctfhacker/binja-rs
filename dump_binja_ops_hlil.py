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
for (k, v) in binaryninja.highlevelil.HighLevelILInstruction.ILOperations.items():
    for (name, typ) in v:
        llil_types.append(typ)

llil_types = set(llil_types)

''' 
Initialize the enum for the Operations
'''
print('enum HighLevelILOperation {')

for (k, v) in binaryninja.highlevelil.HighLevelILInstruction.ILOperations.items():
    name = make_camel(str(k))

    # Translate the name into a more verbose name, or use the original if not found
    name = name_translation.get(name, name)

    translation = {
        'expr': 'HighLevelILInstruction',
        'int': 'u64',
        'reg_ssa': 'SSARegister',
        'reg': 'Register',
        'flag': 'Flag',
        'flag_ssa': 'SSAFlag',
        'reg_ssa_list': 'Vec<SSARegister>',
        'expr_list': 'Vec<HighLevelILInstruction>',
        'int_list': 'Vec<u64>',
        'target_map': 'HashMap<u64, u64>',
        'flag_ssa_list': 'Vec<SSAFlag>',
        'label': 'GotoLabel',
        'member_index': 'u64',
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
print('impl HighLevelILOperation {')

print('    pub fn from_instr(instr: BNHighLevelILInstruction, func: &HighLevelILFunction, expr_index: u64)')
print('            -> HighLevelILOperation {')

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

for (k, v) in binaryninja.highlevelil.HighLevelILInstruction.ILOperations.items():
    name = make_camel(str(k))

    # Translate the name into a more verbose name, or use the original if not found
    name = name_translation.get(name, name)

    print('            BN{} => {{'.format(str(k).replace(".", "_")))
    for (kind, typ) in v:
        print('                let {} = {}!();'.format(kind, typ))
    print('                HighLevelILOperation::{} {{'.format(name))
    print('                    {}'.format(', '.join([str(name) for (name, typ) in v])))
    print('                }')
    print('            }')

print('        }')
print('    }')


print('}')

for (k, v) in binaryninja.highlevelil.HighLevelILInstruction.ILOperations.items():
    name = make_camel(str(k))

    # Translate the name into a more verbose name, or use the original if not found
    name = name_translation.get(name, name)

    names = ['/* {} */ {}'.format(translation.get(x[1], x[1]), x[0]) for x in v]

    print('        HighLevelILOperation::{} {{ {} }}=> {{'.format(name, ', '.join(names)))
    print('            let mut res = Vec::new();')
    for (kind, typ) in v:
        if typ == 'expr':
            print('            if let Some(vars) = {}.ssa_vars() {{'.format(kind))
            print('                for var in vars {')
            print('                    res.push(var.clone());')
            print('                }')
            print('            }')
        elif typ == 'expr_list':
            print('            for instr in {} {{'.format(kind))
            print('                if let Some(vars) = instr.ssa_vars() {')
            print('                    for var in vars {')
            print('                        res.push(var.clone());')
            print('                    }')
            print('                }')
            print('            }')
        elif typ == 'var_ssa':
            print('            res.push({}.clone());'.format(kind))
        elif typ == 'var_ssa_list':
            print('            for var in {} {{'.format(kind))
            print('                res.push(var.clone());')
            print('            }')

    print('            res');
    print('        }');

