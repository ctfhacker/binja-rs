# HLIL Parser

Dumps the first 10 HLIL instructions from all HLIL instructions gathered

## Example

```
cargo run --release -- <BINARY>
```

```
Using plugin dir: C:\Program Files\Vector35\BinaryNinja\plugins
BNInitCorePlugins: true
HLIL instructions
if (arg7 != 0)
    if (arg2 == 0)
        *arg7 = 0
    else if (arg2 == 1)
        int32_t rax_2 = 0x100
        int32_t rcx_1 = zx.d(arg3) + 1
        *(arg7 + 0x10) = arg4
        *(arg7 + 0x18) = arg5
        if (arg3 != 0)
            rax_2 = rcx_1
        *arg7 = rax_2
    if (*(arg7 + 0x28) != 0)
        *(arg7 + 0x30)
        0x140001b20(arg1) 
if (arg2 == 0)
    *arg7 = 0
else if (arg2 == 1)
    int32_t rax_2 = 0x100
    int32_t rcx_1 = zx.d(arg3) + 1
    *(arg7 + 0x10) = arg4
    *(arg7 + 0x18) = arg5
    if (arg3 != 0)
        rax_2 = rcx_1
    *arg7 = rax_2 
*arg7 = 0 
if (arg2 == 1)
    int32_t rax_2 = 0x100
    int32_t rcx_1 = zx.d(arg3) + 1
    *(arg7 + 0x10) = arg4
    *(arg7 + 0x18) = arg5
    if (arg3 != 0)
        rax_2 = rcx_1
    *arg7 = rax_2 
int32_t rax_2 = 0x100 
int32_t rcx_1 = zx.d(arg3) + 1 
*(arg7 + 0x10) = arg4 
*(arg7 + 0x18) = arg5 
if (arg3 != 0)
    rax_2 = rcx_1 
rax_2 = rcx_1 
Took 32.5364ms

HLIL instructions gathered in parallel
if (arg7 != 0)
    if (arg2 == 0)
        *arg7 = 0
    else if (arg2 == 1)
        int32_t rax_2 = 0x100
        int32_t rcx_1 = zx.d(arg3) + 1
        *(arg7 + 0x10) = arg4
        *(arg7 + 0x18) = arg5
        if (arg3 != 0)
            rax_2 = rcx_1
        *arg7 = rax_2
    if (*(arg7 + 0x28) != 0)
        *(arg7 + 0x30)
        0x140001b20(arg1) 
if (arg2 == 0)
    *arg7 = 0
else if (arg2 == 1)
    int32_t rax_2 = 0x100
    int32_t rcx_1 = zx.d(arg3) + 1
    *(arg7 + 0x10) = arg4
    *(arg7 + 0x18) = arg5
    if (arg3 != 0)
        rax_2 = rcx_1
    *arg7 = rax_2 
*arg7 = 0 
if (arg2 == 1)
    int32_t rax_2 = 0x100
    int32_t rcx_1 = zx.d(arg3) + 1
    *(arg7 + 0x10) = arg4
    *(arg7 + 0x18) = arg5
    if (arg3 != 0)
        rax_2 = rcx_1
    *arg7 = rax_2 
int32_t rax_2 = 0x100 
int32_t rcx_1 = zx.d(arg3) + 1 
*(arg7 + 0x10) = arg4 
*(arg7 + 0x18) = arg5 
if (arg3 != 0)
    rax_2 = rcx_1 
rax_2 = rcx_1 
Took 23.644ms
```
