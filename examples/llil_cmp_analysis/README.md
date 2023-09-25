# LLIL Parser

Dumps the first 10 LLIL instructions from all LLIL instructions gathered

## Example

```
cargo run --release -- <BINARY>
```

```
Using plugin dir: C:\Program Files\Vector35\BinaryNinja\plugins
BNInitCorePlugins: true

LLIL instructions
push(rbx) 
rsp = rsp - 0x40 
r10 = [rsp + 0x80 {arg7}].q 
rbx = rcx 
if (r10 == 0) then 5 @ 0x140001082 else 8 @ 0x140001026 
rsp = rsp + 0x40 
rbx = pop 
<return> jump(pop) 
r11 = [rsp + 0x70 {arg5}].q 
eax = edx 
Took 15.4241ms

LLIL instructions gathered in parallel
push(rbx) 
rsp = rsp - 0x40 
r10 = [rsp + 0x80 {arg7}].q 
rbx = rcx 
if (r10 == 0) then 5 @ 0x140001082 else 8 @ 0x140001026 
rsp = rsp + 0x40 
rbx = pop 
<return> jump(pop) 
r11 = [rsp + 0x70 {arg5}].q 
eax = edx 
Took 5.9513ms
```
