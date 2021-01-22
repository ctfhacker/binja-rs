# MLIL Parser

Dumps the first 10 MLIL instructions from all MLIL instructions gathered

## Example

```
cargo run --release -- <BINARY>
```

```
Using plugin dir: C:\Program Files\Vector35\BinaryNinja\plugins
BNInitCorePlugins: true
MLIL instructions
r10 = arg7 
rbx = arg1 
if (r10 == 0) then 3 @ 0x140001087 else 4 @ 0x140001026 
return  
r11_1 = arg5 
rax_1 = arg2 
if (arg2 == 0) then 7 @ 0x140001054 else 9 @ 0x140001034 
[r10].d = 0 
goto 10 @ 0x140001058 
if (rax_1 != 1) then 10 @ 0x140001058 else 12 @ 0x140001036 
Took 40.9109ms

MLIL instructions gathered in parallel
r10 = arg7 
rbx = arg1 
if (r10 == 0) then 3 @ 0x140001087 else 4 @ 0x140001026 
return  
r11_1 = arg5 
rax_1 = arg2 
if (arg2 == 0) then 7 @ 0x140001054 else 9 @ 0x140001034 
[r10].d = 0 
goto 10 @ 0x140001058 
if (rax_1 != 1) then 10 @ 0x140001058 else 12 @ 0x140001036 
Took 28.8506ms
```
