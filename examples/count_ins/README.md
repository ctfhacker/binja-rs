# Count instructions

Filters through LLIL, MLIL, and HLIL instructions for `Call` instructions and dumps
the first 10 found of each type

## Example:

```
Using plugin dir: /home/user/binaryninja/plugins
BNInitCorePlugins: true
Analysis took: 9.843183800
[3.5544378s] LLIL_CALL: 1117
[0x3768] call(rax) 
[0x3eb6] call(0x11fd0) 
[0x3ec7] call(0x3cc0) 
[0x3eda] call(0x3920) 
[0x3ee6] call(0x38e0) 
[0x3efc] call(0x16340) 
[0x4cf8] call(0x135b0) 
[0x3ff8] call(0x37a0) 
[0x3f73] call(0x135b0) 
[0x404d] call(0x37a0) 

[1.8502557s] MLIL_CALL: 1116
[0x3768] rax = __gmon_start__() 
[0x3eb6] 0x11fd0(rdi, arg3, arg4, arg5) 
[0x3ec7] 0x3cc0(6, 0x17c4a) 
[0x3eda] 0x3920("coreutils", 0x17d95) 
[0x3ee6] 0x38e0("coreutils") 
[0x3efc] 0x16340(0xcca0) 
[0x4cf8] 0x135b0(nullptr, 7) 
[0x3ff8] rax_4 = 0x37a0("QUOTING_STYLE") 
[0x3f73] 0x135b0(nullptr, 7) 
[0x404d] rax_7 = 0x37a0("COLUMNS") 

[5.2088434s] HLIL_CALL: 386
[0x3768] __gmon_start__() 
[0x3eb6] 0x11fd0(rdi, arg3, arg4, arg5) 
[0x3ec7] setlocale(6, 0x17c4a) 
[0x3eda] bindtextdomain("coreutils", 0x17d95) 
[0x3ee6] textdomain("coreutils") 
[0x3efc] 0x16340(0xcca0) 
[0x4cf8] 0x135b0(nullptr, 7) 
[0x3f5d] abort() 
[0x3f73] 0x135b0(nullptr, 7) 
[0x5322] 0x135b0(nullptr, 3) 
```
