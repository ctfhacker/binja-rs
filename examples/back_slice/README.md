# Back slicing example

Example of using HLIL SSA to perform a backslice of variables. 

This example attempts to find any variables that are used in any function call
that is added or subtracted by 8 anywhere in its backslice. The ending 
condition filter was just an arbitrary filter to show walking the HLIL 
Operation tree.

## Example

```
cargo run --release -- timeout.exe
```

```
Using plugin dir: C:\Program Files\Vector35\BinaryNinja\plugins
BNInitCorePlugins: true
Exprs: 32205

[1/7][0x1400036bf:sub_140003674] return HeapSize(rax_3#1, 0, arg1#0) __tailcall 
[2/7][0x140003432:sub_1400033ec] int32_t rax_3#1 = 0x140003674(*arg1#0 @ mem#2) @ mem#2 -> mem#3 
[3/7][0x140004bab:sub_140004b40] rax#3 = 0x1400033ec(rbx_1#1) @ mem#2 -> mem#3 
[4/7][0x140004b92:sub_140004b40] void* rbx_1#1 = arg1#0 + 0x10 
[5/7][0x140004cc8:sub_140004c9c] 0x140004b40(rax#2) @ mem#1 -> mem#2 
[6/7][0x140004cc1:sub_140004c9c] rax#2 = ϕ(rax#1, rax#3) 
[7/7][0x140004cb3:sub_140004c9c] rax#1 = *(arg1#0 + 8) @ mem#0 

[1/6][0x1400036bf:sub_140003674] return HeapSize(rax_3#1, 0, arg1#0) __tailcall 
[2/6][0x140003432:sub_1400033ec] int32_t rax_3#1 = 0x140003674(*arg1#0 @ mem#2) @ mem#2 -> mem#3 
[3/6][0x140004b87:sub_140004b40] rax#2 = 0x1400033ec(arg1#0 + 0x10) @ mem#0 -> mem#1 
[4/6][0x140004cc8:sub_140004c9c] 0x140004b40(rax#2) @ mem#1 -> mem#2 
[5/6][0x140004cc1:sub_140004c9c] rax#2 = ϕ(rax#1, rax#3) 
[6/6][0x140004cb3:sub_140004c9c] rax#1 = *(arg1#0 + 8) @ mem#0 
Time taken: 330.0938ms
```
