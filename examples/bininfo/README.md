# bininfo

Replicating the [bininfo](https://github.com/Vector35/binaryninja-api/tree/dev/examples/bin-info) example provided by Vector35 leveraging [binja-rs](https://github.com/ctfhacker/binja-rs)

## Example 

```
cargo run -- /bin/ls
```

```
Binary View: <BinaryView: "/bin/ls", start: 0x400000, len: 0x21f368>
Target: /bin/ls
TYPE: ELF
START: 0x400000
ENTRY: 0x4049a0
Function Count: 192
---------- 10 Functions ----------
0x4022b8: _init
0x402300: __uflow
0x402310: getenv
0x402320: sigprocmask
0x402330: raise
0x402340: free
0x402370: abort
0x402380: __errno_location
0x402390: strncmp
0x4023c0: strcpy
---------- 10 Strings ----------
Address [len]: Data
0x400238 [27]: /lib64/ld-linux-x86-64.so.2
0x401031 [15]: libselinux.so.1
0x401041 [27]: _ITM_deregisterTMCloneTable
0x40105d [14]: __gmon_start__
0x40106c [19]: _Jv_RegisterClasses
0x401080 [25]: _ITM_registerTMCloneTable
0x40109a [5]: _inity
0x4010a0 [11]: fgetfilecon
0x4010ac [7]: freecon
0x4010b4 [11]: lgetfilecon
```

## With introspection

Enable the `binja-rs/introspection` feature in `Cargo.toml` to enable the 
performance introspection

```
Calculated OS frequency: 3911980960
Total time: 6.76s    (26451425132 cycles)
                            TIMER | HITS | TIMES
Startup__InitPlugins              | 3    |     2027441169 cycles   7.66% |
BinaryView__NewFromFilename       | 1    |      764439205 cycles   2.89% | (99.19% with child timers)
BinaryView__Strings               | 1    |        4397314 cycles   0.02% |
BinaryView__UpdateAnalysisAndWait | 1    |    23437405691 cycles  88.61% |
BinaryView__EntryPoint            | 1    |           1068 cycles   0.00% |
BinaryView__Len                   | 1    |          84688 cycles   0.00% |
BinaryView__Start                 | 1    |            992 cycles   0.00% |
BinaryView__Name                  | 1    |          12497 cycles   0.00% |
BinaryView__TypeName              | 1    |           1654 cycles   0.00% |
BinaryView__Open                  | 1    |        8006742 cycles   0.03% |
BinaryView__AvailableViewTypes    | 1    |         242153 cycles   0.00% |
Remainder                         |      |      209391959 cycles   0.79%
```