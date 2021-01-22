# Cyclomatic complexity

Dumps the 5 most connected functions and 5 most connected basic blocks for
the given binary


## Example:

```
cargo run --release -- <BINARY>
```

```
Using plugin dir: C:\Program Files\Vector35\BinaryNinja\plugins
BNInitCorePlugins: true
CONNECTED FUNCTIONS
7: Function { start: 0x1400015b0, symbol_type: "FunctionSymbol", symbol: sub_1400015b0, name: sub_1400015b0 }
3: Function { start: 0x140001230, symbol_type: "FunctionSymbol", symbol: sub_140001230, name: sub_140001230 }
3: Function { start: 0x1400019c0, symbol_type: "FunctionSymbol", symbol: sub_1400019c0, name: sub_1400019c0 }
3: Function { start: 0x140001b20, symbol_type: "FunctionSymbol", symbol: sub_140001b20, name: sub_140001b20 }
2: Function { start: 0x14000109c, symbol_type: "FunctionSymbol", symbol: sub_14000109c, name: sub_14000109c }
CONNECTED BASIC BLOCKS
6: <BasicBlock func: Some(sub_140001480), start: 0x1400014e9, len: 86>
5: <BasicBlock func: Some(sub_140001010), start: 0x140001058, len: 9>
5: <BasicBlock func: Some(sub_1400015b0), start: 0x140001708, len: 15>
5: <BasicBlock func: Some(sub_1400015b0), start: 0x14000173c, len: 8>
5: <BasicBlock func: Some(sub_1400018ac), start: 0x1400018f0, len: 2>
```
