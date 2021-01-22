# Binary Ninja Rust Bindings

Work in progress Rust bindings geared towards analysis. Features added as they come up
when 

## Examples

[back_slice](./back_slice): Example implementation of a back slicing algorithm over SSA
[bininfo](./bininfo): Dump basic binary information for a given binary
[count_ins](./count_ins): Counts the number of LLIL, MLIL, HLIL calls for a given binary
[cyclomatic_complexity](./cyclomatic_complexity): Dumps a sample of most connected
functions and basic blocks for a given binary
[llil_parser](./llil_parser): Dumps LLIL instructions
[mlil_parser](./mlil_parser): Dumps MLIL instructions
[hlil_parser](./hlil_parser): Dumps HLIL instructions

## Building bindings

The first time the bindings are cloned, they don't exist. Need to pull in the API and 
build them by running the following:

```
cd binja-sys
git clone -b dev https://github.com/vector35/binaryninja-api
cargo build --release
```

## Debugging

If this error is shown:

### Linux

```
error while loading shared libraries: libbinaryninjacore.so.1: cannot open shared object file: No such file or directory
```

Add the binaryninjacore path to `LD_LIBRARY_PATH`:

```
export LD_LIBRARY_PATH=$HOME/binaryninja
```

### Windows

Ensure the path to `binaryninjacore.dll` is in your PATH.

## Issues

Sometimes there is an error when creating from the `BinaryViewType`. Not exactly sure why
this happens, but running the binary again usually fixes it. 

```
C:\Users\rando\workspace\binja-rs\examples\back_slice>target\release\back_slice.exe timeout.exe
Using plugin dir: C:\Program Files\Vector35\BinaryNinja\plugins
BNInitCorePlugins: true
thread 'main' panicked at 'called `Result::unwrap()` on an `Err` value: BNCreateBinaryViewOfType(self.handle, data.handle()) failed', src\main.rs:24:54
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
```
