# Binary Ninja Rust Bindings

## Building bindings

The first time the bindings are cloned, they don't exist. Build them by running the following in the `binja-rs` directory:

```
cargo build --features binja-sys/build_time_bindings
```

## Debugging

If this error is shown:

```
error while loading shared libraries: libbinaryninjacore.so.1: cannot open shared object file: No such file or directory
```

Add the binaryninjacore path to `LD_LIBRARY_PATH`:

```
export LD_LIBRARY_PATH=$HOME/binaryninja
```
