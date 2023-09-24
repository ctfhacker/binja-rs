//! Provides logging capabilities from Binja Ninja core

use std::convert::TryInto;

/// Log level to report from Binja Ninja core
#[repr(u32)]
pub enum LogLevel {
    Debug = 0,
    Info = 1, 
    Warning = 2,
    Error = 3,
    Alert = 4,
}

/// Enable log to stdout
pub fn to_stdout(level: LogLevel) {
    unsafe { binja_sys::BNLogToStdout((level as u32).try_into().unwrap()); }
}

/// Enable log to stderr
pub fn to_stderr(level: LogLevel) {
    unsafe { binja_sys::BNLogToStderr((level as u32).try_into().unwrap()); }
}
