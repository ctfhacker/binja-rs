//! Provides logging capabilities from Binja Ninja core

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
    unsafe { core::BNLogToStdout(level as u32); }
}

/// Enable log to stderr
pub fn to_stderr(level: LogLevel) {
    unsafe { core::BNLogToStderr(level as u32); }
}
