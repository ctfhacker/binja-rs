//! Provides plugin commands listing and execution
#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(improper_ctypes)]

use core::*;

use anyhow::{Result, anyhow};

use crate::binjastr::BinjaStr;
use crate::binaryview::BinaryView;

pub struct Plugins;

impl Plugins {
    /// Get the list of the available plugin commands
    pub fn get_list() -> Vec<PluginCommand> {
        unsafe {
            let mut count = 0;

            let mut res = Vec::new();

            // Get the list of plugin commands
            unsafe { 
                let commands = BNGetAllPluginCommands(&mut count);
                let commands_slice = std::slice::from_raw_parts(commands, count as usize);

                // Parse each found plugin command
                for command in commands_slice {
                    res.push(PluginCommand::new(*command));
                }

                // Free the found plugin list
                BNFreePluginCommandList(commands);
            }

            res
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PluginCommandType {
    Default = 0,
    Address,
    Range,
    Function,
    LowLevelILFunction,
    LowLevelILInstruction,
    MediumLevelILFunction,
    MediumLevelILInstruction
}

impl PluginCommandType {
    pub fn from_plugincommandtype(type_: BNPluginCommandType) -> PluginCommandType {
        match type_ {
            BNPluginCommandType_DefaultPluginCommand =>
                PluginCommandType::Default,
            BNPluginCommandType_AddressPluginCommand =>
                PluginCommandType::Address,
            BNPluginCommandType_RangePluginCommand =>
                PluginCommandType::Range,
            BNPluginCommandType_FunctionPluginCommand =>
                PluginCommandType::Function,
            BNPluginCommandType_LowLevelILFunctionPluginCommand =>
                PluginCommandType::LowLevelILFunction,
            BNPluginCommandType_LowLevelILInstructionPluginCommand =>
                PluginCommandType::LowLevelILInstruction,
            BNPluginCommandType_MediumLevelILFunctionPluginCommand =>
                PluginCommandType::MediumLevelILFunction,
            BNPluginCommandType_MediumLevelILInstructionPluginCommand =>
                PluginCommandType::MediumLevelILFunction,
            _ => unreachable!()
        }
    }
}

pub struct PluginCommand {
    name: BinjaStr,
    description: BinjaStr,
    type_: PluginCommandType,
    bnplugincommand: BNPluginCommand
}

impl PluginCommand {
    pub fn new(command: BNPluginCommand) -> PluginCommand {
        PluginCommand {
            name: BinjaStr::new(command.name),
            description: BinjaStr::new(command.description),
            type_: PluginCommandType::from_plugincommandtype(command.type_),
            bnplugincommand: command.clone()
        }
    }

    pub fn name(&self) -> &str {
        self.name.as_str()
    }

    pub fn description(&self) -> &str {
        self.description.as_str()
    }

    pub fn command_type(&self) -> PluginCommandType {
        self.type_
    }

    pub fn execute(&self, bv: &BinaryView) -> Result<()> {
        match self.command_type() {
            PluginCommandType::Default => {
                // Get the function for the Default PluginCommandType
                if let Some(func) = self.bnplugincommand.defaultCommand {
                    unsafe { 
                        func(self.bnplugincommand.context, bv.handle());
                    }
                    return Ok(());
                }
            }
            _ => panic!("Unimplemented command type: {:?}", self.command_type())
        }

        Err(anyhow!("PluginCommand type {:?} doesn't have the func"))
    }
}


/*
pub name : * mut::std::os::raw::c_char , 
pub description : * mut::std::os::raw::c_char , 
pub type_ : BNPluginCommandType , 
pub context : * mut::std::os::raw::c_void , 
pub defaultCommand :::std::option::Option <unsafe extern "C" fn (ctxt : * mut::std::os::raw::c_void , view : * mut BNBinaryView) > , 
pub addressCommand :::std::option::Option < unsafe extern "C" fn (ctxt : * mut::std::os::raw::c_void , view : * mut BNBinaryView , addr : u64) > , 
pub rangeCommand :::std::option::Option < unsafe extern "C" fn (ctxt : * mut::std::os::raw::c_void , view : * mut BNBinaryView , addr : u64 , len : u64) > , 
pub functionCommand :::std::option::Option < unsafe extern "C" fn (ctxt : * mut::std::os::raw::c_void , view : * mut BNBinaryView , func : * mut BNFunction) > , 
pub lowLevelILFunctionCommand :::std::option::Option < unsafe extern "C" fn (ctxt : * mut::std::os::raw::c_void , view : * mut BNBinaryView , func : * mut BNLowLevelILFunction) > , 
pub lowLevelILInstructionCommand :::std::option::Option < unsafe extern "C" fn (ctxt : * mut::std::os::raw::c_void , view : * mut BNBinaryView , func : * mut BNLowLevelILFunction , instr : size_t) > , 
pub mediumLevelILFunctionCommand :::std::option::Option < unsafe extern "C" fn (ctxt : * mut::std::os::raw::c_void , view : * mut BNBinaryView , func : * mut BNMediumLevelILFunction) > , 
pub mediumLevelILInstructionCommand :::std::option::Option < unsafe extern "C" fn (ctxt : * mut::std::os::raw::c_void , view : * mut BNBinaryView , func : * mut BNMediumLevelILFunction , instr : size_t) > , 
pub defaultIsValid :::std::option::Option < unsafe extern "C" fn (ctxt : * mut::std::os::raw::c_void , view : * mut BNBinaryView) -> bool > , 
pub addressIsValid :::std::option::Option < unsafe extern "C" fn (ctxt : * mut::std::os::raw::c_void , view : * mut BNBinaryView , addr : u64) -> bool > , 
pub rangeIsValid :::std::option::Option < unsafe extern "C" fn (ctxt : * mut::std::os::raw::c_void , view : * mut BNBinaryView , addr : u64 , len : u64) -> bool > , 
pub functionIsValid :::std::option::Option < unsafe extern "C" fn (ctxt : * mut::std::os::raw::c_void , view : * mut BNBinaryView , func : * mut BNFunction) -> bool > , 
pub lowLevelILFunctionIsValid :::std::option::Option < unsafe extern "C" fn (ctxt : * mut::std::os::raw::c_void , view : * mut BNBinaryView , func : * mut BNLowLevelILFunction) -> bool > , 
pub lowLevelILInstructionIsValid :::std::option::Option < unsafe extern "C" fn (ctxt : * mut::std::os::raw::c_void , view : * mut BNBinaryView , func : * mut BNLowLevelILFunction , instr : size_t) -> bool > , 
pub mediumLevelILFunctionIsValid :::std::option::Option < unsafe extern "C" fn (ctxt : * mut::std::os::raw::c_void , view : * mut BNBinaryView , func : * mut BNMediumLevelILFunction) -> bool > , 
pub mediumLevelILInstructionIsValid :::std::option::Option < unsafe extern "C" fn (ctxt : * mut::std::os::raw::c_void , view : * mut BNBinaryView , func : * mut BNMediumLevelILFunction , instr : size_t) -> bool > , 
*/
