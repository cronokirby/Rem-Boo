use bincode::{Decode, Encode};

/// Represents a location for an operand.
#[derive(Clone, Copy, Encode, Decode, Debug, PartialEq)]
pub enum Location {
    /// The top of the relevant stack.
    Top,
    /// An indexed value in the public data.
    Public(u32),
}

/// Represents a simple binary instruction.
#[derive(Clone, Encode, Decode, Debug, PartialEq)]
pub enum BinaryInstruction {
    /// xor both operands, bitwise.
    Xor,
    /// and both operands, bitwise
    And,
}

/// Represents an instruction in the program.
#[derive(Clone, Encode, Decode, Debug, PartialEq)]
pub enum Instruction {
    /// A binary instruction, between the top of the stack, and a location.
    ///
    /// The result gets pushed on top of the stack.
    Binary(BinaryInstruction, Location),
    /// Assert that the top of the stack is equal to some location.
    AssertEq(Location),
    /// Push the top element of the stack onto the stack, duplicating it.
    PushTop,
    /// Copy a private element onto the stack.
    PushPrivate(u32),
}

/// Represents a program in our bytecode.
#[derive(Clone, Encode, Decode, Debug, PartialEq)]
pub struct Program {
    /// The number of public input bytes.
    pub public_size: u32,
    /// The number of private input bytes.
    pub private_size: u32,
    /// The instructions composing the program.
    pub instructions: Vec<Instruction>,
}
