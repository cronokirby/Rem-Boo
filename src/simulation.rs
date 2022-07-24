use crate::buffer::Buffer;
use crate::bytecode::{BinaryInstruction, Instruction, Location, Program};
use crate::number::Number;

/// Represents an interpreter, which can run operations.
///
/// We use this to abstract both over data types, and over operations.
/// Using basic operations, we can build up more complicated operations.
trait Interpreter {
    /// The type of immediate arguments
    type Immediate;

    /// Compute the and of the top two items on the stack.
    fn and(&mut self);

    /// Compute the and of the top of the stack and an immediate value.
    fn and_imm(&mut self, imm: Self::Immediate);

    /// Compute the xor of the top two items on the stack.
    fn xor(&mut self);

    /// Compute the xor of the top of the stack and an immediate value.
    fn xor_imm(&mut self, imm: Self::Immediate);

    /// Push some data onto the stack.
    fn push(&mut self, imm: Self::Immediate);

    /// Copy an element, counting from the top of the stack.
    fn copy_top(&mut self, index: u32);

    /// Assert that the top element is equal to some value, consuming it.
    fn assert_eq(&mut self, imm: Self::Immediate);
}

fn exec_binary<T: Number, I: Interpreter<Immediate = T>>(
    interpreter: &mut I,
    instruction: &BinaryInstruction,
    location: Location,
    public: &Buffer<T>,
) {
    match location {
        Location::Top => match instruction {
            BinaryInstruction::Xor => interpreter.xor(),
            BinaryInstruction::And => interpreter.and(),
        },
        Location::Public(i) => {
            let data = public.read(i).unwrap();
            match instruction {
                BinaryInstruction::Xor => interpreter.xor_imm(data),
                BinaryInstruction::And => interpreter.and_imm(data),
            }
        }
    }
}

fn exec_instruction<T: Number, I: Interpreter<Immediate = T>>(
    interpreter: &mut I,
    instruction: &Instruction,
    public: &Buffer<T>,
    private: &Buffer<T>,
) {
    match instruction {
        Instruction::Binary(instr, loc) => exec_binary(interpreter, instr, *loc, public),
        Instruction::AssertEq(loc) => match loc {
            Location::Top => {
                interpreter.xor();
                interpreter.assert_eq(T::zero());
            }
            Location::Public(i) => {
                let data = public.read(*i).unwrap();
                interpreter.assert_eq(data);
            }
        },
        Instruction::PushTop => interpreter.copy_top(0),
        Instruction::PushPrivate(i) => {
            let data = private.read(*i).unwrap();
            interpreter.push(data);
        }
    }
}

fn exec_program<T: Number, I: Interpreter<Immediate = T>>(
    interpreter: &mut I,
    program: &Program,
    public: &Buffer<T>,
    private: &Buffer<T>,
) {
    for instruction in &program.instructions {
        exec_instruction(interpreter, instruction, public, private)
    }
}
