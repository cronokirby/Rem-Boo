use rand_core::RngCore;

use crate::{
    buffer::MultiBuffer,
    bytecode::{BinaryInstruction, Instruction, Location, Program},
    rng::PRNG,
};

enum Error {
    BadProgram,
}

type Result<T> = std::result::Result<T, Error>;

/// An Interpreter running programs, creating an execution trace.
///
/// This trace records values we need to store later for the proof.
/// In particular, we need to store the results of and operations.
struct Interpreter<'a> {
    public: &'a MultiBuffer,
    private: &'a MultiBuffer,
    trace: MultiBuffer,
    stack: MultiBuffer,
}

impl<'a> Interpreter<'a> {
    /// Create a new interpreter, with the public and private inputs.
    pub fn new(public: &'a MultiBuffer, private: &'a MultiBuffer) -> Self {
        Self {
            public,
            private,
            trace: MultiBuffer::new(),
            stack: MultiBuffer::new(),
        }
    }

    fn pop_u64(&mut self) -> Result<u64> {
        self.stack.pop_u64().ok_or(Error::BadProgram)
    }

    fn binary(&mut self, instr: &BinaryInstruction) -> Result<()> {
        let left = self.pop_u64()?;
        let right = self.pop_u64()?;
        match instr {
            BinaryInstruction::Xor => self.stack.push_u64(left ^ right),
            BinaryInstruction::And => {
                let out = left & right;
                self.trace.push_u64(out);
                self.stack.push_u64(out);
            }
        }
        Ok(())
    }

    fn instruction(&mut self, instr: &Instruction) -> Result<()> {
        match instr {
            Instruction::Binary(instr, location) => match location {
                Location::Public(_) => self.pop_u64().map(|_| ()),
                Location::Top => self.binary(instr),
            },
            Instruction::AssertEq(_) => self.pop_u64().map(|_| ()),
            Instruction::PushTop => {
                let top = self.pop_u64()?;
                self.stack.push_u64(top);
                Ok(())
            }
            Instruction::PushPrivate(i) => {
                let data = self.private.read_u64(*i).ok_or(Error::BadProgram)?;
                self.stack.push_u64(data);
                Ok(())
            }
        }
    }

    /// Run the interpreter, storing the trace.
    pub fn run(&mut self, program: &Program) -> Result<()> {
        for instr in &program.instructions {
            self.instruction(instr)?;
        }
        Ok(())
    }

    /// Consume the interpreter, extracting the trace.
    pub fn trace(self) -> MultiBuffer {
        self.trace
    }
}

/// Create the auxilary bits, given all of the and bits of the execution.
fn create_aux_bits(prngs: &mut [PRNG], and_trace: &MultiBuffer) -> MultiBuffer {
    let mut out = MultiBuffer::new();
    for &(mut x) in and_trace.iter_u64() {
        for prng in prngs.iter_mut() {
            x ^= prng.next_u64();
        }
        out.push_u64(x);
    }
    out
}
