use std::sync::Mutex;

use rand_core::RngCore;

use crate::{
    buffer::{MultiBuffer, MultiQueue},
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

/// Create the and bits for each party.
///
/// Every party, except one, will sample these bits randomly.
/// The other parties
fn create_and_bits(prngs: &mut [PRNG], and_trace: &MultiBuffer) -> Vec<MultiBuffer> {
    let mut out = vec![MultiBuffer::new(); prngs.len() + 1];
    for &(mut x) in and_trace.iter_u64() {
        for (i, prng) in prngs.iter_mut().enumerate() {
            let y = prng.next_u64();
            out[i + 1].push_u64(x);
            x ^= y;
        }
        out[0].push_u64(x);
    }
    out
}

/// Represents a single party in the simulated multi-party computation.
///
/// This party can run a lot of the computation itself, but sometimes needs
/// additional information from the outside. We drive the execution by
/// calling specific methods on the party.
struct Party {
    /// The rng used to generate new bits for and results.
    rng: PRNG,
    /// This party's sharing of the bits used for calculating ands.
    and_bits: MultiQueue,
    /// This party's share of the private input.
    private: MultiBuffer,
    /// This holds the current state of the party's stack.
    stack: MultiBuffer,
}

impl Party {
    pub fn new(rng: PRNG, and_bits: MultiBuffer, private: MultiBuffer) -> Self {
        Self {
            rng,
            and_bits: MultiQueue::new(and_bits),
            private,
            stack: MultiBuffer::new(),
        }
    }

    pub fn xor64(&mut self) {
        let a = self.stack.pop_u64().unwrap();
        self.xor64imm(a)
    }

    pub fn xor64imm(&mut self, imm: u64) {
        let b = self.stack.pop_u64().unwrap();
        self.stack.push_u64(imm ^ b)
    }

    pub fn and64imm(&mut self, imm: u64) {
        let b = self.stack.pop_u64().unwrap();
        self.stack.push_u64(imm & b);
    }

    pub fn push_top64(&mut self) {
        let top = self.stack.pop_u64().unwrap();
        self.stack.push_u64(top);
    }

    pub fn push_priv64(&mut self, i: u32) {
        let x = self.private.read_u64(i).unwrap();
        self.stack.push_u64(x);
    }

    pub fn top64(&self) -> u64 {
        self.stack.top_u64().unwrap()
    }
}
