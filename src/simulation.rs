use std::iter;

use crate::buffer::{Buffer, BufferQueue, Queue};
use crate::bytecode::{BinaryInstruction, Instruction, Location, Program};
use crate::number::Number;
use crate::rng::Prng;

/// Represents an interpreter, which can run operations.
///
/// We use this to abstract both over data types, and over operations.
/// Using basic operations, we can build up more complicated operations.
pub trait Interpreter {
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

    /// Read and push some private data onto the stack.
    fn push_private(&mut self, index: u32);

    /// Copy an element, counting from the top of the stack.
    fn copy_bottom(&mut self, index: u32);

    /// Copy an element, counting from the top of the stack.
    fn copy_top(&mut self, index: u32);

    /// Assert that the top element is equal to some value, consuming it.
    fn assert_eq(&mut self, imm: Self::Immediate) -> bool;
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
) -> Option<()> {
    match instruction {
        Instruction::Binary(instr, loc) => exec_binary(interpreter, instr, *loc, public),
        Instruction::AssertEq(loc) => match loc {
            Location::Top => {
                interpreter.xor();
                if !interpreter.assert_eq(T::zero()) {
                    return None;
                }
            }
            Location::Public(i) => {
                let data = public.read(*i).unwrap();
                interpreter.assert_eq(data);
            }
        },
        Instruction::CopyBottom(i) => interpreter.copy_top(*i),
        Instruction::CopyTop(i) => interpreter.copy_top(*i),
        Instruction::PushPrivate(i) => interpreter.push_private(*i),
    };
    Some(())
}

#[must_use]
pub fn exec_program<T: Number, I: Interpreter<Immediate = T>>(
    interpreter: &mut I,
    program: &Program,
    public: &Buffer<T>,
) -> Option<()> {
    for instruction in &program.instructions {
        exec_instruction(interpreter, instruction, public)?;
    }
    Some(())
}

pub struct Tracer<'a, T> {
    private: &'a Buffer<T>,
    trace: Buffer<T>,
    stack: Buffer<T>,
}

impl<'a, T> Tracer<'a, T> {
    pub fn new(private: &'a Buffer<T>) -> Self {
        Self {
            private,
            trace: Buffer::new(),
            stack: Buffer::new(),
        }
    }

    pub fn trace(self) -> Buffer<T> {
        self.trace
    }
}

impl<'a, T: Number> Interpreter for Tracer<'a, T> {
    type Immediate = T;

    fn and(&mut self) {
        let a = self.stack.pop().unwrap();
        let b = self.stack.pop().unwrap();
        let out = a & b;
        self.stack.push(out);
        self.trace.push(out);
    }

    fn and_imm(&mut self, imm: Self::Immediate) {
        let a = self.stack.pop().unwrap();
        self.stack.push(a & imm);
    }

    fn xor(&mut self) {
        let a = self.stack.pop().unwrap();
        let b = self.stack.pop().unwrap();
        self.stack.push(a ^ b);
    }

    fn xor_imm(&mut self, imm: Self::Immediate) {
        let a = self.stack.pop().unwrap();
        self.stack.push(a ^ imm);
    }

    fn push_private(&mut self, index: u32) {
        let data = self.private.read(index).unwrap();
        self.stack.push(data);
    }

    fn copy_top(&mut self, index: u32) {
        self.copy_bottom(self.stack.len() as u32 - 1 - index);
    }

    fn copy_bottom(&mut self, index: u32) {
        let data = self
            .stack
            .read(self.stack.len() as u32 - 1 - index)
            .unwrap();
        self.stack.push(data);
    }

    fn assert_eq(&mut self, imm: Self::Immediate) -> bool {
        let data = self.stack.pop().unwrap();
        data == imm
    }
}

struct Party<'a, T> {
    private: &'a Buffer<T>,
    stack: Buffer<T>,
}

impl<'a, T> Party<'a, T> {
    fn new(private: &'a Buffer<T>) -> Self {
        Self {
            private,
            stack: Buffer::new(),
        }
    }
}

impl<'a, T: Number> Party<'a, T> {
    fn xor(&mut self) {
        let a = self.stack.pop().unwrap();
        self.xor_imm(a);
    }

    fn xor_imm(&mut self, imm: T) {
        let b = self.stack.pop().unwrap();
        self.stack.push(imm ^ b);
    }

    fn and_imm(&mut self, imm: T) {
        let b = self.stack.pop().unwrap();
        self.stack.push(imm & b);
    }

    fn and<Q: Queue<T>>(&mut self, rng: &mut Prng, and_bits: &mut Q, za: T, zb: T) -> T {
        let a = self.stack.pop().unwrap();
        let b = self.stack.pop().unwrap();
        let c = T::random(rng);
        self.stack.push(c);
        (za & b) ^ (zb & a) ^ and_bits.next() ^ c
    }

    fn push(&mut self, imm: T) {
        self.stack.push(imm);
    }

    fn pop(&mut self) -> T {
        self.stack.pop().unwrap()
    }

    fn push_private(&mut self, index: u32) {
        let data = self.private.read(index).unwrap();
        self.stack.push(data);
    }

    fn copy_top(&mut self, index: u32) {
        self.copy_bottom(self.stack.len() as u32 - 1 - index);
    }

    fn copy_bottom(&mut self, index: u32) {
        let data = self
            .stack
            .read(self.stack.len() as u32 - 1 - index)
            .unwrap();
        self.stack.push(data);
    }
}

type PartyState<'a, T> = (Prng, BufferQueue<'a, T>, Party<'a, T>, Buffer<T>);

pub struct Simulator<'a, Q, T> {
    parties: Vec<PartyState<'a, T>>,
    input_party: Party<'a, T>,
    extra_messages: Q,
}

impl<'a, T, Q> Simulator<'a, Q, T> {
    pub fn new(
        masked_input: &'a Buffer<T>,
        rngs: Vec<Prng>,
        and_bits: &'a [Buffer<T>],
        masks: &'a [Buffer<T>],
        extra_messages: Q,
    ) -> Self {
        let mut parties = Vec::with_capacity(rngs.len());
        for ((rng, and_bits), masks) in rngs.into_iter().zip(and_bits.iter()).zip(masks.iter()) {
            parties.push((
                rng,
                BufferQueue::new(and_bits),
                Party::new(masks),
                Buffer::new(),
            ));
        }
        let input_party = Party::new(masked_input);
        Self {
            parties,
            input_party,
            extra_messages,
        }
    }

    fn iter_parties(&mut self) -> impl Iterator<Item = &mut Party<'a, T>> {
        self.parties
            .iter_mut()
            .map(|(_, _, party, _)| party)
            .chain(iter::once(&mut self.input_party))
    }

    pub fn messages(self) -> Vec<Buffer<T>> {
        self.parties
            .into_iter()
            .map(|(_, _, _, messages)| messages)
            .collect()
    }
}

impl<'a, T: Number, Q: Queue<T>> Interpreter for Simulator<'a, Q, T> {
    type Immediate = T;

    fn and(&mut self) {
        let za = self.input_party.pop();
        let zb = self.input_party.pop();
        let mut zc = za & zb;
        for (rng, and_bits, party, messages) in self.parties.iter_mut() {
            let s_share = party.and(rng, and_bits, za, zb);
            messages.push(s_share);
            zc ^= s_share;
        }
        zc ^= self.extra_messages.next();
        self.input_party.push(zc);
    }

    fn and_imm(&mut self, imm: Self::Immediate) {
        for party in self.iter_parties() {
            party.and_imm(imm);
        }
    }

    fn xor(&mut self) {
        for party in self.iter_parties() {
            party.xor();
        }
    }

    fn xor_imm(&mut self, imm: Self::Immediate) {
        self.input_party.xor_imm(imm);
    }

    fn push_private(&mut self, index: u32) {
        for party in self.iter_parties() {
            party.push_private(index);
        }
    }

    fn copy_top(&mut self, index: u32) {
        for party in self.iter_parties() {
            party.copy_top(index);
        }
    }

    fn copy_bottom(&mut self, index: u32) {
        for party in self.iter_parties() {
            party.copy_bottom(index);
        }
    }

    fn assert_eq(&mut self, imm: Self::Immediate) -> bool {
        // Strategy: xor with immediate, pull the value, assert zero
        self.xor_imm(imm);
        let mut output = self.input_party.pop();
        for (_, _, party, messages) in self.parties.iter_mut() {
            let mask = party.pop();
            messages.push(mask);
            output ^= mask;
        }
        output ^= self.extra_messages.next();
        output == T::zero()
    }
}
