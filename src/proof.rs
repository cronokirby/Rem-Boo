use std::iter;

use bincode::{config, encode_into_slice, encode_into_std_write, Decode, Encode};
use rand_core::{CryptoRng, RngCore};

use crate::{
    buffer::{MultiBuffer, MultiQueue},
    bytecode::{BinaryInstruction, Instruction, Location, Program},
    rng::{Seed, PRNG},
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
    /// This party's share of the private input.
    private: MultiBuffer,
    /// This holds the current state of the party's stack.
    stack: MultiBuffer,
}

impl Party {
    pub fn new(private: MultiBuffer) -> Self {
        Self {
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

    pub fn and64(&mut self, rng: &mut PRNG, and_bits: &mut MultiQueue, za: u64, zb: u64) -> u64 {
        let a = self.stack.pop_u64().unwrap();
        let b = self.stack.pop_u64().unwrap();
        let c = rng.next_u64();
        self.stack.push_u64(c);
        (za & b) ^ (zb & a) ^ and_bits.next_u64() ^ c
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

    pub fn pop64(&mut self) -> u64 {
        self.stack.pop_u64().unwrap()
    }

    pub fn push64(&mut self, x: u64) {
        self.stack.push_u64(x)
    }
}

struct Simulator {
    public: MultiBuffer,
    /// For each party, we hold:
    /// - Their RNG
    /// - A queue for their and bits
    /// - The party itself
    /// - The outgoing messages
    parties: Vec<(PRNG, MultiQueue, Party, MultiBuffer)>,
    input_party: Party,
}

impl Simulator {
    pub fn new(
        public: MultiBuffer,
        masked_input: MultiBuffer,
        rngs: Vec<PRNG>,
        and_bits: Vec<MultiBuffer>,
        masks: Vec<MultiBuffer>,
    ) -> Self {
        let mut parties = Vec::with_capacity(rngs.len());
        for ((rng, and_bits), masks) in rngs
            .into_iter()
            .zip(and_bits.into_iter())
            .zip(masks.into_iter())
        {
            parties.push((
                rng,
                MultiQueue::new(and_bits),
                Party::new(masks),
                MultiBuffer::new(),
            ));
        }
        let input_party = Party::new(masked_input);
        Self {
            public,
            parties,
            input_party,
        }
    }

    fn iter_parties(&mut self) -> impl Iterator<Item = &mut Party> {
        self.parties
            .iter_mut()
            .map(|(_, _, party, _)| party)
            .chain(iter::once(&mut self.input_party))
    }

    fn pub64(&mut self, i: u32) -> u64 {
        self.public.read_u64(i).unwrap()
    }

    fn push_top64(&mut self) {
        for party in self.iter_parties() {
            party.push_top64();
        }
    }

    fn push_priv64(&mut self, i: u32) {
        for party in self.iter_parties() {
            party.push_priv64(i);
        }
    }

    fn assert_eq64(&mut self, loc: Location) -> bool {
        // Strategy: xor with the location, pull the value, assert zero
        self.xor64(loc);
        let mut output = self.input_party.pop64();
        for (_, _, party, messages) in self.parties.iter_mut() {
            let mask = party.pop64();
            messages.push_u64(mask);
            output ^= mask;
        }
        output == 0
    }

    fn xor64(&mut self, loc: Location) {
        match loc {
            Location::Top => self.iter_parties().for_each(|party| party.xor64()),
            Location::Public(i) => {
                let imm = self.pub64(i);
                self.input_party.xor64imm(imm);
            }
        };
    }

    fn and64(&mut self, loc: Location) {
        match loc {
            // For public values, the parties can compute locally
            Location::Public(i) => {
                let imm = self.pub64(i);
                self.iter_parties().for_each(|party| party.and64imm(imm));
            }
            // For private values, we need interaction
            Location::Top => {
                let za = self.input_party.pop64();
                let zb = self.input_party.pop64();
                let mut zc = za & zb;
                for (rng, and_bits, party, messages) in self.parties.iter_mut() {
                    let s_share = party.and64(rng, and_bits, za, zb);
                    messages.push_u64(s_share);
                    zc ^= s_share;
                }
                self.input_party.push64(zc);
            }
        }
    }

    fn instruction(&mut self, instr: &Instruction) -> bool {
        match instr {
            Instruction::Binary(instr, loc) => match instr {
                BinaryInstruction::Xor => self.xor64(*loc),
                BinaryInstruction::And => self.and64(*loc),
            },
            Instruction::AssertEq(loc) => return self.assert_eq64(*loc),
            Instruction::PushTop => self.push_top64(),
            Instruction::PushPrivate(i) => self.push_priv64(*i),
        };
        true
    }

    pub fn run(&mut self, program: &Program) -> bool {
        for instr in &program.instructions {
            if !self.instruction(instr) {
                return false;
            }
        }
        true
    }
}

const COMMITMENT_SIZE: usize = 32;

/// Represents a commitment to some value.
///
/// This commitment can be compared for equality without worrying about constant-time.
#[derive(Clone, Copy, Debug, Encode, Decode, PartialEq)]
struct Commitment([u8; COMMITMENT_SIZE]);

/// A key used to produce a commitment.
///
/// The key allows the commitment to be hiding.
#[derive(Debug, Clone, PartialEq)]
pub struct CommitmentKey([u8; blake3::KEY_LEN]);

impl CommitmentKey {
    /// Generate a random CommitmentKey.
    pub fn random<R: RngCore + CryptoRng>(rng: &mut R) -> Self {
        let mut bytes = [0u8; blake3::KEY_LEN];
        rng.fill_bytes(&mut bytes[..]);
        Self(bytes)
    }
}

/// Represents the state of a party that we commit to.
/// 
/// This contains a seed for their randomness, along with
enum State {
    WithoutAux(Seed),
    WithAux(Seed, MultiBuffer),
}

impl State {
    fn commit(&self, key: &CommitmentKey) -> Commitment {
        let mut hasher = blake3::Hasher::new_keyed(&key.0);
        let (seed, maybe_buffer) = match self {
            State::WithoutAux(seed) => (seed, None),
            State::WithAux(seed, buffer) => (seed, Some(buffer)),
        };
        // The encoding will be SEED_LEN || SEED || BUFFER_ENCODING.
        // Encoding the length of the seed isn't strictly necessary, but doesn't hurt.
        // In theory, we don't need to because it's a fixed size.
        encode_into_std_write(seed, &mut hasher, config::standard())
            .expect("failed to write value in commitment");
        if let Some(buffer) = maybe_buffer {
            encode_into_std_write(buffer, &mut hasher, config::standard())
                .expect("failed to write value in commitment");
        };
        Commitment(*hasher.finalize().as_bytes())
    }
}
