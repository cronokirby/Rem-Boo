use std::{iter, mem};

use bincode::{config, encode_into_std_write, Decode, Encode};
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
struct Party<'a> {
    /// This party's share of the private input.
    private: &'a MultiBuffer,
    /// This holds the current state of the party's stack.
    stack: MultiBuffer,
}

impl<'a> Party<'a> {
    pub fn new(private: &'a MultiBuffer) -> Self {
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

struct Simulator<'a> {
    public: &'a MultiBuffer,
    /// For each party, we hold:
    /// - Their RNG
    /// - A queue for their and bits
    /// - The party itself
    /// - The outgoing messages
    parties: Vec<(PRNG, MultiQueue<'a>, Party<'a>, MultiBuffer)>,
    input_party: Party<'a>,
}

impl<'a> Simulator<'a> {
    pub fn new(
        public: &'a MultiBuffer,
        masked_input: &'a MultiBuffer,
        rngs: Vec<PRNG>,
        and_bits: &'a [MultiBuffer],
        masks: &'a [MultiBuffer],
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

    fn iter_parties(&mut self) -> impl Iterator<Item = &mut Party<'a>> {
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

    pub fn messages(self) -> Vec<MultiBuffer> {
        self.parties
            .into_iter()
            .map(|(_, _, _, messages)| messages)
            .collect()
    }
}

const COMMITMENT_SIZE: usize = 32;

/// Represents a commitment to some value.
///
/// This commitment can be compared for equality without worrying about constant-time.
#[derive(Clone, Copy, Default, Debug, Encode, Decode, PartialEq)]
struct Commitment([u8; COMMITMENT_SIZE]);

/// A key used to produce a commitment.
///
/// The key allows the commitment to be hiding.
#[derive(Debug, Clone, Encode, Decode, PartialEq)]
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
#[derive(Clone, Copy, Debug)]
enum State<'a> {
    WithoutAux(&'a Seed),
    WithAux(&'a Seed, &'a MultiBuffer),
}

impl<'a> State<'a> {
    fn commit(self, key: &CommitmentKey) -> Commitment {
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

#[derive(Clone, Copy, Debug, Encode, Decode)]
struct Hash([u8; blake3::OUT_LEN]);

impl From<blake3::Hash> for Hash {
    fn from(val: blake3::Hash) -> Self {
        Hash(val.into())
    }
}

#[derive(Clone, Debug, Encode, Decode)]
struct ResponseInstance {
    first_aux: Option<MultiBuffer>,
    party_seeds: Vec<Seed>,
    masked_input: MultiBuffer,
    messages: MultiBuffer,
    commitment: Commitment,
    message_hash_key: [u8; blake3::KEY_LEN],
}

#[derive(Clone, Debug, Encode, Decode)]
struct Response {
    excluded_root_seeds: Vec<Seed>,
    excluded_message_hashes: Vec<Hash>,
    instances: Vec<ResponseInstance>,
}

struct Prover {
    m: usize,
    n: usize,
    commitment: Hash,
    root_seeds: Vec<Seed>,
    message_hashes: Vec<Hash>,
    message_hash_keys: Vec<[u8; blake3::KEY_LEN]>,
    all_states: Vec<(MultiBuffer, Vec<Seed>)>,
    all_commitment_keys: Vec<Vec<CommitmentKey>>,
    all_commitments: Vec<Vec<Commitment>>,
    all_masked_inputs: Vec<MultiBuffer>,
    all_messages: Vec<Vec<MultiBuffer>>,
}

impl Prover {
    pub fn setup<R: RngCore + CryptoRng>(
        rng: &mut R,
        m: usize,
        n: usize,
        program: &Program,
        public: &MultiBuffer,
        private: &MultiBuffer,
    ) -> Result<Self> {
        // First, execute the program to get a trace of the and bits.
        let mut interpreter = Interpreter::new(public, private);
        interpreter.run(program)?;
        let trace = interpreter.trace();

        let mut root_seeds = Vec::with_capacity(m);
        for _ in 0..m {
            root_seeds.push(Seed::random(rng));
        }

        let mut commitment_hashes = Vec::with_capacity(m);
        let mut message_hashes = Vec::with_capacity(m);
        let mut message_hash_keys = Vec::with_capacity(m);
        let mut all_states = Vec::with_capacity(m);
        let mut all_commitment_keys = Vec::with_capacity(m);
        let mut all_commitments = Vec::with_capacity(m);
        let mut all_masked_inputs = Vec::with_capacity(m);
        let mut all_messages = Vec::with_capacity(m);

        for root_seed in &root_seeds {
            let mut party_seeds = Vec::with_capacity(n);
            let mut commitment_keys = Vec::with_capacity(n);
            {
                let mut prng = PRNG::seeded(root_seed);
                for _ in 0..n {
                    party_seeds.push(Seed::random(&mut prng));
                    commitment_keys.push(CommitmentKey::random(&mut prng));
                }
            }

            // Split party seeds into and seeds and prngs
            let mut and_seeds = Vec::with_capacity(n);
            let mut prngs = Vec::with_capacity(n);
            for seed in &party_seeds {
                let mut prng = PRNG::seeded(seed);
                and_seeds.push(Seed::random(&mut prng));
                prngs.push(PRNG::seeded(&Seed::random(&mut prng)));
            }

            // Now, generate the and bits. The first party doesn't get random bits.
            let mut and_bits = Vec::with_capacity(n);
            and_bits.push(trace.clone());
            for seed in &and_seeds[1..] {
                let mut prng = PRNG::seeded(seed);
                let aux = MultiBuffer::random(&mut prng, and_bits[0].len_u64());
                and_bits[0].xor(&aux);
                and_bits.push(aux);
            }

            // Generate the state commitments
            let mut commitments = Vec::with_capacity(n);
            for (i, ((party_seed, aux), commitment_key)) in party_seeds
                .iter()
                .zip(and_bits.iter())
                .zip(commitment_keys.iter())
                .enumerate()
            {
                let state = if i == 0 {
                    State::WithAux(party_seed, aux)
                } else {
                    State::WithoutAux(party_seed)
                };
                commitments.push(state.commit(commitment_key));
            }

            // Next we need to run the simulation.
            // First, prepare the masks for each party, and the masked input
            let mut masked_input = private.clone();
            let mut masks = Vec::with_capacity(n);
            for prng in &mut prngs {
                let mask = MultiBuffer::random(prng, masked_input.len_u64());
                masked_input.xor(&mask);
                masks.push(mask);
            }

            // Then, run the simulation
            let mut simulator = Simulator::new(public, &masked_input, prngs, &and_bits, &masks);
            if !simulator.run(program) {
                return Err(Error::BadProgram);
            }

            // Finally, extract out the messages, and hash everything
            let messages = simulator.messages();
            let commitment_hash: Hash = {
                let mut hasher = blake3::Hasher::new();
                encode_into_std_write(&commitments, &mut hasher, config::standard())
                    .expect("failed to call hash function");
                hasher.finalize().into()
            };
            let mut message_hash_key = [0u8; blake3::KEY_LEN];
            rng.fill_bytes(&mut message_hash_key);
            message_hash_keys.push(message_hash_key);
            let message_hash: Hash = {
                let mut hasher = blake3::Hasher::new_keyed(&message_hash_key);
                encode_into_std_write(&masked_input, &mut hasher, config::standard())
                    .expect("failed to call hash function");
                encode_into_std_write(&messages, &mut hasher, config::standard())
                    .expect("failed to call hash function");
                hasher.finalize().into()
            };

            commitment_hashes.push(commitment_hash);
            message_hashes.push(message_hash);
            all_states.push((mem::take(&mut and_bits[0]), party_seeds));
            all_commitment_keys.push(commitment_keys);
            all_commitments.push(commitments);
            all_masked_inputs.push(masked_input);
            all_messages.push(messages);
        }

        let hash0: Hash = {
            let mut hasher = blake3::Hasher::new();
            encode_into_std_write(&commitment_hashes, &mut hasher, config::standard())
                .expect("failed to call hash function");
            hasher.finalize().into()
        };
        let hash1: Hash = {
            let mut hasher = blake3::Hasher::new();
            encode_into_std_write(&message_hashes, &mut hasher, config::standard())
                .expect("failed to call hash function");
            hasher.finalize().into()
        };
        let commitment: Hash = {
            let mut hasher = blake3::Hasher::new();
            encode_into_std_write((hash0, hash1), &mut hasher, config::standard())
                .expect("failed to call hash function");
            hasher.finalize().into()
        };

        Ok(Prover {
            m,
            n,
            commitment,
            root_seeds,
            message_hashes,
            message_hash_keys,
            all_states,
            all_commitments,
            all_commitment_keys,
            all_masked_inputs,
            all_messages,
        })
    }

    pub fn commitment(&self) -> Hash {
        self.commitment
    }

    pub fn response(mut self, included: &[Option<usize>]) -> Response {
        debug_assert_eq!(included.len(), self.m);
        let included_count = included.iter().filter(|x| x.is_some()).count();
        let excluded_count = included.len() - included_count;

        let mut excluded_root_seeds = Vec::with_capacity(excluded_count);
        let mut excluded_message_hashes = Vec::with_capacity(excluded_count);
        included
            .iter()
            .zip(self.root_seeds.into_iter())
            .zip(self.message_hashes.into_iter())
            .filter(|((included, _), _)| included.is_some())
            .for_each(|((_, root_seed), message_hashes)| {
                excluded_root_seeds.push(root_seed);
                excluded_message_hashes.push(message_hashes);
            });

        let mut instances = Vec::with_capacity(included_count);
        for (i, j) in included
            .iter()
            .enumerate()
            .filter_map(|(i, j)| j.map(|j| (i, j)))
        {
            let (aux, mut states) = mem::take(&mut self.all_states[i]);
            let first_aux = if j == 0 { None } else { Some(aux) };
            states.remove(j);
            let party_seeds = states;
            let masked_input = mem::take(&mut self.all_masked_inputs[i]);
            let messages = mem::take(&mut self.all_messages[i][j]);
            let commitment = mem::take(&mut self.all_commitments[i][j]);
            let message_hash_key = mem::take(&mut self.message_hash_keys[i]);

            instances.push(ResponseInstance {
                first_aux,
                party_seeds,
                masked_input,
                messages,
                commitment,
                message_hash_key,
            })
        }

        Response {
            excluded_root_seeds,
            excluded_message_hashes,
            instances,
        }
    }
}
