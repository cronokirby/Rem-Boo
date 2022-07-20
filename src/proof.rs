use std::fmt::Debug;
use std::{iter, mem};

use bincode::{config, encode_into_std_write, Decode, Encode};
use rand_core::{CryptoRng, RngCore};

use crate::{
    buffer::{MultiBuffer, MultiQueue, NullQueue, Queue},
    bytecode::{BinaryInstruction, Instruction, Location, Program},
    constants,
    rng::{random_selections, Seed, PRNG},
};

#[derive(Debug)]
pub enum Error {
    BadProgram,
}

pub type Result<T> = std::result::Result<T, Error>;

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

    fn and64(&mut self, loc: Location) -> Result<()> {
        match loc {
            Location::Top => {
                let left = self.pop_u64()?;
                let right = self.pop_u64()?;
                let out = left & right;
                self.stack.push_u64(out);
                self.trace.push_u64(out);
                Ok(())
            }
            Location::Public(i) => {
                let a = self.public.read_u64(i).ok_or(Error::BadProgram)?;
                let b = self.pop_u64()?;
                self.stack.push_u64(a & b);
                Ok(())
            }
        }
    }

    fn xor64(&mut self, loc: Location) -> Result<()> {
        match loc {
            Location::Top => {
                let left = self.pop_u64()?;
                let right = self.pop_u64()?;
                self.stack.push_u64(left ^ right);
                Ok(())
            }
            Location::Public(_) => Ok(()),
        }
    }

    fn instruction(&mut self, instr: &Instruction) -> Result<()> {
        match instr {
            Instruction::Binary(instr, location) => match instr {
                BinaryInstruction::Xor => self.xor64(*location),
                BinaryInstruction::And => self.and64(*location),
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
#[derive(Debug)]
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

#[derive(Debug)]
struct Simulator<'a, Q> {
    public: &'a MultiBuffer,
    /// For each party, we hold:
    /// - Their RNG
    /// - A queue for their and bits
    /// - The party itself
    /// - The outgoing messages
    parties: Vec<(PRNG, MultiQueue<'a>, Party<'a>, MultiBuffer)>,
    input_party: Party<'a>,
    extra_messages: Q,
}

impl<'a, Q: Queue + Debug> Simulator<'a, Q> {
    pub fn new(
        public: &'a MultiBuffer,
        masked_input: &'a MultiBuffer,
        rngs: Vec<PRNG>,
        and_bits: &'a [MultiBuffer],
        masks: &'a [MultiBuffer],
        extra_messages: Q,
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
            extra_messages,
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
        output ^= self.extra_messages.next_u64();
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
                zc ^= self.extra_messages.next_u64();
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

#[derive(Clone, Copy, Default, Debug, Encode, Decode, PartialEq)]
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
    commitment_keys: Vec<CommitmentKey>,
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

            // Next, we want to setup the execution trace.
            // First, we need to extract out the input masks:
            let mut masks = Vec::with_capacity(n);
            for prng in &mut prngs {
                let mask = MultiBuffer::random(prng, private.len_u64());
                masks.push(mask);
            }
            let mut global_mask = masks[0].clone();
            for mask in &masks[1..] {
                global_mask.xor(mask);
            }

            // Second, execute the program to get a trace of the and bits.
            let mut interpreter = Interpreter::new(public, &global_mask);
            interpreter.run(program)?;
            let trace = interpreter.trace();

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
            // First, prepare the masked input.
            let mut masked_input = private.clone();
            masked_input.xor(&global_mask);

            // Then, run the simulation
            let mut simulator =
                Simulator::new(public, &masked_input, prngs, &and_bits, &masks, NullQueue);
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
            .filter(|((included, _), _)| included.is_none())
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
            let mut commitment_keys = mem::take(&mut self.all_commitment_keys[i]);
            commitment_keys.remove(j);
            let party_seeds = states;
            let masked_input = mem::take(&mut self.all_masked_inputs[i]);
            let messages = mem::take(&mut self.all_messages[i][j]);
            let commitment = mem::take(&mut self.all_commitments[i][j]);
            let message_hash_key = mem::take(&mut self.message_hash_keys[i]);

            instances.push(ResponseInstance {
                first_aux,
                party_seeds,
                commitment_keys,
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

#[derive(Debug)]
pub struct Proof {
    commitment: Hash,
    response: Response,
}

pub fn prove<R: RngCore + CryptoRng>(
    rng: &mut R,
    ctx: &[u8],
    program: &Program,
    public: &MultiBuffer,
    private: &MultiBuffer,
) -> Result<Proof> {
    let prover = Prover::setup(
        rng,
        constants::FULL_SET_COUNT,
        constants::PARTY_COUNT,
        program,
        public,
        private,
    )?;

    let commitment = prover.commitment();

    let mut hasher = blake3::Hasher::new_derive_key(constants::CHALLENGE_CONTEXT);
    encode_into_std_write(program, &mut hasher, config::standard()).unwrap();
    encode_into_std_write(public, &mut hasher, config::standard()).unwrap();
    encode_into_std_write(commitment, &mut hasher, config::standard()).unwrap();
    encode_into_std_write(ctx, &mut hasher, config::standard()).unwrap();

    let mut prng = PRNG::from_hasher(hasher);
    let challenge = random_selections(
        &mut prng,
        constants::FULL_SET_COUNT,
        constants::SUBSET_COUNT,
        constants::PARTY_COUNT,
    );

    let response = prover.response(&challenge);

    Ok(Proof {
        commitment,
        response,
    })
}

pub fn verify(ctx: &[u8], program: &Program, public: &MultiBuffer, proof: &Proof) -> bool {
    let mut hasher = blake3::Hasher::new_derive_key(constants::CHALLENGE_CONTEXT);
    encode_into_std_write(program, &mut hasher, config::standard()).unwrap();
    encode_into_std_write(public, &mut hasher, config::standard()).unwrap();
    encode_into_std_write(proof.commitment, &mut hasher, config::standard()).unwrap();
    encode_into_std_write(ctx, &mut hasher, config::standard()).unwrap();
    let mut prng = PRNG::from_hasher(hasher);
    let challenge = random_selections(
        &mut prng,
        constants::FULL_SET_COUNT,
        constants::SUBSET_COUNT,
        constants::PARTY_COUNT,
    );
    let n = constants::PARTY_COUNT;

    let mut and_size = 0;

    // First calculate the commitment hashes
    // First, do those for the excluded items
    let mut commitment_hashes = vec![Hash::default(); constants::FULL_SET_COUNT];
    for (hash, root_seed) in challenge
        .iter()
        .zip(commitment_hashes.iter_mut())
        .filter_map(|(c, x)| if c.is_none() { Some(x) } else { None })
        .zip(&proof.response.excluded_root_seeds)
    {
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

        // Next, we want to setup the execution trace.
        // First, we need to extract out the input masks:
        let mut masks = Vec::with_capacity(n);
        for prng in &mut prngs {
            let mask = MultiBuffer::random(prng, program.private_size as usize);
            masks.push(mask);
        }
        let mut global_mask = masks[0].clone();
        for mask in &masks[1..] {
            global_mask.xor(mask);
        }

        // Second, execute the program to get a trace of the and bits.
        let mut interpreter = Interpreter::new(public, &global_mask);
        if interpreter.run(program).is_err() {
            println!("interpreter failed (0)");
            return false;
        }
        let trace = interpreter.trace();
        and_size = trace.len_u64();

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

        *hash = {
            let mut hasher = blake3::Hasher::new();
            encode_into_std_write(&commitments, &mut hasher, config::standard())
                .expect("failed to call hash function");
            hasher.finalize().into()
        };
    }

    // Next, do the included items
    for ((j, hash), instance) in challenge
        .iter()
        .zip(commitment_hashes.iter_mut())
        .filter_map(|(c, x)| c.map(|c| (c, x)))
        .zip(&proof.response.instances)
    {
        let mut commitments = vec![Commitment::default(); n];

        for (i, commitment) in commitments.iter_mut().enumerate() {
            if i == j {
                *commitment = instance.commitment;
            } else {
                let index = if i < j { i } else { i - 1 };
                let state = if i == 0 {
                    if instance.first_aux.is_none() {
                        println!("first aux is none");
                        return false;
                    }
                    State::WithAux(
                        &instance.party_seeds[index],
                        instance.first_aux.as_ref().unwrap(),
                    )
                } else {
                    State::WithoutAux(&instance.party_seeds[index])
                };
                *commitment = state.commit(&instance.commitment_keys[index]);
            }
        }

        *hash = {
            let mut hasher = blake3::Hasher::new();
            encode_into_std_write(&commitments, &mut hasher, config::standard())
                .expect("failed to call hash function");
            hasher.finalize().into()
        };
    }

    // Next, do the message hashes
    let mut message_hashes = vec![Hash::default(); constants::FULL_SET_COUNT];
    // First, copy in the excluded hashes
    for (out_hash, in_hash) in challenge
        .iter()
        .zip(message_hashes.iter_mut())
        .filter_map(|(c, m)| if c.is_none() { Some(m) } else { None })
        .zip(&proof.response.excluded_message_hashes)
    {
        *out_hash = *in_hash;
    }

    // Finally, reconstruct the messages for the included parts
    for ((j, hash), instance) in challenge
        .iter()
        .zip(message_hashes.iter_mut())
        .filter_map(|(c, m)| c.map(|j| (j, m)))
        .zip(&proof.response.instances)
    {
        // Split party seeds into and seeds and prngs
        let mut and_seeds = Vec::with_capacity(n);
        let mut prngs = Vec::with_capacity(n);
        for seed in &instance.party_seeds {
            let mut prng = PRNG::seeded(seed);
            and_seeds.push(Seed::random(&mut prng));
            prngs.push(PRNG::seeded(&Seed::random(&mut prng)));
        }
        // Next, we want to setup the execution trace.
        // First, we need to extract out the input masks:
        let mut masks = Vec::with_capacity(n);
        for prng in &mut prngs {
            let mask = MultiBuffer::random(prng, program.private_size as usize);
            masks.push(mask);
        }

        // Now, generate the and bits. The first party doesn't get random bits.
        let mut and_bits = Vec::with_capacity(n - 1);
        for (i, seed) in and_seeds[0..].iter().enumerate() {
            if i == 0 {
                if let Some(aux) = &instance.first_aux {
                    and_bits.push(aux.clone());
                    continue;
                }
            }
            let mut prng = PRNG::seeded(seed);
            let aux = MultiBuffer::random(&mut prng, and_size);
            and_bits.push(aux);
        }

        let mut simulator = Simulator::new(
            public,
            &instance.masked_input,
            prngs,
            &and_bits,
            &masks,
            MultiQueue::new(&instance.messages),
        );
        if !simulator.run(program) {
            return false;
        }
        let mut messages = simulator.messages();
        messages.insert(j, instance.messages.clone());

        *hash = {
            let mut hasher = blake3::Hasher::new_keyed(&instance.message_hash_key);
            encode_into_std_write(&instance.masked_input, &mut hasher, config::standard())
                .expect("failed to call hash function");
            encode_into_std_write(&messages, &mut hasher, config::standard())
                .expect("failed to call hash function");
            hasher.finalize().into()
        }
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

    commitment == proof.commitment
}

#[cfg(test)]
mod test {
    use super::*;
    use rand_core::OsRng;

    use crate::{
        buffer::MultiBuffer, bytecode::BinaryInstruction::*, bytecode::Instruction::*,
        bytecode::Location::*, bytecode::Program,
    };

    fn run_instance(
        ctx: &[u8],
        program: &Program,
        public: &MultiBuffer,
        private: &MultiBuffer,
    ) -> bool {
        let proof = prove(&mut OsRng, ctx, program, public, private);
        assert!(proof.is_ok());
        let proof = proof.unwrap();
        verify(ctx, program, public, &proof)
    }

    fn assert_instance(ctx: &[u8], program: &Program, public: &MultiBuffer, private: &MultiBuffer) {
        assert!(run_instance(ctx, program, public, private));
    }

    #[test]
    fn test_empty_program() {
        let program = Program {
            public_size: 0,
            private_size: 0,
            instructions: vec![],
        };
        let public = MultiBuffer::new();
        let private = MultiBuffer::new();
        assert_instance(b"context", &program, &public, &private)
    }

    #[test]
    fn test_single_assertion() {
        let program = Program {
            public_size: 1,
            private_size: 1,
            instructions: vec![PushPrivate(0), AssertEq(Public(0))],
        };
        let mut public = MultiBuffer::new();
        public.push_u64(0xDEADBEEF);
        let mut private = MultiBuffer::new();
        private.push_u64(0xDEADBEEF);
        assert_instance(b"context", &program, &public, &private)
    }

    #[test]
    fn test_xor() {
        let program = Program {
            public_size: 1,
            private_size: 2,
            instructions: vec![
                PushPrivate(0),
                PushPrivate(1),
                Binary(Xor, Top),
                AssertEq(Public(0)),
            ],
        };
        let mut public = MultiBuffer::new();
        public.push_u64(0b11011);
        let mut private = MultiBuffer::new();
        private.push_u64(0b00111);
        private.push_u64(0b11100);
        assert_instance(b"context", &program, &public, &private)
    }

    #[test]
    fn test_and() {
        let program = Program {
            public_size: 1,
            private_size: 2,
            instructions: vec![
                PushPrivate(0),
                PushPrivate(1),
                Binary(And, Top),
                AssertEq(Public(0)),
            ],
        };
        let mut public = MultiBuffer::new();
        public.push_u64(0b00100);
        let mut private = MultiBuffer::new();
        private.push_u64(0b00111);
        private.push_u64(0b11100);
        assert_instance(b"context", &program, &public, &private)
    }

    #[test]
    fn test_and_public() {
        let program = Program {
            public_size: 2,
            private_size: 1,
            instructions: vec![PushPrivate(0), Binary(And, Public(0)), AssertEq(Public(1))],
        };
        let mut public = MultiBuffer::new();
        public.push_u64(0b0011);
        public.push_u64(0b0001);
        let mut private = MultiBuffer::new();
        private.push_u64(0b1101);
        assert_instance(b"context", &program, &public, &private)
    }
}
