use std::fmt::Debug;
use std::mem;

use bincode::{config, encode_into_std_write, Decode, Encode};
use rand_core::{CryptoRng, RngCore};

use crate::buffer::{Buffer, BufferQueue};
use crate::simulation::{exec_program, Simulator, Tracer};
use crate::{
    buffer::NullQueue,
    bytecode::Program,
    constants,
    rng::{random_selections, Prng, Seed},
};

#[derive(Debug)]
pub enum Error {
    BadProgram,
}

pub type Result<T> = std::result::Result<T, Error>;

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
    WithAux(&'a Seed, &'a Buffer<u64>),
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
    first_aux: Option<Buffer<u64>>,
    party_seeds: Vec<Seed>,
    commitment_keys: Vec<CommitmentKey>,
    masked_input: Buffer<u64>,
    messages: Buffer<u64>,
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
    commitment: Hash,
    root_seeds: Vec<Seed>,
    message_hashes: Vec<Hash>,
    message_hash_keys: Vec<[u8; blake3::KEY_LEN]>,
    all_states: Vec<(Buffer<u64>, Vec<Seed>)>,
    all_commitment_keys: Vec<Vec<CommitmentKey>>,
    all_commitments: Vec<Vec<Commitment>>,
    all_masked_inputs: Vec<Buffer<u64>>,
    all_messages: Vec<Vec<Buffer<u64>>>,
}

impl Prover {
    pub fn setup<R: RngCore + CryptoRng>(
        rng: &mut R,
        m: usize,
        n: usize,
        program: &Program,
        public: &Buffer<u64>,
        private: &Buffer<u64>,
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
                let mut prng = Prng::seeded(root_seed);
                for _ in 0..n {
                    party_seeds.push(Seed::random(&mut prng));
                    commitment_keys.push(CommitmentKey::random(&mut prng));
                }
            }

            // Split party seeds into and seeds and prngs
            let mut and_seeds = Vec::with_capacity(n);
            let mut prngs = Vec::with_capacity(n);
            for seed in &party_seeds {
                let mut prng = Prng::seeded(seed);
                and_seeds.push(Seed::random(&mut prng));
                prngs.push(Prng::seeded(&Seed::random(&mut prng)));
            }

            // Next, we want to setup the execution trace.
            // First, we need to extract out the input masks:
            let mut masks = Vec::with_capacity(n);
            for prng in &mut prngs {
                let mask = Buffer::random(prng, private.len());
                masks.push(mask);
            }
            let mut global_mask = masks[0].clone();
            for mask in &masks[1..] {
                global_mask.xor(mask);
            }

            // Second, execute the program to get a trace of the and bits.
            let mut tracer = Tracer::new(&global_mask);
            if exec_program(&mut tracer, program, public).is_none() {
                return Err(Error::BadProgram);
            }
            let trace = tracer.trace();

            // Now, generate the and bits. The first party doesn't get random bits.
            let mut and_bits = Vec::with_capacity(n);
            and_bits.push(trace.clone());
            for seed in &and_seeds[1..] {
                let mut prng = Prng::seeded(seed);
                let aux = Buffer::random(&mut prng, and_bits[0].len());
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
            let mut simulator = Simulator::new(&masked_input, prngs, &and_bits, &masks, NullQueue);
            if exec_program(&mut simulator, program, public).is_none() {
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
    public: &Buffer<u64>,
    private: &Buffer<u64>,
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

    let mut prng = Prng::from_hasher(hasher);
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

pub fn verify(ctx: &[u8], program: &Program, public: &Buffer<u64>, proof: &Proof) -> bool {
    let mut hasher = blake3::Hasher::new_derive_key(constants::CHALLENGE_CONTEXT);
    encode_into_std_write(program, &mut hasher, config::standard()).unwrap();
    encode_into_std_write(public, &mut hasher, config::standard()).unwrap();
    encode_into_std_write(proof.commitment, &mut hasher, config::standard()).unwrap();
    encode_into_std_write(ctx, &mut hasher, config::standard()).unwrap();
    let mut prng = Prng::from_hasher(hasher);
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
            let mut prng = Prng::seeded(root_seed);
            for _ in 0..n {
                party_seeds.push(Seed::random(&mut prng));
                commitment_keys.push(CommitmentKey::random(&mut prng));
            }
        }

        // Split party seeds into and seeds and prngs
        let mut and_seeds = Vec::with_capacity(n);
        let mut prngs = Vec::with_capacity(n);
        for seed in &party_seeds {
            let mut prng = Prng::seeded(seed);
            and_seeds.push(Seed::random(&mut prng));
            prngs.push(Prng::seeded(&Seed::random(&mut prng)));
        }

        // Next, we want to setup the execution trace.
        // First, we need to extract out the input masks:
        let mut masks = Vec::with_capacity(n);
        for prng in &mut prngs {
            let mask = Buffer::random(prng, program.private_size as usize);
            masks.push(mask);
        }
        let mut global_mask = masks[0].clone();
        for mask in &masks[1..] {
            global_mask.xor(mask);
        }

        // Second, execute the program to get a trace of the and bits.
        let mut tracer = Tracer::new(&global_mask);
        if exec_program(&mut tracer, program, public).is_none() {
            return false;
        }
        let trace = tracer.trace();
        and_size = trace.len();

        // Now, generate the and bits. The first party doesn't get random bits.
        let mut and_bits = Vec::with_capacity(n);
        and_bits.push(trace.clone());
        for seed in &and_seeds[1..] {
            let mut prng = Prng::seeded(seed);
            let aux = Buffer::random(&mut prng, and_bits[0].len());
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
            let mut prng = Prng::seeded(seed);
            and_seeds.push(Seed::random(&mut prng));
            prngs.push(Prng::seeded(&Seed::random(&mut prng)));
        }
        // Next, we want to setup the execution trace.
        // First, we need to extract out the input masks:
        let mut masks = Vec::with_capacity(n);
        for prng in &mut prngs {
            let mask = Buffer::random(prng, program.private_size as usize);
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
            let mut prng = Prng::seeded(seed);
            let aux = Buffer::random(&mut prng, and_size);
            and_bits.push(aux);
        }

        let mut simulator = Simulator::new(
            &instance.masked_input,
            prngs,
            &and_bits,
            &masks,
            BufferQueue::new(&instance.messages),
        );
        if exec_program(&mut simulator, program, public).is_none() {
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
        buffer::Buffer, bytecode::BinaryInstruction::*, bytecode::Instruction::*,
        bytecode::Location::*, bytecode::Program,
    };

    fn run_instance(
        ctx: &[u8],
        program: &Program,
        public: &Buffer<u64>,
        private: &Buffer<u64>,
    ) -> bool {
        let proof = prove(&mut OsRng, ctx, program, public, private);
        assert!(proof.is_ok());
        let proof = proof.unwrap();
        verify(ctx, program, public, &proof)
    }

    fn assert_instance(ctx: &[u8], program: &Program, public: &Buffer<u64>, private: &Buffer<u64>) {
        assert!(run_instance(ctx, program, public, private));
    }

    #[test]
    fn test_empty_program() {
        let program = Program {
            public_size: 0,
            private_size: 0,
            instructions: vec![],
        };
        let public = Buffer::new();
        let private = Buffer::new();
        assert_instance(b"context", &program, &public, &private)
    }

    #[test]
    fn test_single_assertion() {
        let program = Program {
            public_size: 1,
            private_size: 1,
            instructions: vec![PushPrivate(0), AssertEq(Public(0))],
        };
        let mut public = Buffer::new();
        public.push(0xDEADBEEF);
        let mut private = Buffer::new();
        private.push(0xDEADBEEF);
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
        let mut public = Buffer::new();
        public.push(0b11011);
        let mut private = Buffer::new();
        private.push(0b00111);
        private.push(0b11100);
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
        let mut public = Buffer::new();
        public.push(0b00100);
        let mut private = Buffer::new();
        private.push(0b00111);
        private.push(0b11100);
        assert_instance(b"context", &program, &public, &private)
    }

    #[test]
    fn test_and_public() {
        let program = Program {
            public_size: 2,
            private_size: 1,
            instructions: vec![PushPrivate(0), Binary(And, Public(0)), AssertEq(Public(1))],
        };
        let mut public = Buffer::new();
        public.push(0b0011);
        public.push(0b0001);
        let mut private = Buffer::new();
        private.push(0b1101);
        assert_instance(b"context", &program, &public, &private)
    }
}
