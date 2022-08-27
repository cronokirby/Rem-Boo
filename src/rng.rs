use bincode::{Decode, Encode};
use rand_core::{CryptoRng, RngCore};
use std::io::{BufReader, Read};

use crate::constants::PRNG_CONTEXT;

/// The number of bytes in an RNG seed
const SEED_LEN: usize = blake3::KEY_LEN;

/// Represents the seed to a pseudo-random RNG.
#[derive(Debug, Default, Encode, Decode, Clone)]
pub struct Seed([u8; blake3::KEY_LEN]);

impl Seed {
    /// Generate a random Seed.
    pub fn random<R: RngCore + CryptoRng>(rng: &mut R) -> Self {
        let mut bytes = [0u8; SEED_LEN];
        rng.fill_bytes(&mut bytes[..]);
        Self(bytes)
    }
}

/// The number of bytes we buffer in our RNG.
///
/// Using 64 is a good match with the XOF output from BLAKE3.
const BUF_LEN: usize = 64;

pub struct PRNG {
    reader: BufReader<blake3::OutputReader>,
}

impl PRNG {
    pub fn new(seed: &Seed) -> Self {
        let mut hasher = blake3::Hasher::new_keyed(&seed.0);
        hasher.update(PRNG_CONTEXT);
        Self::from_hasher(hasher)
    }

    pub fn from_hasher(hasher: blake3::Hasher) -> Self {
        let reader = hasher.finalize_xof();
        // Create the output with an uninitialized buffer, but fill it immediately
        Self {
            reader: BufReader::with_capacity(BUF_LEN, reader),
        }
    }
}

impl RngCore for PRNG {
    fn next_u32(&mut self) -> u32 {
        let mut bytes = [0u8; 4];
        self.fill_bytes(&mut bytes);
        u32::from_le_bytes(bytes)
    }

    fn next_u64(&mut self) -> u64 {
        let mut bytes = [0u8; 8];
        self.fill_bytes(&mut bytes);
        u64::from_le_bytes(bytes)
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.reader.read_exact(dest).unwrap();
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl CryptoRng for PRNG {}
