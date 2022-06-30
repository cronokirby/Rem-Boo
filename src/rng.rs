use bincode::{Decode, Encode};
use rand_core::{CryptoRng, RngCore};

use crate::constants::PRNG_CONTEXT;

/// The number of bytes in an RNG seed
const SEED_LEN: usize = blake3::KEY_LEN;

#[derive(Debug, Encode, Decode, Clone, PartialEq)]
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

/// Represents a pseudo-random number generator.
/// 
/// This generator can be initialized via a seed, or a BLAKE3 hasher.
/// Given the same seed, or a hasher in the same state, the same output will
/// be generated.
/// 
/// When generating a single bit, an entire byte of randomness is consumed,
/// so generating 32 bits will not yield the same result as generating a u32.
pub struct PRNG {
    /// The reader used to generate new pseudo-random bytes.
    reader: blake3::OutputReader,
    /// The buffer to hold a partially consumed output from the hash.
    buf: [u8; BUF_LEN],
    /// The number of bytes used in the buffer so far
    used: usize,
}

impl PRNG {
    fn fill_buf(&mut self) {
        self.reader.fill(&mut self.buf)
    }
}

impl PRNG {
    pub fn seeded(seed: &Seed) -> Self {
        // We extend the seed to an arbitrary stream of bits, with some domain separation.
        let mut hasher = blake3::Hasher::new_keyed(&seed.0);
        hasher.update(PRNG_CONTEXT.as_bytes());
        Self::from_hasher(hasher)
    }

    /// Create a BitPRNG from a blake3 hasher.
    ///
    /// This will finalize the hasher, and then read bits from its output.
    pub fn from_hasher(hasher: blake3::Hasher) -> Self {
        let reader = hasher.finalize_xof();
        // Create the output with an uninitialized buffer, but fill it immediately
        let mut out = Self {
            reader,
            buf: [0; BUF_LEN],
            used: 0,
        };
        out.fill_buf();
        out
    }
}
