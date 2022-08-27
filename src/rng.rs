use bincode::{Decode, Encode};
use rand_core::{CryptoRng, RngCore};

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
