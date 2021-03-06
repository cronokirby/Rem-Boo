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
pub struct Prng {
    /// The reader used to generate new pseudo-random bytes.
    reader: blake3::OutputReader,
    /// The buffer to hold a partially consumed output from the hash.
    buf: [u8; BUF_LEN],
    /// The number of bytes used in the buffer so far
    used: usize,
}

impl std::fmt::Debug for Prng {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PRNG").finish()
    }
}

impl Prng {
    fn fill_buf(&mut self) {
        self.reader.fill(&mut self.buf)
    }
}

impl Prng {
    pub fn seeded(seed: &Seed) -> Self {
        // We extend the seed to an arbitrary stream of bits, with some domain separation.
        let mut hasher = blake3::Hasher::new_keyed(&seed.0);
        hasher.update(PRNG_CONTEXT);
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

    /// Read the next u16 from this PRNG.
    pub fn next_u16(&mut self) -> u16 {
        let mut out_bytes = [0u8; 2];
        self.fill_bytes(&mut out_bytes);
        u16::from_le_bytes(out_bytes)
    }

    /// Read the next u8 from this PRNG.
    pub fn next_u8(&mut self) -> u8 {
        let mut out_bytes = [0u8; 1];
        self.fill_bytes(&mut out_bytes);
        out_bytes[0]
    }
}

impl RngCore for Prng {
    fn next_u32(&mut self) -> u32 {
        let mut out_bytes = [0u8; 4];
        self.fill_bytes(&mut out_bytes);
        u32::from_le_bytes(out_bytes)
    }

    fn next_u64(&mut self) -> u64 {
        let mut out_bytes = [0u8; 8];
        self.fill_bytes(&mut out_bytes);
        u64::from_le_bytes(out_bytes)
    }

    fn fill_bytes(&mut self, mut dest: &mut [u8]) {
        let remaining = self.buf.len() - self.used;
        if dest.len() < remaining {
            let new_used = self.used + dest.len();
            dest.copy_from_slice(&self.buf[self.used..new_used]);
            self.used = new_used;
        } else {
            // Idea: copy the remainder of this buffer, and then full chunks,
            // and then a bit of the beginning.
            dest[..remaining].copy_from_slice(&self.buf[self.used..]);
            dest = &mut dest[remaining..];

            while dest.len() >= self.buf.len() {
                self.fill_buf();
                dest[..self.buf.len()].copy_from_slice(&self.buf);
                dest = &mut dest[self.buf.len()..];
            }

            self.fill_buf();
            let new_used = dest.len();
            dest.copy_from_slice(&self.buf[..new_used]);
            self.used = new_used;
        }
    }

    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dest);
        Ok(())
    }
}

impl CryptoRng for Prng {}

fn random_mod<R: RngCore>(rng: &mut R, n: u8) -> u8 {
    let mask = n.checked_next_power_of_two().unwrap_or(0).wrapping_sub(1);
    loop {
        let mut bytes = [0u8; 1];
        rng.fill_bytes(&mut bytes);
        let c = bytes[0] & mask;
        if c < n {
            return c;
        }
    }
}

fn random_subset<R: RngCore>(rng: &mut R, m: u8, tau: u8) -> Vec<bool> {
    assert!(tau < m);
    let mut out = vec![false; m as usize];
    let mut numbers: Vec<usize> = (0..m as usize).collect();
    for i in 0..tau {
        numbers.swap(random_mod(rng, m - i) as usize, (m - 1) as usize);
    }
    for &selected in &numbers[(m - tau) as usize..] {
        out[selected] = true;
    }
    out
}

/// Select a random subset of size tau from m, and then a random index mod n for each element.
///
/// This requires that all the values fit within a single byte, which will be the case,
/// for the security parameters we target.
pub fn random_selections<R: RngCore>(
    rng: &mut R,
    m: usize,
    tau: usize,
    n: usize,
) -> Vec<Option<usize>> {
    let selected = random_subset(rng, u8::try_from(m).unwrap(), u8::try_from(tau).unwrap());
    let nu8 = u8::try_from(n).unwrap();
    selected
        .into_iter()
        .map(|x| {
            if x {
                Some(random_mod(rng, nu8) as usize)
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod test {
    use rand_core::OsRng;

    use super::*;

    #[test]
    fn test_prng_reproducability() {
        let seed = Seed([0xAB; SEED_LEN]);
        let mut rng1 = Prng::seeded(&seed);
        let mut rng2 = Prng::seeded(&seed);
        assert_eq!(rng1.next_u8(), rng2.next_u8());
        assert_eq!(rng1.next_u16(), rng2.next_u16());
        assert_eq!(rng1.next_u32(), rng2.next_u32());
        assert_eq!(rng1.next_u64(), rng2.next_u64());
        let mut data1 = [0u8; 2 * BUF_LEN];
        rng1.fill_bytes(&mut data1);
        let mut data2 = [0u8; 2 * BUF_LEN];
        rng2.fill_bytes(&mut data2);
        assert_eq!(data1, data2);
    }

    #[test]
    fn test_random_subset_has_right_size() {
        let m = 218;
        let tau = 65;
        let subset = random_subset(&mut OsRng, m, tau);
        assert_eq!(subset.len(), m as usize);
        assert_eq!(subset.iter().filter(|x| **x).count(), tau as usize);
    }
}
