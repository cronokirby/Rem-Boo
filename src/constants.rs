/// The context string used for our PRNG.
///
/// This provides some level of domain seperation for the random bytes we
/// generate from a seed.
pub const PRNG_CONTEXT: &[u8] = b"Rem-Boo v0.1 PRNG CONTEXT";
/// The context string we use for our commitments.
pub const COMMITMENT_CONTEXT: &[u8] = b"Rem-Boo v0.1 COMMITMENT CONTEXT";
/// The context string we use for deriving challenges.
pub const CHALLENGE_CONTEXT: &str = "Rem-Boo v0.1 CHALLENGE CONTEXT";
// These constants aim to achieve 128 bits of security.
/// The full number of simulations the prover does.
pub const FULL_SET_COUNT: usize = 218;
/// The number of simulations the verifier opens.
pub const SUBSET_COUNT: usize = 65;
/// The number of parties in each simulation.
pub const PARTY_COUNT: usize = 4;
