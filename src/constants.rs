/// The context string used for our PRNG.
///
/// This provides some level of domain seperation for the random bytes we
/// generate from a seed.
pub const PRNG_CONTEXT: &[u8] = b"Rem-Boo v0.1 PRNG CONTEXT";
/// The context string we use for our commitments.
pub const COMMITMENT_CONTEXT: &str = "Rem-Boo v0.1 COMMITMENT CONTEXT";
