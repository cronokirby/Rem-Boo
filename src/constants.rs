/// The context string used for our PRNG.
///
/// This provides some level of domain seperation for the random bytes we
/// generate from a seed.
pub const PRNG_CONTEXT: &[u8] = b"rem-boo v0 PRNG CONTEXT";
