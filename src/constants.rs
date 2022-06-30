/// The context string used for our PRNG.
///
/// This provides some level of domain seperation for the random bytes we
/// generate from a seed.
pub const PRNG_CONTEXT: &str = "Rem-Boo v0.1 PRNG CONTEXT";
