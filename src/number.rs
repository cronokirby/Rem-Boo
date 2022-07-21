use std::ops::{BitAnd, BitAndAssign, BitXor, BitXorAssign};

use rand_core::RngCore;

/// A trait for the common functionalities we need for numbers.
///
/// We need this trait to abstract over the various kinds of stacks we have.
pub trait Number: BitAnd + BitAndAssign + BitXor + BitXorAssign + Sized {
    /// Generate a random number.
    fn random<R: RngCore>(rng: &mut R) -> Self;
}

impl Number for u64 {
    fn random<R: RngCore>(rng: &mut R) -> Self {
        rng.next_u64()
    }
}
