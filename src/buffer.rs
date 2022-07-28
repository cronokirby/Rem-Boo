use bincode::{Decode, Encode};
use rand_core::{CryptoRng, RngCore};

use crate::number::Number;

/// Represents a buffer over a given type.
///
/// This buffer is used for input, and for holding our stacks.
#[derive(Clone, Encode, Default, Decode, Debug, PartialEq)]
pub struct Buffer<T> {
    data: Vec<T>,
}

impl<T> Buffer<T> {
    /// Create a new empty buffer.
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn push(&mut self, x: T) {
        self.data.push(x)
    }

    pub fn pop(&mut self) -> Option<T> {
        self.data.pop()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }
}

impl<T: Clone> Buffer<T> {
    pub fn read(&self, i: u32) -> Option<T> {
        self.data.get(i as usize).cloned()
    }

    pub fn top(&self) -> Option<T> {
        self.data.last().cloned()
    }
}

impl<T: Number> Buffer<T> {
    pub fn random<R: RngCore + CryptoRng>(rng: &mut R, len: usize) -> Self {
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            data.push(T::random(rng));
        }
        Self { data }
    }

    pub fn xor(&mut self, other: &Buffer<T>) {
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a ^= *b;
        }
    }
}

/// Represents a queue yielding new items.
///
/// The main use in this trait is both generalizing over when we need an
/// actual queue, and when we need a fake queue yielding 0. This is used
/// to provide the auxilary messages not part of the simulation. The prover
/// doesn't need these, and wants a queue which doesn't impact the simulation,
/// while the verifier wants real messages provided to it inside the proof.
pub trait Queue<T> {
    fn next(&mut self) -> T;
}

/// Represents a queue which always yields 0.
///
/// This means that xoring with the result of the queue does nothing,
/// which is what the prover wants to do.
#[derive(Clone, Copy, Debug)]
pub struct NullQueue;

impl Queue<u64> for NullQueue {
    fn next(&mut self) -> u64 {
        0
    }
}

pub struct BufferQueue<'a, T> {
    buffer: &'a Buffer<T>,
    i: usize,
}

impl<'a, T> BufferQueue<'a, T> {
    pub fn new(buffer: &'a Buffer<T>) -> Self {
        Self { buffer, i: 0 }
    }
}

impl<'a, T: Clone> Queue<T> for BufferQueue<'a, T> {
    fn next(&mut self) -> T {
        let out = self.buffer.data[self.i].clone();
        self.i += 1;
        out
    }
}
