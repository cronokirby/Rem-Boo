use bincode::{Decode, Encode};
use rand_core::{CryptoRng, RngCore};

use crate::number::Number;

/// Represents a buffer over a given type.
///
/// This buffer is used for input, and for holding our stacks.
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

/// A MultiBuffer holds a single buffer for each data size.
///
/// The reason to do this is for efficiency, so that we can always have
/// aligned accesses to these values.
///
/// This buffer can be used as a random access array, or even as a stack.
#[derive(Clone, Encode, Default, Decode, Debug, PartialEq)]
pub struct MultiBuffer {
    /// The buffer containing u64s.
    u64s: Vec<u64>,
}

impl MultiBuffer {
    // Create a new empty MultiBuffer.
    pub fn new() -> Self {
        Self { u64s: Vec::new() }
    }

    /// Create a random MultiBuffer with certain sizes
    pub fn random<R: RngCore + CryptoRng>(rng: &mut R, len64: usize) -> Self {
        let mut u64s = Vec::with_capacity(len64);
        for _ in 0..len64 {
            u64s.push(rng.next_u64());
        }
        Self { u64s }
    }

    pub fn len_u64(&self) -> usize {
        self.u64s.len()
    }

    /// Xor this buffer with the values of another buffer
    pub fn xor(&mut self, other: &MultiBuffer) {
        for (a, b) in self.u64s.iter_mut().zip(other.u64s.iter()) {
            *a ^= b;
        }
    }

    /// Read a u64 by index, if possible.
    pub fn read_u64(&self, i: u32) -> Option<u64> {
        self.u64s.get(i as usize).copied()
    }

    /// Push a u64 into this buffer.
    pub fn push_u64(&mut self, x: u64) {
        self.u64s.push(x);
    }

    /// Pop a u64 from this buffer, if possible.
    pub fn pop_u64(&mut self) -> Option<u64> {
        self.u64s.pop()
    }

    pub fn top_u64(&self) -> Option<u64> {
        self.u64s.last().copied()
    }

    /// Iterate over all the u64s in the buffer.
    pub fn iter_u64(&self) -> impl Iterator<Item = &u64> {
        self.u64s.iter()
    }
}

/// Represents a queue yielding new items.
///
/// The main use in this trait is both generalizing over when we need an
/// actual queue, and when we need a fake queue yielding 0. This is used
/// to provide the auxilary messages not part of the simulation. The prover
/// doesn't need these, and wants a queue which doesn't impact the simulation,
/// while the verifier wants real messages provided to it inside the proof.
pub trait Queue {
    fn next_u64(&mut self) -> u64;
}

/// Represents a queue which always yields 0.
///
/// This means that xoring with the result of the queue does nothing,
/// which is what the prover wants to do.
#[derive(Clone, Copy, Debug)]
pub struct NullQueue;

impl Queue for NullQueue {
    fn next_u64(&mut self) -> u64 {
        0
    }
}

/// A wrapper over a MultiBuffer providing a FIFO queue.
///
/// This is useful in situations where we need to read the elements of a buffer
/// in sequence.
#[derive(Debug)]
pub struct MultiQueue<'a> {
    buffer: &'a MultiBuffer,
    /// The next index to read a u64 from the buffer.
    i_u64: usize,
}

impl<'a> MultiQueue<'a> {
    /// Create a new queue by wrapping a buffer.
    pub fn new(buffer: &'a MultiBuffer) -> Self {
        Self { buffer, i_u64: 0 }
    }
}

impl<'a> Queue for MultiQueue<'a> {
    /// Read the next u64 value from the queue.
    fn next_u64(&mut self) -> u64 {
        let out = self.buffer.u64s[self.i_u64];
        self.i_u64 += 1;
        out
    }
}
