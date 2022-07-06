use bincode::{Decode, Encode};

/// A MultiBuffer holds a single buffer for each data size.
///
/// The reason to do this is for efficiency, so that we can always have
/// aligned accesses to these values.
///
/// This buffer can be used as a random access array, or even as a stack.
#[derive(Clone, Encode, Decode, Debug, PartialEq)]
pub struct MultiBuffer {
    /// The buffer containing u64s.
    u64s: Vec<u64>,
}

impl MultiBuffer {
    // Create a new empty MultiBuffer.
    pub fn new() -> Self {
        Self { u64s: Vec::new() }
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

/// A wrapper over a MultiBuffer providing a FIFO queue.
///
/// This is useful in situations where we need to read the elements of a buffer
/// in sequence.
pub struct MultiQueue {
    buffer: MultiBuffer,
    /// The next index to read a u64 from the buffer.
    i_u64: usize,
}

impl MultiQueue {
    /// Create a new queue by wrapping a buffer.
    pub fn new(buffer: MultiBuffer) -> Self {
        Self { buffer, i_u64: 0 }
    }

    /// Read the next u64 value from the queue.
    pub fn next_u64(&mut self) -> u64 {
        let out = self.buffer.u64s[self.i_u64];
        self.i_u64 += 1;
        out
    }
}
