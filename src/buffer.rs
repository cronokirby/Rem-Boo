/// A MultiBuffer holds a single buffer for each data size.
///
/// The reason to do this is for efficiency, so that we can always have
/// aligned accesses to these values.
///
/// This buffer can be used as a random access array, or even as a stack.
#[derive(Clone, Debug, PartialEq)]
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
}
