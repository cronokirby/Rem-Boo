use crate::{
    bits::{Bit, BitBuf},
    circuit::Circuit,
};

struct Tracer {
    mem: BitBuf,
    trace: BitBuf,
}

impl Tracer {
    /// Create a new tracer, with a given input buffer.
    ///
    /// We also take the circuit as input, since we use the metadata on input
    /// lengths to setup various internal data structures.
    pub fn new(input: &BitBuf, circuit: &Circuit) -> Self {
        assert!(input.len() == circuit.priv_size);
        // Setup memory to contain the input and then zero bits.
        let mut mem = input.clone();
        mem.resize(circuit.mem_size);
        let trace = BitBuf::with_capacity(circuit.trace_size);
        Self { mem, trace }
    }
}
