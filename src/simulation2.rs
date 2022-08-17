use crate::{
    baker::{Circuit, Instruction},
    bits::{Bit, BitBuf, BitQueue},
};

struct Tracer<'a> {
    circuit: &'a Circuit,
    input_masks: &'a BitBuf,
    and_masks: BitQueue<'a>,
    mem: BitBuf,
    trace: BitBuf,
}

impl<'a> Tracer<'a> {
    /// Create a new tracer, with a given list of wire masks.
    ///
    /// These masks should contain the masks for each input of the circuit,
    /// follow by masks for each and gate.
    ///
    /// We also take the circuit as input, since we use the metadata on input
    /// lengths to setup various internal data structures.
    pub fn new(circuit: &'a Circuit, input_masks: &'a BitBuf, and_masks: &'a BitBuf) -> Self {
        assert!(input_masks.len() >= circuit.priv_size);
        assert!(and_masks.len() >= circuit.and_size);
        // Setup memory to contain the input.
        let mut mem = input_masks.clone();
        mem.increase_capacity_to(circuit.mem_size);
        let trace = BitBuf::with_capacity(circuit.and_size);
        Self {
            circuit,
            input_masks,
            and_masks: BitQueue::new(and_masks),
            mem,
            trace,
        }
    }

    fn instruction(&mut self, instruction: Instruction) {
        match instruction {
            Instruction::Zero => self.mem.push(Bit::zero()),
            Instruction::Assert(_) => {}
            Instruction::Not(_) => {}
            Instruction::Xor(a, b) => self.mem.push(self.mem.read(a) ^ self.mem.read(b)),
            Instruction::And(a, b) => {
                let c = self.mem.read(a) & self.mem.read(b);
                
            },
        }
    }

    pub fn run(&mut self) {
        for instruction in &self.circuit.instructions {
            self.instruction(*instruction);
        }
    }
}

pub fn trace(circuit: &Circuit, public_input: &BitBuf, priv_input: &BitBuf) -> BitBuf {
    let mut tracer = Tracer::new(circuit, public_input, priv_input);
    tracer.run();
    tracer.trace
}

#[derive(Debug, Clone)]
pub struct PartyMasks {
    pub input_masks: BitBuf,
    pub and_out_masks: BitBuf,
    pub and_val_masks: BitBuf,
}

struct Simulator {}

pub fn simulate(
    circuit: &Circuit,
    public_input: &BitBuf,
    masked_input: &BitBuf,
    masks: &[PartyMasks],
) -> Vec<BitBuf> {
    todo!()
}
