use crate::{
    bits::{Bit, BitBuf},
    circuit::Instruction,
    circuit::{Circuit, Op1, Op2},
};

struct Tracer<'a> {
    circuit: &'a Circuit,
    public_input: &'a BitBuf,
    wire_masks: &'a BitBuf,
    and_i: usize,
    mem: BitBuf,
    mem_i: usize,
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
    pub fn new(circuit: &'a Circuit, public_input: &'a BitBuf, wire_masks: &'a BitBuf) -> Self {
        assert!(wire_masks.len() == circuit.priv_size + circuit.and_size);
        assert!(public_input.len() == circuit.pub_size);
        // Setup memory to contain the input and then zero bits.
        let mut mem = wire_masks.clone();
        mem.resize(circuit.mem_size);
        let trace = BitBuf::with_capacity(circuit.and_size);
        Self {
            circuit,
            public_input,
            wire_masks,
            and_i: circuit.priv_size,
            mem,
            mem_i: circuit.priv_size,
            trace,
        }
    }

    fn instr1(&mut self, op: Op1, input: usize) {
        match op {
            Op1::Not => {
                self.mem.write(self.mem_i, self.mem.read(input));
                self.mem_i += 1;
            }
            Op1::Assert => {}
        }
    }

    fn instr2(&mut self, op: Op2, left: usize, right: usize) {
        let l = self.mem.read(left);
        let r = self.mem.read(right);
        let out = match op {
            Op2::Xor => l ^ r,
            Op2::And => {
                self.trace.push(l & r);
                let and_mask = self.wire_masks.read(self.and_i);
                self.and_i += 1;
                and_mask
            }
        };
        self.mem.write(self.mem_i, out);
        self.mem_i += 1;
    }

    fn instr_pub(&mut self, op: Op2, input: usize, public_input: usize) {
        let l = self.mem.read(input);
        let r = self.public_input.read(public_input);
        let out = match op {
            Op2::Xor => l ^ r,
            Op2::And => l & r,
        };
        self.mem.write(self.mem_i, out);
        self.mem_i += 1;
    }

    fn instruction(&mut self, instruction: Instruction) {
        match instruction {
            Instruction::Instr1 { op, input } => self.instr1(op, input),
            Instruction::Instr2 { op, left, right } => self.instr2(op, left, right),
            Instruction::InstrPub {
                op,
                input,
                public_input,
            } => self.instr_pub(op, input, public_input),
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
