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
            Instruction::CheckZero(_) => {}
            Instruction::Not(_) => {}
            Instruction::Xor(a, b) => self.mem.push(self.mem.read(a) ^ self.mem.read(b)),
            Instruction::And(a, b) => {
                let c = self.mem.read(a) & self.mem.read(b);
            }
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

struct Party<'a> {
    mem: BitBuf,
    out: BitBuf,
    and_val_masks: BitQueue<'a>,
    and_out_masks: BitQueue<'a>,
}

impl<'a> Party<'a> {
    fn new(circuit: &Circuit, masks: &'a PartyMasks) -> Self {
        let mut mem = masks.input_masks.clone();
        mem.increase_capacity_to(circuit.mem_size);
        let out = BitBuf::new();
        let and_val_masks = BitQueue::new(&masks.and_val_masks);
        let and_out_masks = BitQueue::new(&masks.and_out_masks);
        Self {
            mem,
            out,
            and_val_masks,
            and_out_masks,
        }
    }
}

pub enum Error {
    AssertionFailed(usize),
}

struct Simulator<'a> {
    circuit: &'a Circuit,
    assertion_count: usize,
    input_mem: BitBuf,
    parties: Vec<Party<'a>>,
}

impl<'a> Simulator<'a> {
    fn new(circuit: &'a Circuit, masked_input: &BitBuf, masks: &'a [PartyMasks]) -> Self {
        let mut input_mem = masked_input.clone();
        input_mem.increase_capacity_to(circuit.mem_size);
        let parties = masks
            .iter()
            .map(|masks| Party::new(circuit, masks))
            .collect();
        Self {
            circuit,
            assertion_count: 0,
            input_mem,
            parties,
        }
    }

    fn zero(&mut self) {
        for party in &mut self.parties {
            party.mem.push(Bit::zero());
        }
        self.input_mem.push(Bit::zero());
    }

    fn check_zero(&mut self, a: usize) -> Result<(), Error> {
        let mut mask = Bit::zero();
        for party in &mut self.parties {
            let mask_share = party.mem.read(a);
            party.out.push(mask_share);
            mask ^= mask_share;
        }
        let out = self.input_mem.read(a) ^ mask;
        if bool::from(out) {
            return Err(Error::AssertionFailed(self.assertion_count));
        }
        self.assertion_count += 1;
        Ok(())
    }

    fn xor(&mut self, a: usize, b: usize) {
        for party in &mut self.parties {
            party.mem.push(party.mem.read(a) ^ party.mem.read(b));
        }
        self.input_mem
            .push(self.input_mem.read(a) ^ self.input_mem.read(b));
    }

    fn not(&mut self, a: usize) {
        for party in &mut self.parties {
            party.mem.push(party.mem.read(a));
        }
        self.input_mem.push(!self.input_mem.read(a));
    }

    fn and(&mut self, a: usize, b: usize) {
        let z_a = self.input_mem.read(a);
        let z_b = self.input_mem.read(b);
        let mut s = Bit::zero();
        for party in &mut self.parties {
            let s_share = (z_a & party.mem.read(b))
                ^ (z_b & party.mem.read(a))
                ^ party.and_val_masks.next()
                ^ party.and_out_masks.next();
            party.out.push(s_share);
            s ^= s_share;
        }
        self.input_mem.push(s ^ (z_a & z_b));
    }

    fn instruction(&mut self, instr: &Instruction) -> Result<(), Error> {
        match *instr {
            Instruction::Zero => self.zero(),
            Instruction::CheckZero(a) => self.check_zero(a)?,
            Instruction::Not(a) => self.not(a),
            Instruction::Xor(a, b) => self.xor(a, b),
            Instruction::And(a, b) => self.and(a, b),
        };
        Ok(())
    }

    fn run(&mut self) -> Result<(), Error> {
        for instruction in &self.circuit.instructions {
            self.instruction(instruction)?;
        }
        Ok(())
    }

    fn output(self) -> Vec<BitBuf> {
        self.parties.into_iter().map(|party| party.out).collect()
    }
}

pub fn simulate(
    circuit: &Circuit,
    masked_input: &BitBuf,
    masks: &[PartyMasks],
) -> Result<Vec<BitBuf>, Error> {
    let mut simulator = Simulator::new(circuit, masked_input, masks);
    simulator.run()?;
    Ok(simulator.output())
}
