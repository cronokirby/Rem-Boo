use bincode::{Decode, Encode};

#[derive(Clone, Copy, Debug, Decode, Encode)]
pub enum Op1 {
    Not,
    Assert,
}

#[derive(Clone, Copy, Debug, Decode, Encode)]
pub enum Op2 {
    Xor,
    And,
}

#[derive(Clone, Copy, Debug, Decode, Encode)]
pub enum Instruction {
    Instr1 {
        op: Op1,
        input: usize,
        out: usize,
    },
    Instr2 {
        op: Op2,
        left: usize,
        right: usize,
        out: usize,
    },
    InstrPub {
        op: Op2,
        input: usize,
        public_input: usize,
        out: usize,
    },
}

impl Instruction {
    fn max_addr(&self) -> usize {
        match self {
            Instruction::Instr1 { input, out, .. } => *input.max(out),
            Instruction::Instr2 {
                left, right, out, ..
            } => *left.max(right).max(out),
            Instruction::InstrPub { input, out, .. } => *input.max(out),
        }
    }
}

#[derive(Clone, Debug, Decode, Encode)]
pub struct Circuit {
    pub priv_size: usize,
    pub pub_size: usize,
    pub(crate) mem_size: usize,
    pub instructions: Vec<Instruction>,
}

impl Circuit {
    pub fn new(priv_size: usize, pub_size: usize, instructions: Vec<Instruction>) -> Self {
        // Take the maximum memory address accessed.
        let mem_size = instructions.iter().map(|x| x.max_addr()).max().unwrap_or(0);
        // And then make sure we can also fit the input in memory.
        let mem_size = mem_size.max(priv_size);
        Self {
            priv_size,
            pub_size,
            mem_size,
            instructions,
        }
    }
}
