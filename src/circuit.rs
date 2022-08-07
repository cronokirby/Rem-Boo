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
    },
    Instr2 {
        op: Op2,
        left: usize,
        right: usize,
    },
    InstrPub {
        op: Op2,
        input: usize,
        public_input: usize,
    },
}

impl Instruction {
    fn has_output(&self) -> bool {
        !matches!(
            self,
            Instruction::Instr1 {
                op: Op1::Assert,
                ..
            }
        )
    }

    fn is_priv_and(&self) -> bool {
        matches!(self, Instruction::Instr2 { op: Op2::And, .. })
    }
}

#[derive(Clone, Debug, Decode, Encode)]
pub struct Circuit {
    pub priv_size: usize,
    pub pub_size: usize,
    pub(crate) mem_size: usize,
    pub(crate) and_size: usize,
    pub instructions: Vec<Instruction>,
}

impl Circuit {
    pub fn new(priv_size: usize, pub_size: usize, instructions: Vec<Instruction>) -> Self {
        // Take the maximum memory address accessed.
        let mem_size = instructions.iter().filter(|x| x.has_output()).count();
        // And then make sure we can also fit the input in memory.
        let mem_size = mem_size.max(priv_size);

        let trace_size = instructions.iter().filter(|x| x.is_priv_and()).count();
        Self {
            priv_size,
            pub_size,
            mem_size,
            and_size: trace_size,
            instructions,
        }
    }
}
