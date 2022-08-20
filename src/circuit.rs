use bincode::{Decode, Encode};

#[derive(Clone, Copy, Debug, Decode, Encode)]
pub enum Instruction {
    CheckZero(usize),
    Not(usize),
    Xor(usize, usize),
    And(usize, usize),
    XorPub(usize, usize),
    AndPub(usize, usize),
}

impl Instruction {
    fn has_output(&self) -> bool {
        !matches!(self, Instruction::CheckZero(_))
    }

    fn is_priv_and(&self) -> bool {
        matches!(self, Instruction::And(_, _))
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
        let mem_size = mem_size + priv_size;

        let and_size = instructions.iter().filter(|x| x.is_priv_and()).count();
        Self {
            priv_size,
            pub_size,
            mem_size,
            and_size,
            instructions,
        }
    }
}
