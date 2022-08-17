use crate::bits::BitBuf;
use crate::circuit;

#[derive(Debug, Clone, Copy)]
pub enum Instruction {
    Zero,
    Assert(usize),
    Not(usize),
    Xor(usize, usize),
    And(usize, usize),
}

#[derive(Debug, Clone)]
pub struct Circuit {
    pub priv_size: usize,
    pub mem_size: usize,
    pub and_size: usize,
    pub instructions: Vec<Instruction>,
}

pub fn bake(circuit: &circuit::Circuit, public: BitBuf) -> Circuit {
    let priv_size = circuit.priv_size;
    let mem_size = circuit.mem_size;
    let and_size = circuit.and_size;

    let instructions = circuit
        .instructions
        .iter()
        .filter_map(|instr| match *instr {
            circuit::Instruction::Assert(a) => Some(Instruction::Assert(a)),
            circuit::Instruction::Not(a) => Some(Instruction::Assert(a)),
            circuit::Instruction::Xor(a, b) => Some(Instruction::Xor(a, b)),
            circuit::Instruction::And(a, b) => Some(Instruction::And(a, b)),
            circuit::Instruction::XorPub(p, a) => {
                if public.read(p).into() {
                    Some(Instruction::Not(a))
                } else {
                    None
                }
            }
            circuit::Instruction::AndPub(p, a) => {
                if public.read(p).into() {
                    None
                } else {
                    Some(Instruction::Zero)
                }
            }
        })
        .collect();

    Circuit {
        priv_size,
        mem_size,
        and_size,
        instructions,
    }
}
