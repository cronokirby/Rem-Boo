use crate::bits::BitBuf;
use crate::circuit;

#[derive(Debug, Clone, Copy)]
pub enum Instruction {
    Zero,
    Read(usize),
    CheckZero(usize),
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

pub fn bake(circuit: &circuit::Circuit, mut public: BitBuf) -> Circuit {
    public.resize(circuit.pub_size);
    let priv_size = circuit.priv_size;
    let mem_size = circuit.mem_size;
    let and_size = circuit.and_size;

    let instructions = circuit
        .instructions
        .iter()
        .map(|instr| match *instr {
            circuit::Instruction::CheckZero(a) => Instruction::CheckZero(a),
            circuit::Instruction::Not(a) => Instruction::CheckZero(a),
            circuit::Instruction::Xor(a, b) => Instruction::Xor(a, b),
            circuit::Instruction::And(a, b) => Instruction::And(a, b),
            circuit::Instruction::XorPub(p, a) => {
                if public.read(p).into() {
                    Instruction::Not(a)
                } else {
                    Instruction::Read(a)
                }
            }
            circuit::Instruction::AndPub(p, a) => {
                if public.read(p).into() {
                    Instruction::Read(a)
                } else {
                    Instruction::Zero
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
