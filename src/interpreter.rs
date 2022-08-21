use crate::baker::{Circuit, Instruction};
use crate::bits::{Bit, BitBuf};
use crate::simulation::Error;

pub struct Interpreter<'a> {
    circuit: &'a Circuit,
    mem: BitBuf,
    assertion_count: usize,
}

impl<'a> Interpreter<'a> {
    fn new(circuit: &'a Circuit, input: &BitBuf) -> Self {
        let mut mem = input.clone();
        mem.increase_capacity_to(circuit.mem_size);
        Self {
            circuit,
            mem,
            assertion_count: 0,
        }
    }

    fn instruction(&mut self, instruction: Instruction) -> Result<(), Error> {
        match instruction {
            Instruction::Zero => self.mem.push(Bit::zero()),
            Instruction::Read(a) => self.mem.push(self.mem.read(a)),
            Instruction::CheckZero(a) => {
                if self.mem.read(a) != Bit::zero() {
                    return Err(Error::AssertionFailed(self.assertion_count));
                }
                self.assertion_count += 1;
            }
            Instruction::Not(a) => self.mem.push(!self.mem.read(a)),
            Instruction::Xor(a, b) => self.mem.push(self.mem.read(a) ^ self.mem.read(b)),
            Instruction::And(a, b) => self.mem.push(self.mem.read(a) & self.mem.read(b)),
        };
        Ok(())
    }

    fn run(&mut self) -> Result<(), Error> {
        for instruction in &self.circuit.instructions {
            self.instruction(*instruction)?;
        }
        Ok(())
    }

    fn output(self) -> BitBuf {
        self.mem
    }
}

pub fn interpret(circuit: &Circuit, input: &BitBuf) -> Result<BitBuf, Error> {
    let mut interpreter = Interpreter::new(circuit, input);
    interpreter.run()?;
    Ok(interpreter.output())
}
