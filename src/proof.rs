use rand_core::{CryptoRng, RngCore};

use crate::{baker::bake, bits::BitBuf, circuit::Circuit, interpreter::interpret, Error};

pub struct Proof {
    ctx: Vec<u8>,
    private_input: Vec<u8>,
}

pub fn prove<R: CryptoRng + RngCore>(
    rng: &mut R,
    ctx: &[u8],
    circuit: &Circuit,
    private_input: &[u8],
    public_input: &[u8],
) -> Result<Proof, Error> {
    Ok(Proof {
        ctx: ctx.to_owned(),
        private_input: private_input.to_owned(),
    })
}

pub fn verify(ctx: &[u8], circuit: &Circuit, public_input: &[u8], proof: &Proof) -> bool {
    if ctx != proof.ctx {
        return false;
    }
    let public = BitBuf::from_bytes(public_input);
    let baked = bake(circuit, public);
    let mut private = BitBuf::from_bytes(&proof.private_input);
    private.resize(circuit.priv_size);
    interpret(&baked, &private).is_ok()
}

#[cfg(test)]
mod test {

    use crate::circuit::{Circuit, Instruction};
    use rand_core::OsRng;
    use Instruction::*;

    use super::*;

    fn create_and_check_proof(
        ctx: &[u8],
        circuit: &Circuit,
        private: &[u8],
        public: &[u8],
    ) -> Result<bool, Error> {
        let proof = prove(&mut OsRng, ctx, circuit, private, public)?;
        Ok(verify(ctx, circuit, public, &proof))
    }

    #[test]
    fn test_empty_circuit_works() {
        let circuit = Circuit::new(0, 0, Vec::new());
        let ctx = b"ctx";
        assert_eq!(Ok(true), create_and_check_proof(ctx, &circuit, b"", b""));
    }
}
