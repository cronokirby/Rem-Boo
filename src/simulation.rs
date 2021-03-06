/// Represents an interpreter, which can run operations.
///
/// We use this to abstract both over data types, and over operations.
/// Using basic operations, we can build up more complicated operations.
trait Interpreter {
    /// The type of immediate arguments
    type Immediate;

    /// Compute the and of the top two items on the stack.
    fn and(&mut self);

    /// Compute the and of the top of the stack and an immediate value.
    fn and_imm(&mut self, imm: Self::Immediate);

    /// Compute the xor of the top two items on the stack.
    fn xor(&mut self);

    /// Compute the xor of the top of the stack and an immediate value.
    fn xor_imm(&mut self, imm: Self::Immediate);
}
