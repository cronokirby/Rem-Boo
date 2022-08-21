use std::{error, fmt};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Error {
    AssertionFailed(usize),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::AssertionFailed(x) => write!(f, "assertion {x} failed"),
        }
    }
}

impl error::Error for Error {}
