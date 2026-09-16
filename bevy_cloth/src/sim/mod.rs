//! solver side code

pub mod constraints;
pub mod mesh_gen;

use std::fmt;

/// errors raised when turning a rest-pose mesh into solver tables.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ClothError {
    NotGrid {
        vertex_count: usize,
    },
    /// index buffer has the wrong length for the derived resolution.
    IndexCount {
        expected: usize,
        actual: usize,
    },
    /// index buffer is not the source layout; slot of the first difference.
    IndexLayout {
        first_mismatch: usize,
    },
    /// two particles share a bit-identical position
    DuplicateVertex {
        a: usize,
        b: usize,
    },
    /// `attached` names a particle that does not exist.
    AttachmentOutOfRange {
        particle: u32,
    },
}

impl fmt::Display for ClothError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotGrid { vertex_count } => {
                write!(f, "vertex count {vertex_count} is not (res + 1)^2")
            }
            Self::IndexCount { expected, actual } => {
                write!(f, "index count {actual} not match the expected {expected}")
            }
            Self::IndexLayout { first_mismatch } => {
                write!(
                    f,
                    "index buffer is not the source layout, first difference at {first_mismatch}"
                )
            }
            Self::DuplicateVertex { a, b } => {
                write!(f, "particles {a} and {b} share the same position")
            }
            Self::AttachmentOutOfRange { particle } => {
                write!(f, "attached particle {particle} does not exist")
            }
        }
    }
}
