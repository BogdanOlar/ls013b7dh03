//! LS013B7DH03 Sharp LCD driver for [embedded-hal v1](https://github.com/rust-embedded/embedded-hal)
//!
#![no_std]

pub use embedded_hal as hal;
pub mod ls013b7dh03;

/// Invert the bit order in the given byte
pub(crate) const fn reverse_bits(mut b: u8) -> u8 {
    b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
    b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
    b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
    return b;
}

// =======================
// For unit tests only!
#[cfg(test)]
#[macro_use]
extern crate std;
// =======================
