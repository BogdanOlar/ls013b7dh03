//! LS013B7DH03 Sharp LCD driver for [embedded-hal v1](https://github.com/rust-embedded/embedded-hal)
//!
#![no_std]

// =======================
// For unit tests only!
#[cfg(test)]
#[macro_use]
extern crate std;
// =======================

pub use embedded_hal as hal;

use embedded_hal::{
    digital::OutputPin,
    spi::{Mode, Phase, Polarity, SpiBus},
};

/// Convenience `Spi::Mode` struct instance needed by the Ls013b7dh03 display.
/// Feel free to use this to initialize the Spi passed to this driver
pub const SPIMODE: Mode = Mode {
    polarity: Polarity::IdleLow,
    phase: Phase::CaptureOnFirstTransition,
};

/// The width, in pixels of the Ls013b7dh03 display
pub const WIDTH: usize = 128;
/// The height, in pixels of the Ls013b7dh03 display
pub const HEIGHT: usize = 128;

const LINE_WIDTH_BYTE_COUNT: usize = WIDTH / (u8::BITS as usize);
const LINE_PADDING_BYTE_COUNT: usize = 1;
const LINE_ADDRESS_BYTE_COUNT: usize = 1;
const LINE_TOTAL_BYTE_COUNT: usize =
    LINE_ADDRESS_BYTE_COUNT + LINE_WIDTH_BYTE_COUNT + LINE_PADDING_BYTE_COUNT;

/// The buffer size this driver needs
pub const BUF_SIZE: usize = HEIGHT * LINE_TOTAL_BYTE_COUNT;

const LINE_CACHE_DWORD_COUNT: usize = HEIGHT / u32::BITS as usize;
const FILLER_BYTE: u8 = 0xFF;

/// LCD Mode flags
#[derive(Debug)]
#[repr(u8)]
enum LcdMode {
    Clear = 0x20,
    Update = 0x80,
}

/// LCD Driver errors
#[derive(Debug, PartialEq)]
pub enum LcdError {
    OutOfBounds { x: u8, y: u8 },
}

/// An Ls013b7dh03 display driver.
pub struct Ls013b7dh03<'a, SPI, CS, CI> {
    spi: SPI,
    cs_pin: CS,
    com_in_pin: CI,
    buffer: &'a mut [u8; BUF_SIZE],
    line_cache: [u32; LINE_CACHE_DWORD_COUNT],
}

impl<'a, SPI, CS, DISP> Ls013b7dh03<'a, SPI, CS, DISP>
where
    SPI: SpiBus,
    CS: OutputPin,
    DISP: OutputPin,
{
    /// Create a new Ls013b7dh03 display driver
    ///
    /// Because this driver does not have its own internal buffer, a `u8` mut slice of size [`BUF_SIZE`] mut be provided
    pub fn new(
        spi: SPI,
        mut cs_pin: CS,
        mut com_in_pin: DISP,
        buffer: &'a mut [u8; BUF_SIZE],
    ) -> Self {
        let _ = com_in_pin.set_low();
        let _ = cs_pin.set_low();

        let mut disp = Self {
            spi,
            cs_pin,
            com_in_pin,
            buffer,
            line_cache: [0; LINE_CACHE_DWORD_COUNT],
        };

        // send clear command
        let _ = disp.cs_pin.set_high();
        let _ = disp.spi.write(&[LcdMode::Clear as u8, 0x00]);
        let _ = disp.cs_pin.set_low();

        // initialize buffer
        disp.init_buffer();

        // clear lines cache
        disp.clear_y_cache();

        disp
    }

    /// Initialize the internal buffer:
    /// - Write the on-wire address for each line, so that we only calculate them once
    /// - Set all pixels to OFF state (which corresponds to bit `1`)
    /// - Write the filler byte a the end of each line, so that we don't have to do it ever again
    fn init_buffer(&mut self) {
        // Write addresses and filler bytes to buffer
        for (addr, sl) in self
            .buffer
            .chunks_exact_mut(LINE_TOTAL_BYTE_COUNT)
            .enumerate()
        {
            // LCD address space starts at 1 for y
            sl[0] = reverse_bits((addr + 1) as u8);

            sl[1..(LINE_TOTAL_BYTE_COUNT - 1)]
                .iter_mut()
                .for_each(|b| *b = 0xFF);

            sl[LINE_TOTAL_BYTE_COUNT - 1] = FILLER_BYTE;
        }
    }

    /// Get the buffer index corresponding to a pixel coord, and its bitmask which shows which bit in the byte
    /// represents the pixel.
    fn get_pixel_addr(&self, x: u8, y: u8) -> Result<(usize, u8), LcdError> {
        if (x as usize) < WIDTH && (y as usize) < HEIGHT {
            let col_byte = x as usize / u8::BITS as usize;
            let col_bit = x as usize % u8::BITS as usize;
            let index = (y as usize * LINE_TOTAL_BYTE_COUNT) + (LINE_ADDRESS_BYTE_COUNT + col_byte);

            Ok((index, 0x80 >> col_bit))
        } else {
            Err(LcdError::OutOfBounds { x, y })
        }
    }

    fn clear_y_cache(&mut self) {
        self.line_cache.iter_mut().for_each(|w| *w = 0);
    }

    pub fn enable(&mut self) {
        let _ = self.com_in_pin.set_high();
    }

    pub fn disable(&mut self) {
        let _ = self.com_in_pin.set_low();
    }

    /// Set the state of a pixel at the given coordinates
    pub fn write(&mut self, x: u8, y: u8, is_pixel_on: bool) -> Result<(), LcdError> {
        let (index, bit_mask) = self.get_pixel_addr(x, y)?;

        // mark line as in need of update
        self.line_cache[y as usize / u32::BITS as usize] |=
            1u32 << (y as usize % u32::BITS as usize);

        match is_pixel_on {
            // Pixel ON is represented as a `0` bit, and the bit order must be in reverse
            true => self.buffer[index] &= !bit_mask,
            // Pixel OFF is represented as a `1` bit, and the bit order must be in reverse
            false => self.buffer[index] |= bit_mask,
        }

        Ok(())
    }

    /// Read the state of a pixel at the given coordiantes
    pub fn read(&self, x: u8, y: u8) -> Result<bool, LcdError> {
        let (index, bit_mask) = self.get_pixel_addr(x, y)?;

        Ok((self.buffer[index] & bit_mask) == 0)
    }

    /// Invert the state of a pixel at the given coordinates, and return current state
    pub fn flip(&mut self, x: u8, y: u8) -> Result<bool, LcdError> {
        let (index, bit_mask) = self.get_pixel_addr(x, y)?;

        // mark line as in need of update
        self.line_cache[y as usize / u32::BITS as usize] |=
            1u32 << (y as usize % u32::BITS as usize);

        self.buffer[index] ^= bit_mask;

        Ok((self.buffer[index] & bit_mask) == 0)
    }

    /// Update all display lines which have been modified since the last time this method was called.
    ///
    /// This method will try to aggregate multiple consecutive lines to be updated into a single SPI write, if possible.
    pub fn flush(&mut self) {
        // Only flush if there's something to write to spi
        if self.line_cache.into_iter().any(|w| w != 0) {
            // Bit-by-bit iterator over `self.line_cache` array
            let mut cache_bit_itr = (0..WIDTH).map(|x| {
                let i_byte = x / u32::BITS as usize;
                let i_bit = x % u32::BITS as usize;
                ((self.line_cache[i_byte] & (1u32 << i_bit)) != 0, x)
            });

            // Assert CS
            let _ = self.cs_pin.set_high();

            // Write update command
            let _ = self.spi.write(&[LcdMode::Update as u8]);

            // Check the bits in the `line_cache` to see which display lines need to be updated.
            // If there are one or more consecutive lines which need to be transmitted over SPI, then write them all
            // in the same call to `spi.write()`
            while let Some((is_set, i)) = cache_bit_itr.next() {
                if is_set {
                    let y_start = i;
                    let mut y_end = None;

                    while let Some((is_set, i)) = cache_bit_itr.next() {
                        if is_set {
                            y_end = Some(i);
                        } else {
                            break;
                        }
                    }

                    let i_start = y_start * LINE_TOTAL_BYTE_COUNT;
                    let i_end;

                    if let Some(y_end) = y_end {
                        i_end = (y_end + 1) * LINE_TOTAL_BYTE_COUNT;
                    } else {
                        i_end = i_start + LINE_TOTAL_BYTE_COUNT;
                    }

                    let _ = self.spi.write(&self.buffer[i_start..i_end]);
                }
            }

            // Write filler byte
            let _ = self.spi.write(&[FILLER_BYTE]);

            // Deassert CS
            let _ = self.cs_pin.set_low();

            // We've updated the display, prepare the cache for new writes
            self.clear_y_cache();
        }
    }
}

/// Invert the bit order in the given byte
pub(crate) const fn reverse_bits(mut b: u8) -> u8 {
    b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
    b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
    b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
    b
}

#[cfg(test)]
mod tests {
    use super::{Ls013b7dh03, LINE_TOTAL_BYTE_COUNT};
    use crate::{reverse_bits, LcdError, BUF_SIZE, HEIGHT, WIDTH};
    use core::convert::Infallible;
    use embedded_hal::{
        digital::{ErrorType as PinErrorType, InputPin, OutputPin, PinState},
        spi::{ErrorType as SpiErrorType, SpiBus},
    };
    use std::vec::Vec;

    #[test]
    fn test_init() {
        let mut buffer = [0; BUF_SIZE];
        let disp = build_display(&mut buffer);

        let spi_write_history = disp.spi.data_written;

        assert!(spi_write_history.len() == 1);

        // Only one spi write: to clear display
        assert_eq!(spi_write_history[0], vec![0x20, 0x00]);
    }

    #[test]
    fn test_init_flush() {
        let mut buffer = [0; BUF_SIZE];
        let mut disp = build_display(&mut buffer);

        // This should cause no spi writes, since no pixel was modified
        disp.flush();

        let spi_write_history = disp.spi.data_written;

        assert!(spi_write_history.len() == 1);

        // Only one spi write: to clear display when the driver was created
        assert_eq!(spi_write_history[0], vec![0x20, 0x00]);
    }

    #[test]
    fn test_write() {
        let mut buffer = [0; BUF_SIZE];
        let mut disp = build_display(&mut buffer);

        // Top left
        let res = disp.write(0, 0, true);
        assert!(res.is_ok());

        // Upper left zone
        let res = disp.write(15, 3, true);
        assert!(res.is_ok());

        // Same line
        let res = disp.write(16, 3, true);
        assert!(res.is_ok());

        // Next line
        let res = disp.write(16, 4, true);
        assert!(res.is_ok());

        // Bottom right
        let res = disp.write(127, 127, true);
        assert!(res.is_ok());

        // Out of bounds
        let res = disp.write(128, 128, true);
        assert!(res.is_err());

        // Flush
        disp.flush();

        let spi_write_history = disp.spi.data_written;

        // SPI Writes:
        // - LcdMode::Clear
        // - LcdMode::Update
        // - SPI Line with pixel (0,0)
        // - SPI Lines with pixels (15,3), (16,3), (16,4)
        // - SPI Line with pixel (127,128)
        // - FILLER byte
        assert_eq!(spi_write_history.len(), 6);

        // for d in spi_write_history {
        //     println!("***");
        //     print_spi_lines(d.as_slice());
        // }
    }

    #[test]
    fn test_read() {
        let mut buffer = [0; BUF_SIZE];
        let mut disp = build_display(&mut buffer);

        let _ = disp.write(0, 0, true);
        let _ = disp.write(15, 3, true);
        let _ = disp.write(16, 3, true);
        let _ = disp.write(16, 4, true);
        let _ = disp.write(127, 127, true);
        let _ = disp.write(128, 128, true);

        assert_eq!(disp.read(0, 0), Ok(true));
        assert_eq!(disp.read(15, 3), Ok(true));
        assert_eq!(disp.read(16, 3), Ok(true));
        assert_eq!(disp.read(16, 4), Ok(true));
        assert_eq!(disp.read(127, 127), Ok(true));
        assert_eq!(
            disp.read(128, 128),
            Err(LcdError::OutOfBounds { x: 128, y: 128 })
        );
    }

    #[test]
    fn test_read_write_full() {
        let mut buffer = [0; BUF_SIZE];
        let mut disp = build_display(&mut buffer);

        for x in 0..WIDTH as u8 {
            for y in 0..HEIGHT as u8 {
                assert_eq!(disp.read(x, y), Ok(false));
            }
        }

        for x in 0..WIDTH as u8 {
            for y in 0..HEIGHT as u8 {
                assert!(disp.write(x, y, true).is_ok())
            }
        }

        for x in 0..WIDTH as u8 {
            for y in 0..HEIGHT as u8 {
                assert_eq!(disp.read(x, y), Ok(true));
            }
        }

        for x in 0..WIDTH as u8 {
            for y in 0..HEIGHT as u8 {
                assert!(disp.write(x, y, false).is_ok())
            }
        }

        for x in 0..WIDTH as u8 {
            for y in 0..HEIGHT as u8 {
                assert_eq!(disp.read(x, y), Ok(false));
            }
        }
    }

    #[test]
    fn test_read_flip_full() {
        let mut buffer = [0; BUF_SIZE];
        let mut disp = build_display(&mut buffer);

        for x in 0..WIDTH as u8 {
            for y in 0..HEIGHT as u8 {
                assert_eq!(disp.read(x, y), Ok(false));
            }
        }

        for x in 0..WIDTH as u8 {
            for y in 0..HEIGHT as u8 {
                assert_eq!(disp.flip(x, y), Ok(true));
            }
        }

        for x in 0..WIDTH as u8 {
            for y in 0..HEIGHT as u8 {
                assert_eq!(disp.read(x, y), Ok(true));
            }
        }

        for x in 0..WIDTH as u8 {
            for y in 0..HEIGHT as u8 {
                assert_eq!(disp.flip(x, y), Ok(false));
            }
        }

        for x in 0..WIDTH as u8 {
            for y in 0..HEIGHT as u8 {
                assert_eq!(disp.read(x, y), Ok(false));
            }
        }
    }

    #[test]
    fn test_reverse_bits() {
        let b = 0b10110111;
        let rev_b = 0b11101101;

        assert_eq!(reverse_bits(b), rev_b);
    }

    #[allow(dead_code)]
    fn print_spi_lines(data: &[u8]) {
        let mut lines_itr = data.chunks_exact(LINE_TOTAL_BYTE_COUNT).into_iter();

        println!("spi.write:");
        while let Some(csl) = lines_itr.next() {
            print!("\t");
            print_spi_write(csl);
            println!();
        }

        if lines_itr.remainder().len() > 0 {
            print!("\t*** Remaining bytes: ");
            for b in lines_itr.remainder() {
                print!("0x{:0>2x}, ", *b);
            }
            println!();
        }
    }

    #[allow(dead_code)]
    fn print_spi_write(line: &[u8]) {
        assert!(line.len() == LINE_TOTAL_BYTE_COUNT);

        let addr = reverse_bits(line[0]);

        print!("{: >3} [", addr);

        for b in line[1..(LINE_TOTAL_BYTE_COUNT - 1)].iter() {
            for i in 0..u8::BITS {
                if ((0x80u8 >> i) & *b) != 0 {
                    print!(" ");
                } else {
                    print!("*");
                }
            }
        }

        print!("] fill=0x{:0>2x}", line[LINE_TOTAL_BYTE_COUNT - 1]);
    }

    struct PinFixture {
        pub state: PinState,
    }

    impl InputPin for PinFixture {
        fn is_high(&mut self) -> Result<bool, Self::Error> {
            Ok(self.state == PinState::High)
        }

        fn is_low(&mut self) -> Result<bool, Self::Error> {
            Ok(!self.is_high().unwrap())
        }
    }

    impl OutputPin for PinFixture {
        fn set_low(&mut self) -> Result<(), Self::Error> {
            self.state = PinState::Low;
            Ok(())
        }

        fn set_high(&mut self) -> Result<(), Self::Error> {
            self.state = PinState::High;
            Ok(())
        }
    }

    impl PinErrorType for PinFixture {
        type Error = Infallible;
    }

    struct SpiFixture {
        data_written: Vec<Vec<u8>>,
    }

    impl SpiBus for SpiFixture {
        fn read(&mut self, _words: &mut [u8]) -> Result<(), Self::Error> {
            Ok(())
        }

        fn write(&mut self, words: &[u8]) -> Result<(), Self::Error> {
            self.data_written.push(Vec::from(words));
            Ok(())
        }

        fn transfer(&mut self, _read: &mut [u8], _write: &[u8]) -> Result<(), Self::Error> {
            Ok(())
        }

        fn transfer_in_place(&mut self, _words: &mut [u8]) -> Result<(), Self::Error> {
            Ok(())
        }

        fn flush(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
    }

    impl SpiErrorType for SpiFixture {
        type Error = Infallible;
    }

    /// Build a display driver with the spi fixtures and th egiven buffer
    fn build_display(
        buffer: &mut [u8; BUF_SIZE],
    ) -> Ls013b7dh03<SpiFixture, PinFixture, PinFixture> {
        let spi = SpiFixture {
            data_written: Vec::new(),
        };
        let cs_pin = PinFixture {
            state: PinState::High,
        };
        let com_in_pin = PinFixture {
            state: PinState::High,
        };

        Ls013b7dh03::new(spi, cs_pin, com_in_pin, buffer)
    }
}
