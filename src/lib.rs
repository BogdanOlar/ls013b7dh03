//! LS013B7DH03 Sharp LCD driver for [embedded-hal v1.0](https://github.com/rust-embedded/embedded-hal)
//!
//! Use the optional `embedded_graphics` feature if you need to use this driver with the
//! [embeddded-graphics](https://github.com/embedded-graphics/embedded-graphics) 2D graphics library
//!
#![no_std]

// =======================
// For unit tests only!
#[cfg(test)]
#[macro_use]
extern crate std;
// =======================

pub mod prelude {
    pub use crate::{Ls013b7dh03, BUF_SIZE, HEIGHT, SPIMODE};
}

use embedded_hal::{
    digital::OutputPin,
    spi::{Mode, Phase, Polarity, SpiBus},
};

#[cfg(feature = "embedded_graphics")]
use embedded_graphics::{
    draw_target::DrawTarget,
    geometry::{Dimensions, Point, Size},
    pixelcolor::BinaryColor,
    primitives::Rectangle,
    Pixel,
};

/// The width, in pixels of the Ls013b7dh03 display
pub const WIDTH: usize = 128;

/// The height, in pixels of the Ls013b7dh03 display
pub const HEIGHT: usize = 128;

/// The buffer size this driver needs
pub const BUF_SIZE: usize = HEIGHT * LINE_TOTAL_BYTE_COUNT;

/// Convenience `embedded_hal::spi::Mode` struct instance needed by the Ls013b7dh03 display.
/// Feel free to use this to initialize the Spi passed to this driver
pub const SPIMODE: Mode = Mode {
    polarity: Polarity::IdleLow,
    phase: Phase::CaptureOnFirstTransition,
};

const LINE_WIDTH_BYTE_COUNT: usize = WIDTH / (u8::BITS as usize);
const LINE_PADDING_BYTE_COUNT: usize = 1;
const LINE_ADDRESS_BYTE_COUNT: usize = 1;
const LINE_TOTAL_BYTE_COUNT: usize =
    LINE_ADDRESS_BYTE_COUNT + LINE_WIDTH_BYTE_COUNT + LINE_PADDING_BYTE_COUNT;
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

impl<'a, SPI, CS, CI> Ls013b7dh03<'a, SPI, CS, CI>
where
    SPI: SpiBus,
    CS: OutputPin,
    CI: OutputPin,
{
    /// Create a new Ls013b7dh03 display driver
    ///
    /// This driver does not have its own internal buffer, so a `u8` mut slice of size [`BUF_SIZE`] must be provided.
    /// The `spi` parameter should be configured according to the [`SPIMODE`] constant provided by this crate,
    /// with a MAX baudrate of 1.1 MHz.
    /// The `com_in_pin` (Communication Invertion pin) must be handled by the application by using the `enable()` and
    /// `disable()` methods to toggle the pin at 65 Hz
    pub fn new(
        spi: SPI,
        mut cs_pin: CS,
        mut com_in_pin: CI,
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

    /// Return the resources used to create this driver
    pub fn destroy(self) -> (SPI, CS, CI) {
        (self.spi, self.cs_pin, self.com_in_pin)
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
            sl[0] = ((addr + 1) as u8).reverse_bits();

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

            // Pixel bits must be transmitted over SPI in reverse order,
            // so that's also their order in each byte of the buffer
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

        // We only want to mark this line to be transmitted over SPI if the pixel state actually needs changing
        if ((self.buffer[index] & bit_mask) == 0) ^ is_pixel_on {
            // mark line as in need of update
            self.line_cache[y as usize / u32::BITS as usize] |=
                1u32 << (y as usize % u32::BITS as usize);

            // flip the pixel state
            self.buffer[index] ^= bit_mask;
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
            // Bit-by-bit iterator over `self.line_cache`. Yields whether line needs updating, and the y of the line
            let mut y_cache_bits = (0..HEIGHT).map(|y| {
                let i_byte = y / u32::BITS as usize;
                let i_bit = y % u32::BITS as usize;
                ((self.line_cache[i_byte] & (1u32 << i_bit)) != 0, y)
            });

            // Assert CS
            let _ = self.cs_pin.set_high();

            // Write update command
            let spi_ret = self.spi.write(&[LcdMode::Update as u8]);
            assert!(spi_ret.is_ok());

            // Check the bits in the `line_cache` to see which display lines need to be updated.
            // If there are one or more consecutive lines which need to be transmitted over SPI, then write them all
            // in the same call to `spi.write()`
            // ... Find the y of first line which needs updating
            while let Some(y_start) =
                y_cache_bits.find_map(|(is_set, y)| if is_set { Some(y) } else { None })
            {
                // ... Find the y of first line which does NOT need updating
                let y_end = y_cache_bits
                    .find_map(|(is_set, y)| if is_set { None } else { Some(y) })
                    .unwrap_or(HEIGHT);

                // Calculate indexes in the buffer
                let i_start = y_start * LINE_TOTAL_BYTE_COUNT;
                let i_end = y_end * LINE_TOTAL_BYTE_COUNT;

                let spi_ret = self.spi.write(&self.buffer[i_start..i_end]);
                assert!(spi_ret.is_ok());
            }

            // Write filler byte
            let spi_ret = self.spi.write(&[FILLER_BYTE]);
            assert!(spi_ret.is_ok());

            // Deassert CS
            let _ = self.cs_pin.set_low();

            // We've updated the display, prepare the cache for new writes
            self.clear_y_cache();
        }
    }
}

#[cfg(feature = "embedded_graphics")]
impl<'a, SPI, CS, CI> Dimensions for Ls013b7dh03<'a, SPI, CS, CI> {
    fn bounding_box(&self) -> Rectangle {
        Rectangle {
            top_left: Point { x: 0, y: 0 },
            size: Size {
                width: WIDTH as u32,
                height: HEIGHT as u32,
            },
        }
    }
}

#[cfg(feature = "embedded_graphics")]
impl<'a, SPI, CS, CI> DrawTarget for Ls013b7dh03<'a, SPI, CS, CI>
where
    SPI: SpiBus,
    CS: OutputPin,
    CI: OutputPin,
{
    type Color = BinaryColor;
    type Error = core::convert::Infallible;

    fn draw_iter<I>(&mut self, pixels: I) -> Result<(), Self::Error>
    where
        I: IntoIterator<Item = Pixel<Self::Color>>,
    {
        // Check if the pixel coordinates are out of bounds (negative or greater than
        // (WIDTH,HEIGHT)). `DrawTarget` implementation are required to discard any out of bounds
        // pixels without returning an error or causing a panic.
        for (x, y, is_pixel_on) in pixels
            .into_iter()
            .filter(|p| p.0.x >= 0 && p.0.x < WIDTH as i32 && p.0.y >= 0 && p.0.y < HEIGHT as i32)
            .map(|p| (p.0.x as u8, p.0.y as u8, p.1.is_on()))
        {
            // NOTE: we are basically re-implementing `Ls013b7dh03.write()` here because we don't want to do the bounds
            //       check multiple times, AND this method may not return an OutOfBounds error or panic anyway.

            let col_byte = x as usize / u8::BITS as usize;
            let col_bit = x as usize % u8::BITS as usize;
            let index = (y as usize * LINE_TOTAL_BYTE_COUNT) + (LINE_ADDRESS_BYTE_COUNT + col_byte);

            // Pixel bits must be transmitted over SPI in reverse order,
            // so that's also their order in each byte of the buffer
            let bit_mask = 0x80 >> col_bit;

            // We only want to mark this line to be transmitted over SPI if the pixel state actually needs changing
            if ((self.buffer[index] & bit_mask) == 0) ^ is_pixel_on {
                // mark line as in need of update
                self.line_cache[y as usize / u32::BITS as usize] |=
                    1u32 << (y as usize % u32::BITS as usize);

                // flip the pixel state
                self.buffer[index] ^= bit_mask;
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{Ls013b7dh03, LINE_TOTAL_BYTE_COUNT};
    use crate::{LcdError, BUF_SIZE, FILLER_BYTE, HEIGHT, LINE_ADDRESS_BYTE_COUNT, WIDTH};
    use core::convert::Infallible;
    use embedded_hal::{
        digital::{ErrorType as PinErrorType, InputPin, OutputPin, PinState},
        spi::{ErrorType as SpiErrorType, SpiBus},
    };
    use std::{collections::HashMap, vec::Vec};

    #[cfg(feature = "embedded_graphics")]
    use embedded_graphics::{
        draw_target::DrawTarget, geometry::Point, pixelcolor::BinaryColor, Pixel,
    };

    #[test]
    fn consts() {
        assert_eq!(WIDTH % u8::BITS as usize, 0);
        assert_eq!(HEIGHT % u8::BITS as usize, 0);

        // If this fails, then the size of `line_cache` needs to be increased/adjusted
        assert_eq!(HEIGHT % u32::BITS as usize, 0);

        assert_eq!(BUF_SIZE / LINE_TOTAL_BYTE_COUNT, HEIGHT);
        assert_eq!(BUF_SIZE % LINE_TOTAL_BYTE_COUNT, 0);
    }

    #[test]
    fn init() {
        let mut buffer = [0; BUF_SIZE];
        let disp = build_display(&mut buffer);

        let spi_write_history = disp.spi.data_written;

        assert!(spi_write_history.len() == 1);

        // Only one spi write: to clear display
        assert_eq!(spi_write_history[0], vec![0x20, 0x00]);
    }

    #[test]
    fn init_flush() {
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
    fn write() {
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
        let res = disp.write(WIDTH as u8, HEIGHT as u8, true);
        assert_eq!(
            res,
            Err(LcdError::OutOfBounds {
                x: WIDTH as u8,
                y: HEIGHT as u8
            })
        );

        // Flush
        disp.flush();

        let spi_write_history = disp.spi.data_written;

        // SPI Writes:
        // - LcdMode::Clear
        // - LcdMode::Update
        // - SPI Line with pixel (0,0)
        // - SPI Lines with pixels (15,3), (16,3), (16,4)
        // - SPI Line with pixel (127,127)
        // - FILLER byte
        assert_eq!(spi_write_history.len(), 6);
    }

    #[test]
    fn redundant_writes() {
        let mut buffer = [0; BUF_SIZE];
        let mut disp = build_display(&mut buffer);

        // clear the SPI writes before redundant write
        disp.spi.data_written.clear();

        // Write "OFF" to pixels that are already "OFF". Should produce no additional SPI writes.
        for x in 0..WIDTH as u8 {
            for y in 0..HEIGHT as u8 {
                assert_eq!(disp.write(x, y, false), Ok(()));
            }
        }

        // Flush
        disp.flush();

        // No SPI writes were caused by calling write with a pixel state that is identical with the one in the buffer
        assert_eq!(disp.spi.data_written.len(), 0);
    }

    #[test]
    fn read() {
        let mut buffer = [0; BUF_SIZE];
        let mut disp = build_display(&mut buffer);

        let _ = disp.write(0, 0, true);
        let _ = disp.write(15, 3, true);
        let _ = disp.write(16, 3, true);
        let _ = disp.write(16, 4, true);
        let _ = disp.write(127, 127, true);
        let _ = disp.write(WIDTH as u8, HEIGHT as u8, true);

        assert_eq!(disp.read(0, 0), Ok(true));
        assert_eq!(disp.read(15, 3), Ok(true));
        assert_eq!(disp.read(16, 3), Ok(true));
        assert_eq!(disp.read(16, 4), Ok(true));
        assert_eq!(disp.read(127, 127), Ok(true));
        assert_eq!(
            disp.read(WIDTH as u8, HEIGHT as u8),
            Err(LcdError::OutOfBounds {
                x: WIDTH as u8,
                y: HEIGHT as u8
            })
        );
    }

    #[test]
    fn read_write_full() {
        let mut buffer = [0; BUF_SIZE];
        let mut disp = build_display(&mut buffer);

        // clear spi writes
        disp.spi.data_written.clear();

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

        // Check that on-wire data matches the expected values
        {
            disp.flush();

            let mut spi_parsed_pixels = HashMap::<(u8, u8), bool>::new();

            for spi_writes in disp.spi.data_written.clone() {
                for line in parse_spi_lines(&spi_writes) {
                    for (k, v) in line.pixels.iter().map(|p| ((p.x, p.y), p.is_on)) {
                        spi_parsed_pixels.insert(k, v);
                    }
                }
            }

            for x in 0..WIDTH as u8 {
                for y in 0..HEIGHT as u8 {
                    let res = spi_parsed_pixels.get(&(x, y));
                    assert_eq!(res, Some(&true));
                }
            }

            // clear spi writes
            disp.spi.data_written.clear();
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

        // Check that on-wire data matches the expected values
        {
            disp.flush();

            let mut spi_parsed_pixels = HashMap::<(u8, u8), bool>::new();

            for spi_writes in disp.spi.data_written.clone() {
                for line in parse_spi_lines(&spi_writes) {
                    for (k, v) in line.pixels.iter().map(|p| ((p.x, p.y), p.is_on)) {
                        spi_parsed_pixels.insert(k, v);
                    }
                }
            }

            for x in 0..WIDTH as u8 {
                for y in 0..HEIGHT as u8 {
                    let res = spi_parsed_pixels.get(&(x, y));
                    assert_eq!(res, Some(&false));
                }
            }

            // clear spi writes
            disp.spi.data_written.clear();
        }
    }

    #[test]
    fn read_flip_full() {
        let mut buffer = [0; BUF_SIZE];
        let mut disp = build_display(&mut buffer);

        // clear spi writes
        disp.spi.data_written.clear();

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

        // Check that on-wire data matches the expected values
        {
            disp.flush();

            let mut spi_parsed_pixels = HashMap::<(u8, u8), bool>::new();

            for spi_writes in disp.spi.data_written.clone() {
                for line in parse_spi_lines(&spi_writes) {
                    for (k, v) in line.pixels.iter().map(|p| ((p.x, p.y), p.is_on)) {
                        spi_parsed_pixels.insert(k, v);
                    }
                }
            }

            for x in 0..WIDTH as u8 {
                for y in 0..HEIGHT as u8 {
                    let res = spi_parsed_pixels.get(&(x, y));
                    assert_eq!(res, Some(&true));
                }
            }

            // clear spi writes
            disp.spi.data_written.clear();
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

        // Check that on-wire data matches the expected values
        {
            disp.flush();

            let mut spi_parsed_pixels = HashMap::<(u8, u8), bool>::new();

            for spi_writes in disp.spi.data_written.clone() {
                for line in parse_spi_lines(&spi_writes) {
                    for (k, v) in line.pixels.iter().map(|p| ((p.x, p.y), p.is_on)) {
                        spi_parsed_pixels.insert(k, v);
                    }
                }
            }

            for x in 0..WIDTH as u8 {
                for y in 0..HEIGHT as u8 {
                    let res = spi_parsed_pixels.get(&(x, y));
                    assert_eq!(res, Some(&false));
                }
            }

            // clear spi writes
            disp.spi.data_written.clear();
        }
    }

    #[test]
    fn read_write_grid() {
        let mut buffer = [0; BUF_SIZE];
        let mut disp = build_display(&mut buffer);

        const GRID_SIZE_X: u8 = 3;
        const GRID_SIZE_Y: u8 = 3;

        // clear spi writes
        disp.spi.data_written.clear();

        for x in 0..WIDTH as u8 {
            for y in 0..HEIGHT as u8 {
                let res;

                if (y % GRID_SIZE_Y == 0) || (x % GRID_SIZE_X == 0) {
                    res = disp.write(x, y, true);
                } else {
                    res = disp.write(x, y, false);
                }

                assert!(res.is_ok());
            }
        }

        for x in 0..WIDTH as u8 {
            for y in 0..HEIGHT as u8 {
                let res = disp.read(x, y);

                if (y % GRID_SIZE_Y == 0) || (x % GRID_SIZE_X == 0) {
                    assert_eq!(res, Ok(true));
                } else {
                    assert_eq!(res, Ok(false));
                }
            }
        }

        // Check that on-wire data matches the expected values
        {
            disp.flush();

            let mut spi_parsed_pixels = HashMap::<(u8, u8), bool>::new();

            for spi_writes in disp.spi.data_written.clone() {
                for line in parse_spi_lines(&spi_writes) {
                    for (k, v) in line.pixels.iter().map(|p| ((p.x, p.y), p.is_on)) {
                        spi_parsed_pixels.insert(k, v);
                    }
                    assert_eq!(line.filler, FILLER_BYTE);
                }
            }

            for x in 0..WIDTH as u8 {
                for y in 0..HEIGHT as u8 {
                    let res = spi_parsed_pixels.get(&(x, y));

                    if (y % GRID_SIZE_Y == 0) || (x % GRID_SIZE_X == 0) {
                        assert_eq!(res, Some(&true));
                    } else {
                        assert_eq!(res, Some(&false));
                    }
                }
            }

            // clear spi writes
            disp.spi.data_written.clear();
        }
    }

    #[test]
    #[cfg(feature = "embedded_graphics")]
    fn embedded_graphics_redundant_writes() {
        let mut buffer = [0; BUF_SIZE];
        let mut disp = build_display(&mut buffer);

        // clear the SPI writes before redundant write
        disp.spi.data_written.clear();

        // prepare iterator for the pixels to be written to "OFF" state
        // We want to cover each pixel on the screen AND additional out-of-bounds writes
        // (hence the `* 2` for WIDTH and HEIGHT)
        let pixels = (0..(WIDTH * 2))
            .flat_map(move |x| (0..(HEIGHT * 2)).map(move |y| (x, y)))
            .map(|(x, y)| {
                Pixel::<BinaryColor>(
                    Point {
                        x: x as i32,
                        y: y as i32,
                    },
                    BinaryColor::Off,
                )
            });

        // Write "OFF" to pixels that are already "OFF". Should produce no additional SPI writes.
        // do the grid writes using the `embedded-graphics` implementation of `DrawTarget`
        let res = disp.draw_iter(pixels);
        assert!(res.is_ok());

        // Flush
        disp.flush();

        // No SPI writes were caused by calling write with a pixel state that is identical with the one in the buffer
        assert_eq!(disp.spi.data_written.len(), 0);
    }

    #[test]
    #[cfg(feature = "embedded_graphics")]
    fn embedded_graphics_read_write_grid() {
        let mut buffer = [0; BUF_SIZE];
        let mut disp = build_display(&mut buffer);

        const GRID_SIZE_X: usize = 3;
        const GRID_SIZE_Y: usize = 3;

        // clear spi writes
        disp.spi.data_written.clear();

        // prepare iterator for the pixels to be written
        let pixels = (0..WIDTH)
            .flat_map(move |x| (0..HEIGHT).map(move |y| (x, y)))
            .map(|(x, y)| {
                if (y % GRID_SIZE_Y == 0) || (x % GRID_SIZE_X == 0) {
                    Pixel::<BinaryColor>(
                        Point {
                            x: x as i32,
                            y: y as i32,
                        },
                        BinaryColor::On,
                    )
                } else {
                    Pixel::<BinaryColor>(
                        Point {
                            x: x as i32,
                            y: y as i32,
                        },
                        BinaryColor::Off,
                    )
                }
            });

        // do the grid writes using the `embedded-graphics` implementation of `DrawTarget`
        let res = disp.draw_iter(pixels);
        assert!(res.is_ok());

        for x in 0..WIDTH as u8 {
            for y in 0..HEIGHT as u8 {
                let res = disp.read(x, y);

                if (y % GRID_SIZE_Y as u8 == 0) || (x % GRID_SIZE_X as u8 == 0) {
                    assert_eq!(res, Ok(true));
                } else {
                    assert_eq!(res, Ok(false));
                }
            }
        }

        // Check that on-wire data matches the expected values
        {
            disp.flush();

            let mut spi_parsed_pixels = HashMap::<(u8, u8), bool>::new();

            for spi_writes in disp.spi.data_written.clone() {
                for line in parse_spi_lines(&spi_writes) {
                    for (k, v) in line.pixels.iter().map(|p| ((p.x, p.y), p.is_on)) {
                        spi_parsed_pixels.insert(k, v);
                    }
                    assert_eq!(line.filler, FILLER_BYTE);
                }
            }

            for x in 0..WIDTH as u8 {
                for y in 0..HEIGHT as u8 {
                    let res = spi_parsed_pixels.get(&(x, y));

                    if (y % GRID_SIZE_Y as u8 == 0) || (x % GRID_SIZE_X as u8 == 0) {
                        assert_eq!(res, Some(&true));
                    } else {
                        assert_eq!(res, Some(&false));
                    }
                }
            }

            // clear spi writes
            disp.spi.data_written.clear();
        }
    }

    #[test]
    fn destroy() {
        let mut buffer = [0; BUF_SIZE];
        let disp = build_display(&mut buffer);

        let (_spi, _cs_pin, _ci_pin) = disp.destroy();

        // `disp` should no longer be available. The following line would cause a `borrow of moved value` compile error
        // disp.flush();
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

    #[derive(Debug)]
    struct LcdPixel {
        x: u8,
        y: u8,
        is_on: bool,
    }

    #[derive(Debug)]
    struct SpiDisplayLine {
        /// pixel in coord space ([0..WIDTH], [0..HEIGHT]), and wether or not it is ON
        pixels: Vec<LcdPixel>,
        /// Line filler byte
        filler: u8,
    }

    impl From<&[u8]> for SpiDisplayLine {
        fn from(line: &[u8]) -> Self {
            Self {
                pixels: (0..WIDTH)
                    .map(|x| {
                        let i_byte = (x / u8::BITS as usize) + LINE_ADDRESS_BYTE_COUNT;
                        let i_bit = x % u8::BITS as usize;
                        LcdPixel {
                            x: x.try_into().unwrap(),
                            y: line[0].reverse_bits() - 1,
                            is_on: (line[i_byte] & (0x80u8 >> i_bit)) == 0,
                        }
                    })
                    .collect(),
                filler: line[LINE_TOTAL_BYTE_COUNT - 1],
            }
        }
    }

    fn parse_spi_lines(data: &[u8]) -> Vec<SpiDisplayLine> {
        data.chunks_exact(LINE_TOTAL_BYTE_COUNT)
            .into_iter()
            .map(|sl| SpiDisplayLine::from(sl))
            .collect()
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

        let addr = line[0].reverse_bits();

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
}
