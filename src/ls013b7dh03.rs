use crate::reverse_bits;
use embedded_hal::{digital::OutputPin, spi::SpiBus};

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
const FILLER_BYTE: u8 = 0x00;

/// LCD Mode flags
#[derive(Debug)]
#[repr(u8)]
enum LcdMode {
    Clear = 0x20,
    Update = 0x80,
}

#[derive(Debug)]
pub enum LcdError {
    OutOfBounds { x: u8, y: u8 },
    OutOfBoundsY { y: u8 },
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

        // Assert CS
        let _ = disp.cs_pin.set_high();

        // initialize buffer
        disp.init_buffer();

        // clear lines cache
        disp.clear_y_cache();

        // send clear command
        let _ = disp.spi.write(&[LcdMode::Clear as u8, FILLER_BYTE]);

        // Deassert CS
        let _ = disp.cs_pin.set_low();

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

    fn clear_y_cache(&mut self) {
        self.line_cache.iter_mut().for_each(|w| *w = 0);
    }

    fn mark_y_cache(&mut self, y: u8) -> Result<(), LcdError> {
        if let Some(w) = self.line_cache.get_mut(y as usize / u32::BITS as usize) {
            *w = *w | (1u32 << (y as usize % u32::BITS as usize));
            Ok(())
        } else {
            Err(LcdError::OutOfBoundsY { y })
        }
    }

    pub fn enable(&mut self) {
        let _ = self.com_in_pin.set_high();
    }

    pub fn disable(&mut self) {
        let _ = self.com_in_pin.set_low();
    }

    /// Set the state of a pixel at the given coordinates
    pub fn write(&mut self, x: u8, y: u8, is_pixel_on: bool) -> Result<(), LcdError> {
        self.mark_y_cache(y)
            .map_err(|_| LcdError::OutOfBounds { x, y })?;

        if (x as usize) < WIDTH {
            let col_byte = x as usize / u8::BITS as usize;
            let col_bit = x as usize % u8::BITS as usize;
            let index = (y as usize * LINE_TOTAL_BYTE_COUNT) + (LINE_ADDRESS_BYTE_COUNT + col_byte);

            if is_pixel_on {
                // Pixel ON is represented as a `0` bit
                self.buffer[index] &= !(1u8 << col_bit);
            } else {
                // Pixel OFF is represented as a `1` bit
                self.buffer[index] |= 1u8 << col_bit;
            }

            Ok(())
        } else {
            Err(LcdError::OutOfBounds { x, y })
        }
    }

    /// Read the state of a pixel at the given coordiantes
    pub fn read(&self, x: u8, y: u8) -> Result<bool, LcdError> {
        if (x as usize) < WIDTH && (y as usize) < HEIGHT {
            let col_byte = x as usize / u8::BITS as usize;
            let col_bit = x as usize % u8::BITS as usize;
            let index = (y as usize * LINE_TOTAL_BYTE_COUNT) + (LINE_ADDRESS_BYTE_COUNT + col_byte);

            Ok((self.buffer[index] & (1u8 << col_bit)) == 0)
        } else {
            Err(LcdError::OutOfBounds { x, y })
        }
    }

    /// Update all display lines which have been modified since the last time this method was called.
    ///
    /// This method will try to aggregate multiple consecutive lines to be updated into a single SPI write, if possible.
    pub fn flush(&mut self) {
        // Only flush if there's something to write to spi
        if self.line_cache.into_iter().any(|w| w != 0) {
            // Bit-by-bit iterator over `self.line_cache` array
            let mut cache_bit_itr = (0..WIDTH).into_iter().map(|x| {
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

#[cfg(test)]
mod tests {
    use super::{Ls013b7dh03, LINE_CACHE_DWORD_COUNT, LINE_TOTAL_BYTE_COUNT};
    use crate::reverse_bits;
    use core::convert::Infallible;
    use embedded_hal::{
        digital::{ErrorType as PinErrorType, InputPin, OutputPin, PinState},
        spi::{ErrorType as SpiErrorType, SpiBus},
    };
    use std::vec::Vec;

    #[test]
    fn xx() {
        let spi = SpiFixture {
            data_written: Vec::new(),
        };
        let cs_pin = PinFixture {
            state: PinState::High,
        };
        let com_in_pin = PinFixture {
            state: PinState::High,
        };

        let mut buffer: [u8; 2304] = [0; 2304];
        let line_cache: [u32; LINE_CACHE_DWORD_COUNT];
        let write_history;

        // start a scope so that rust allows us to destructure `disp` (and use `buffer`) after its end
        {
            let mut disp = Ls013b7dh03::new(spi, cs_pin, com_in_pin, &mut buffer);

            // Flush initial display state
            disp.flush();

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

            // Flush again
            disp.flush();

            // Save line cache for printing (destructure `disp`)
            line_cache = disp.line_cache;
            let spi = disp.spi;
            write_history = spi.data_written;
        }

        print_formatted_buffer(&buffer, &line_cache);

        for d in write_history {
            print_spi_lines(d.as_slice());
        }
    }

    fn print_spi_lines(data: &[u8]) {
        let mut lines_itr = data.chunks_exact(LINE_TOTAL_BYTE_COUNT).into_iter();

        println!("spi.write:");
        while let Some(csl) = lines_itr.next() {
            print!("\t");
            print_spi_line(csl);
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

    fn print_spi_line(line: &[u8]) {
        assert!(line.len() == LINE_TOTAL_BYTE_COUNT);

        let addr = reverse_bits(line[0]);

        print!("{: >3} [", addr);

        for b in line[1..(LINE_TOTAL_BYTE_COUNT - 1)].iter() {
            for i in 0..u8::BITS {
                if ((1u8 << i) & *b) != 0 {
                    print!(" ");
                } else {
                    print!("*");
                }
            }
        }

        print!("] fill=0x{:0>2x}", line[LINE_TOTAL_BYTE_COUNT - 1]);
    }

    fn print_formatted_buffer(buf: &[u8; 2304], cache: &[u32; LINE_CACHE_DWORD_COUNT]) {
        println!("\t  addr  ________________________________________________________________________________________________________________________________");
        for (i, sl) in buf.chunks_exact(LINE_TOTAL_BYTE_COUNT).enumerate() {
            let lc_byte_index = i / u32::BITS as usize;
            let lc_bit = i % u32::BITS as usize;
            let is_line_tagged = (cache[lc_byte_index] & (1u32 << lc_bit)) != 0;

            let addr = reverse_bits(sl[0]);

            print!(
                "\t{} {: >3} [",
                if is_line_tagged { "->" } else { "  " },
                addr
            );

            for b in sl[1..(LINE_TOTAL_BYTE_COUNT - 1)].iter() {
                for i in 0..u8::BITS {
                    if ((1u8 << i) & *b) != 0 {
                        print!(" ");
                    } else {
                        print!("*");
                    }
                }
            }

            println!("] fill=0x{:0>2x}", sl[LINE_TOTAL_BYTE_COUNT - 1]);
        }
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
}
