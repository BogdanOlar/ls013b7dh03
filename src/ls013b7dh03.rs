use embedded_hal::{digital::OutputPin, spi::SpiBus};

const WIDTH: usize = 128;
const HEIGHT: usize = 128;

const LINE_WIDTH_BYTE_COUNT: usize = WIDTH / (u8::BITS as usize);
const LINE_PADDING_BYTE_COUNT: usize = 1;
const LINE_ADDRESS_BYTE_COUNT: usize = 1;
const LINE_TOTAL_BYTE_COUNT: usize =
    LINE_ADDRESS_BYTE_COUNT + LINE_WIDTH_BYTE_COUNT + LINE_PADDING_BYTE_COUNT;

const BUF_SIZE: usize = HEIGHT * LINE_TOTAL_BYTE_COUNT;

pub struct Ls013b7dh03<'a, SPI, CS, CI> {
    spi: SPI,
    cs_pin: CS,
    com_in_pin: CI,
    buffer: &'a mut [u8; BUF_SIZE],
}

impl<'a, SPI, CS, DISP> Ls013b7dh03<'a, SPI, CS, DISP>
where
    SPI: SpiBus,
    CS: OutputPin,
    DISP: OutputPin,
{
    /// Create an Ls013b7dh03 display
    pub fn new(
        spi: SPI,
        mut cs_pin: CS,
        mut com_in_pin: DISP,
        buffer: &'a mut [u8; BUF_SIZE],
    ) -> Self {
        let _ = com_in_pin.set_low();
        let _ = cs_pin.set_low();

        // TODO: Initialize buffer

        Self {
            spi,
            cs_pin,
            com_in_pin,
            buffer,
        }
    }

    pub fn enable(&mut self) {
        let _ = self.com_in_pin.set_high();
    }

    pub fn disable(&mut self) {
        let _ = self.com_in_pin.set_low();
    }
}
