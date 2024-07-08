# LS013B7DH03

[![crates.io](https://img.shields.io/crates/v/ls013b7dh03)](https://crates.io/crates/ls013b7dh03)
[![docs.rs](https://docs.rs/ls013b7dh03/badge.svg)](https://docs.rs/ls013b7dh03)
[![build and test](https://github.com/BogdanOlar/ls013b7dh03/actions/workflows/rust.yml/badge.svg)](https://github.com/BogdanOlar/ls013b7dh03/actions/workflows/rust.yml)

LS013B7DH03 Sharp LCD driver for [embedded-hal v1.0](https://github.com/rust-embedded/embedded-hal)

Simply add the driver to your `Cargo.toml:

```toml
ls013b7dh03 = { version = "0.4" }
```

Use:

```rs
    use ls013b7dh03::prelude::*;

    let mut buffer = [0u8; BUF_SIZE];
    let mut display = Ls013b7dh03::new(spi, cs, disp_com, &mut buffer);

    for x in 10..100 {
        let _ = display.write(x, 10, true);
    }

    display.flush();
```

An optional implementation for the [embeddded-graphics](https://github.com/embedded-graphics/embedded-graphics) 2D graphics library is provided by the optional `embdded_graphics` feature:

```toml
ls013b7dh03 = { version = "0.4", features = ["embedded_graphics"] }
```

Use:

```rs
    use ls013b7dh03::prelude::*;
    use embedded_graphics::{
        geometry::Point,
        pixelcolor::BinaryColor,
        prelude::*,
        primitives::{Circle, Primitive, PrimitiveStyle},
    };

    let mut buffer = [0u8; BUF_SIZE];
    let mut display = Ls013b7dh03::new(spi, cs, disp_com, &mut buffer);

    let circle = Circle::new(Point::new(50, 50), 50).into_styled(PrimitiveStyle::with_stroke(BinaryColor::On, 2));
    let _ = circle.draw(&mut display);

    display.flush();
```
