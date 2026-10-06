//! The dimension of a force-field parameter: exponents of energy, length,
//! angle, charge and mass.
//!
//! A [`Dim`] is what lets an engine codec convert a parameter it has never
//! seen: `E/L^2` scales by `energy / length²` whatever the parameter is
//! called, so the conversion is per dimension, not per style.
//!
//! The angle exponent has a meaning of its own (`ff-ir-02-protocol` D2):
//! exactly `A` is an angle **value**, stored in `units.angle` (the degree in
//! every preset); a **negative** angle exponent is per **radian** and never
//! converted (`angle harmonic` `k` is `E/A^2`, LAMMPS's energy/rad²); any
//! other positive angle exponent is refused ([`Dim::check`]).

use std::fmt;
use std::str::FromStr;

/// Exponents of energy (`E`), length (`L`), angle (`A`), charge (`Q`) and
/// mass (`M`).
///
/// Parsed from and printed as `num ("/" factor)*`, `num` being `1` or
/// factors joined by `*`, a factor a symbol with an optional positive power:
/// `"E/L^2"`, `"E*L/Q^2"`, `"A"`, `"E/A^2"`, `"1/L"`, `"1"`. Each symbol
/// appears at most once.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Dim {
    pub energy: i8,
    pub length: i8,
    pub angle: i8,
    pub charge: i8,
    pub mass: i8,
}

const SYMBOLS: [char; 5] = ['E', 'L', 'A', 'Q', 'M'];

impl Dim {
    /// A pure number (a periodicity, a sign, a dielectric constant).
    pub const NONE: Dim = Dim::new(0, 0, 0, 0, 0);
    pub const ENERGY: Dim = Dim::new(1, 0, 0, 0, 0);
    pub const LENGTH: Dim = Dim::new(0, 1, 0, 0, 0);
    /// An angle value, in `units.angle` (degrees).
    pub const ANGLE: Dim = Dim::new(0, 0, 1, 0, 0);
    pub const CHARGE: Dim = Dim::new(0, 0, 0, 1, 0);
    pub const MASS: Dim = Dim::new(0, 0, 0, 0, 1);

    /// `E^energy · L^length · A^angle · Q^charge · M^mass`.
    pub const fn new(energy: i8, length: i8, angle: i8, charge: i8, mass: i8) -> Self {
        Self {
            energy,
            length,
            angle,
            charge,
            mass,
        }
    }

    fn exponents(self) -> [i8; 5] {
        [self.energy, self.length, self.angle, self.charge, self.mass]
    }

    fn slot(&mut self, symbol: char) -> &mut i8 {
        match symbol {
            'E' => &mut self.energy,
            'L' => &mut self.length,
            'A' => &mut self.angle,
            'Q' => &mut self.charge,
            _ => &mut self.mass,
        }
    }

    /// Whether this is an angle value (exactly `A`), stored in degrees.
    pub fn is_angle_value(self) -> bool {
        self == Self::ANGLE
    }

    /// The angle rule: a positive angle exponent only as exactly `A`.
    pub fn check(self) -> Result<(), String> {
        if self.angle > 0 && !self.is_angle_value() {
            return Err(
                "a positive angle exponent is an angle value, which is exactly `A`; a force \
                 constant per radian is a negative one (`E/A^2`)"
                    .into(),
            );
        }
        Ok(())
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let factor = |s: char, p: i8| {
            if p == 1 {
                s.to_string()
            } else {
                format!("{s}^{p}")
            }
        };
        let e = self.exponents();
        let num: Vec<String> = (0..5)
            .filter(|&i| e[i] > 0)
            .map(|i| factor(SYMBOLS[i], e[i]))
            .collect();
        if num.is_empty() {
            f.write_str("1")?;
        } else {
            f.write_str(&num.join("*"))?;
        }
        for i in (0..5).filter(|&i| e[i] < 0) {
            write!(f, "/{}", factor(SYMBOLS[i], -e[i]))?;
        }
        Ok(())
    }
}

impl FromStr for Dim {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bad = |why: &str| format!("dimension {s:?}: {why}");
        let mut parts = s.split('/');
        let num = parts.next().unwrap_or_default();
        let mut dim = Dim::NONE;
        let mut seen = Vec::new();
        let mut factor = |text: &str, sign: i8, dim: &mut Dim| -> Result<(), String> {
            let mut chars = text.chars();
            let symbol = chars
                .next()
                .filter(|c| SYMBOLS.contains(c))
                .ok_or_else(|| bad("a factor is one of E, L, A, Q, M"))?;
            let rest = chars.as_str();
            let power: i8 = match rest.strip_prefix('^') {
                None if rest.is_empty() => 1,
                Some(p)
                    if !p.is_empty()
                        && !p.starts_with('0')
                        && p.bytes().all(|b| b.is_ascii_digit()) =>
                {
                    p.parse().map_err(|_| bad("power out of range"))?
                }
                _ => return Err(bad("a power is `^` and a positive integer")),
            };
            if seen.contains(&symbol) {
                return Err(bad("each symbol at most once"));
            }
            seen.push(symbol);
            *dim.slot(symbol) = sign * power;
            Ok(())
        };
        if num != "1" {
            for text in num.split('*') {
                factor(text, 1, &mut dim)?;
            }
        }
        for text in parts {
            factor(text, -1, &mut dim)?;
        }
        Ok(dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_the_spellings_the_ir_uses() {
        assert_eq!("E/L^2".parse::<Dim>(), Ok(Dim::new(1, -2, 0, 0, 0)));
        assert_eq!("E*L/Q^2".parse::<Dim>(), Ok(Dim::new(1, 1, 0, -2, 0)));
        assert_eq!("A".parse::<Dim>(), Ok(Dim::ANGLE));
        assert_eq!("E/A^2".parse::<Dim>(), Ok(Dim::new(1, 0, -2, 0, 0)));
        assert_eq!("1".parse::<Dim>(), Ok(Dim::NONE));
        assert_eq!("1/L".parse::<Dim>(), Ok(Dim::new(0, -1, 0, 0, 0)));
        assert_eq!("E*L^6".parse::<Dim>(), Ok(Dim::new(1, 6, 0, 0, 0)));
        assert_eq!("M".parse::<Dim>(), Ok(Dim::MASS));
        assert_eq!("E/L/A".parse::<Dim>(), Ok(Dim::new(1, -1, -1, 0, 0)));
    }

    #[test]
    fn display_reads_back_to_itself() {
        for text in [
            "1", "E", "E/L^2", "1/L", "E*L^6", "L^3", "E*L/Q^2", "A", "E/A^2", "M", "Q",
        ] {
            let d: Dim = text.parse().unwrap();
            assert_eq!(d.to_string(), text);
            assert_eq!(d.to_string().parse::<Dim>(), Ok(d));
        }
        assert_eq!(Dim::new(-1, 3, -2, 0, 1).to_string(), "L^3*M/E/A^2");
    }

    #[test]
    fn refuses_what_is_not_a_dimension() {
        for bad in [
            "", "K", "E/", "E**L", "E^x", "E^0", "E^-2", "E+L", "E*E", "L/L", "1*E", " E",
        ] {
            assert!(bad.parse::<Dim>().is_err(), "{bad:?} parsed");
        }
    }

    #[test]
    fn only_exactly_a_is_a_positive_angle() {
        assert!(Dim::ANGLE.check().is_ok());
        assert!("E/A^2".parse::<Dim>().unwrap().check().is_ok());
        assert!("A^2".parse::<Dim>().unwrap().check().is_err());
        assert!("E*A".parse::<Dim>().unwrap().check().is_err());
    }
}
