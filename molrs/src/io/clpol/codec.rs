//! Reader of a CL&Pol `alpha.ff` polarisation table.

use std::path::Path;

/// One `alpha.ff` row read from a file; fields and units as
/// [`ClpolPolarizability`](crate::ff::params::ClpolPolarizability).
#[derive(Debug, Clone, PartialEq)]
pub struct ClpolAlphaRow {
    /// Atom type.
    pub type_name: String,
    /// Drude particle mass, u.
    pub m_d: f64,
    /// Sign of the Drude charge.
    pub q_d_sign: f64,
    /// Drude spring constant (`k/2 r²` form), kJ·mol⁻¹·Å⁻².
    pub k_d: f64,
    /// Polarisability, Å³.
    pub alpha: f64,
    /// Thole damping parameter.
    pub a_thole: f64,
}

/// Parse `alpha.ff` text, rows in file order (a type given twice keeps both
/// rows; the later one is the file's last word).
///
/// The file is whitespace-separated rows `type m_D q_D k_D alpha a_thole`;
/// `#` starts a comment, and a line with fewer than six fields (blank, a
/// header) is not a row. molrs ships the paduagroup/clandpol table itself as
/// [`CLPOL_POLARIZABILITY`](crate::ff::params::CLPOL_POLARIZABILITY); this
/// reads a caller's own (an edited or newer `alpha.ff`) into the same row
/// shape, with owned type names.
///
/// # Errors
///
/// `Err` naming the line for a row whose numeric fields do not parse.
pub fn read_clpol_alpha_str(text: &str) -> Result<Vec<ClpolAlphaRow>, String> {
    let mut rows = Vec::new();
    for (lineno, raw) in text.lines().enumerate() {
        let fields: Vec<&str> = raw
            .split('#')
            .next()
            .unwrap_or("")
            .split_whitespace()
            .collect();
        if fields.len() < 6 {
            continue;
        }
        let num = |i: usize| {
            fields[i].parse::<f64>().map_err(|_| {
                format!(
                    "alpha.ff line {}: field {} {:?} is not a number",
                    lineno + 1,
                    i + 1,
                    fields[i]
                )
            })
        };
        rows.push(ClpolAlphaRow {
            type_name: fields[0].to_owned(),
            m_d: num(1)?,
            q_d_sign: num(2)?,
            k_d: num(3)?,
            alpha: num(4)?,
            a_thole: num(5)?,
        });
    }
    Ok(rows)
}

/// Read an `alpha.ff` file; see [`read_clpol_alpha_str`].
///
/// # Errors
///
/// The file's I/O error, or [`read_clpol_alpha_str`]'s.
pub fn read_clpol_alpha(path: impl AsRef<Path>) -> Result<Vec<ClpolAlphaRow>, String> {
    let path = path.as_ref();
    let text =
        std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    read_clpol_alpha_str(&text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::params::CLPOL_POLARIZABILITY;

    #[test]
    fn rows_comments_and_headers() {
        let text = "# alpha.ff\n# type m_D q_D k_D alpha a_thole\nNBT 0.4 -1.0 4184.0 1.698 2.6 # imide N\n\nHC 0.0 0.0 0.0 0.323 0.0\n";
        let rows = read_clpol_alpha_str(text).unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].type_name, "NBT");
        assert_eq!(rows[0].alpha, 1.698);
        assert_eq!(rows[1].k_d, 0.0);
    }

    #[test]
    fn a_bad_number_names_its_line() {
        let err = read_clpol_alpha_str("X 0.4 -1 big 1 2.6\n").unwrap_err();
        assert!(err.contains("line 1"), "{err}");
    }

    #[test]
    fn the_shipped_table_reads_back_through_the_text_form() {
        let text: String = CLPOL_POLARIZABILITY
            .iter()
            .map(|r| {
                format!(
                    "{} {} {} {} {} {}\n",
                    r.type_name, r.m_d, r.q_d_sign, r.k_d, r.alpha, r.a_thole
                )
            })
            .collect();
        let rows = read_clpol_alpha_str(&text).unwrap();
        assert_eq!(rows.len(), CLPOL_POLARIZABILITY.len());
        for (row, want) in rows.iter().zip(CLPOL_POLARIZABILITY) {
            assert_eq!(row.type_name, want.type_name);
            assert_eq!((row.alpha, row.k_d), (want.alpha, want.k_d));
        }
    }
}
