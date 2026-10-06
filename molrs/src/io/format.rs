//! One door for every single-structure file: [`read_frame`] / [`write_frame`]
//! pick the format from the file name (or an explicit format name) and hand
//! off to that format's own reader or writer.
//!
//! The per-format functions under [`crate::io::data`] and
//! [`crate::io::trajectory`] stay the API for format-specific options (a dump's
//! column list, a CIF's every block); this module only decides *which* of them
//! a path means, so a caller that loads "whatever file the user named" does not
//! keep its own extension table.
//!
//! | [`FrameFormat`] | name | file names | read | write |
//! |---|---|---|---|---|
//! | `Pdb` | `pdb` | `*.pdb`, `*.ent` | first `MODEL` | ✓ |
//! | `Xyz` | `xyz` | `*.xyz`, `*.extxyz` | first frame | ✓ |
//! | `Sdf` | `sdf` | `*.sdf`, `*.mol` | first record | — |
//! | `Mol2` | `mol2` | `*.mol2` | first molecule | ✓ |
//! | `Gro` | `gro` | `*.gro` | first frame | ✓ |
//! | `Cif` | `cif` | `*.cif` | first data block | ✓ |
//! | `Poscar` | `poscar` | `*.poscar`, `*.vasp`, `POSCAR*`, `CONTCAR*` | ✓ | ✓ |
//! | `Xsf` | `xsf` | `*.xsf` | ✓ | ✓ |
//! | `Cube` | `cube` | `*.cube`, `*.cub` | ✓ | ✓ |
//! | `Inpcrd` | `inpcrd` | `*.inpcrd`, `*.rst7`, `*.restrt`, `*.crd` | ✓ | — |
//! | `LammpsData` | `lammps_data` | `*.data`, `*.lmp` | ✓ | ✓ |
//! | `LammpsDump` | `lammps_dump` | `*.lammpstrj`, `*.dump` | first snapshot | ✓ (one snapshot) |
//!
//! A format name is matched case-insensitively and also accepts every
//! file-name extension in its row (`"lammpstrj"` is `lammps_dump`, `"mol"` is
//! `sdf`), so a script's `filetype` keyword and a file extension go through
//! the same table.

use std::fs::File;
use std::io::{BufReader, BufWriter, Error, ErrorKind, Result, Write};
use std::path::Path;

use molrs::store::Frame;

use crate::io::reader::FrameReader;

/// A single-structure file format [`read_frame`] / [`write_frame`] dispatch to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameFormat {
    /// Protein Data Bank.
    Pdb,
    /// XYZ / extended XYZ.
    Xyz,
    /// MDL SDF / MOL V2000.
    Sdf,
    /// Tripos MOL2.
    Mol2,
    /// GROMACS GRO.
    Gro,
    /// Crystallographic Information File.
    Cif,
    /// VASP POSCAR / CONTCAR.
    Poscar,
    /// XCrySDen structure file.
    Xsf,
    /// Gaussian cube.
    Cube,
    /// AMBER ASCII coordinates (inpcrd / restart).
    Inpcrd,
    /// LAMMPS data file.
    LammpsData,
    /// LAMMPS dump (`dump custom`) trajectory.
    LammpsDump,
}

impl FrameFormat {
    /// Every format, in the table order of the module documentation.
    pub const ALL: [FrameFormat; 12] = [
        FrameFormat::Pdb,
        FrameFormat::Xyz,
        FrameFormat::Sdf,
        FrameFormat::Mol2,
        FrameFormat::Gro,
        FrameFormat::Cif,
        FrameFormat::Poscar,
        FrameFormat::Xsf,
        FrameFormat::Cube,
        FrameFormat::Inpcrd,
        FrameFormat::LammpsData,
        FrameFormat::LammpsDump,
    ];

    /// The canonical format name (`"pdb"`, `"lammps_data"`, …).
    pub fn name(self) -> &'static str {
        match self {
            FrameFormat::Pdb => "pdb",
            FrameFormat::Xyz => "xyz",
            FrameFormat::Sdf => "sdf",
            FrameFormat::Mol2 => "mol2",
            FrameFormat::Gro => "gro",
            FrameFormat::Cif => "cif",
            FrameFormat::Poscar => "poscar",
            FrameFormat::Xsf => "xsf",
            FrameFormat::Cube => "cube",
            FrameFormat::Inpcrd => "inpcrd",
            FrameFormat::LammpsData => "lammps_data",
            FrameFormat::LammpsDump => "lammps_dump",
        }
    }

    /// The file-name extensions (lowercase, no dot) that name this format.
    pub fn extensions(self) -> &'static [&'static str] {
        match self {
            FrameFormat::Pdb => &["pdb", "ent"],
            FrameFormat::Xyz => &["xyz", "extxyz"],
            FrameFormat::Sdf => &["sdf", "mol"],
            FrameFormat::Mol2 => &["mol2"],
            FrameFormat::Gro => &["gro"],
            FrameFormat::Cif => &["cif"],
            FrameFormat::Poscar => &["poscar", "vasp"],
            FrameFormat::Xsf => &["xsf"],
            FrameFormat::Cube => &["cube", "cub"],
            FrameFormat::Inpcrd => &["inpcrd", "rst7", "restrt", "crd"],
            FrameFormat::LammpsData => &["data", "lmp"],
            FrameFormat::LammpsDump => &["lammpstrj", "dump"],
        }
    }

    /// Whether [`write_frame`] can write this format.
    pub fn is_writable(self) -> bool {
        !matches!(self, FrameFormat::Sdf | FrameFormat::Inpcrd)
    }

    /// The format a name means: a canonical name or any of its extensions,
    /// case-insensitively, with or without a leading dot.
    pub fn from_name(name: &str) -> Option<FrameFormat> {
        let name = name.trim().trim_start_matches('.').to_ascii_lowercase();
        Self::ALL
            .into_iter()
            .find(|f| f.name() == name || f.extensions().contains(&name.as_str()))
    }

    /// The format a path names: its extension, or — for the extension-less
    /// VASP convention — a file name starting with `POSCAR` or `CONTCAR`.
    pub fn from_path(path: &Path) -> Option<FrameFormat> {
        if let Some(format) = path.extension().and_then(|e| e.to_str()).and_then(|e| {
            Self::ALL
                .into_iter()
                .find(|f| f.extensions().contains(&e.to_ascii_lowercase().as_str()))
        }) {
            return Some(format);
        }
        let stem = path.file_name()?.to_str()?.to_ascii_uppercase();
        (stem.starts_with("POSCAR") || stem.starts_with("CONTCAR")).then_some(FrameFormat::Poscar)
    }

    /// The format to use for `path`: `format` when given (an unknown name is
    /// an error, not a fallback), else the one [`from_path`](Self::from_path)
    /// finds.
    ///
    /// # Errors
    ///
    /// [`ErrorKind::InvalidInput`] when `format` names no format, or when it
    /// is `None` and the path names none either.
    pub fn resolve(path: &Path, format: Option<&str>) -> Result<FrameFormat> {
        match format {
            Some(name) => Self::from_name(name).ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidInput,
                    format!(
                        "unknown structure format {name:?}; known: {}",
                        Self::known_names()
                    ),
                )
            }),
            None => Self::from_path(path).ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidInput,
                    format!(
                        "cannot tell the format of {} from its name; pass a format \
                         (one of {})",
                        path.display(),
                        Self::known_names()
                    ),
                )
            }),
        }
    }

    fn known_names() -> String {
        Self::ALL.map(FrameFormat::name).join(", ")
    }
}

impl std::fmt::Display for FrameFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

fn invalid_data(message: impl std::fmt::Display) -> Error {
    Error::new(ErrorKind::InvalidData, message.to_string())
}

fn empty(path: &Path, format: FrameFormat) -> Error {
    invalid_data(format!(
        "{}: the {format} file holds no structure",
        path.display()
    ))
}

/// Read one structure from `path`.
///
/// `format` is a format name or extension (see [`FrameFormat::from_name`]);
/// `None` picks the format from the file name. A multi-structure file
/// (trajectory, multi-model PDB, multi-record SDF, multi-block CIF) gives its
/// first structure; use the format's own reader for the rest.
///
/// # Errors
///
/// [`ErrorKind::InvalidInput`] when the format cannot be determined, the
/// format reader's error otherwise (an empty file is
/// [`ErrorKind::InvalidData`]).
pub fn read_frame(path: impl AsRef<Path>, format: Option<&str>) -> Result<Frame> {
    let path = path.as_ref();
    let format = FrameFormat::resolve(path, format)?;
    match format {
        FrameFormat::Pdb => crate::io::data::pdb::read_pdb_frame(path),
        FrameFormat::Xyz => crate::io::data::xyz::read_xyz_frame(path),
        FrameFormat::Sdf => {
            let mut reader =
                crate::io::data::sdf::SDFReader::new(BufReader::new(File::open(path)?));
            reader.read()?.ok_or_else(|| empty(path, format))
        }
        FrameFormat::Mol2 => crate::io::data::mol2::read_mol2(path),
        FrameFormat::Gro => crate::io::data::gro::read_gro(path)?
            .into_iter()
            .next()
            .ok_or_else(|| empty(path, format)),
        FrameFormat::Cif => crate::io::data::cif::read_cif(path),
        FrameFormat::Poscar => crate::io::data::poscar::read_poscar(path),
        FrameFormat::Xsf => crate::io::data::xsf::read_xsf(path),
        FrameFormat::Cube => crate::io::data::cube::read_cube(path).map_err(invalid_data),
        FrameFormat::Inpcrd => crate::io::data::inpcrd::read_amber_inpcrd(path),
        FrameFormat::LammpsData => crate::io::data::lammps_data::read_lammps_data(path),
        FrameFormat::LammpsDump => crate::io::trajectory::lammps_dump::open_lammps_dump(path)?
            .read()?
            .ok_or_else(|| empty(path, format)),
    }
}

/// Write `frame` to `path`, replacing any file there.
///
/// `format` is a format name or extension (see [`FrameFormat::from_name`]);
/// `None` picks the format from the file name. A LAMMPS dump is written as a
/// one-snapshot trajectory with every `atoms` column.
///
/// # Errors
///
/// [`ErrorKind::InvalidInput`] when the format cannot be determined or has no
/// writer ([`FrameFormat::is_writable`]); the format writer's error otherwise.
pub fn write_frame(path: impl AsRef<Path>, frame: &Frame, format: Option<&str>) -> Result<()> {
    let path = path.as_ref();
    let format = FrameFormat::resolve(path, format)?;
    if !format.is_writable() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            format!("molrs has no {format} writer"),
        ));
    }
    match format {
        FrameFormat::Pdb => {
            let mut w = BufWriter::new(File::create(path)?);
            crate::io::data::pdb::write_pdb_frame(&mut w, frame)?;
            w.flush()
        }
        FrameFormat::Xyz => {
            let mut w = BufWriter::new(File::create(path)?);
            crate::io::data::xyz::write_xyz_frame(&mut w, frame)?;
            w.flush()
        }
        FrameFormat::Mol2 => crate::io::data::mol2::write_mol2(path, frame),
        FrameFormat::Gro => crate::io::data::gro::write_gro(path, frame),
        FrameFormat::Cif => crate::io::data::cif::write_cif(path, frame),
        FrameFormat::Poscar => crate::io::data::poscar::write_poscar(path, frame),
        FrameFormat::Xsf => crate::io::data::xsf::write_xsf(path, frame),
        FrameFormat::Cube => crate::io::data::cube::write_cube(path, frame).map_err(invalid_data),
        FrameFormat::LammpsData => crate::io::data::lammps_data::write_lammps_data(path, frame),
        FrameFormat::LammpsDump => crate::io::trajectory::lammps_dump::write_lammps_dump(
            path,
            std::slice::from_ref(frame),
            None,
        ),
        FrameFormat::Sdf | FrameFormat::Inpcrd => unreachable!("refused above"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::op::types::F;
    use molrs::store::Block;
    use ndarray::Array1;

    fn water() -> Frame {
        let mut atoms = Block::new();
        let col = |v: &[F]| Array1::from_vec(v.to_vec()).into_dyn();
        atoms.insert("x", col(&[0.0, 0.9572, -0.24])).unwrap();
        atoms.insert("y", col(&[0.0, 0.0, 0.927])).unwrap();
        atoms.insert("z", col(&[0.0, 0.0, 0.0])).unwrap();
        atoms
            .insert(
                "element",
                Array1::from_vec(vec!["O".to_string(), "H".into(), "H".into()]).into_dyn(),
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.simbox = Some(
            molrs::spatial::SimBox::cube(10.0, ndarray::array![0.0, 0.0, 0.0], [true; 3]).unwrap(),
        );
        frame
    }

    #[test]
    fn names_and_extensions_resolve_case_insensitively() {
        assert_eq!(FrameFormat::from_name("PDB"), Some(FrameFormat::Pdb));
        assert_eq!(
            FrameFormat::from_name(".lammpstrj"),
            Some(FrameFormat::LammpsDump)
        );
        assert_eq!(FrameFormat::from_name("mol"), Some(FrameFormat::Sdf));
        assert_eq!(
            FrameFormat::from_name("lammps_data"),
            Some(FrameFormat::LammpsData)
        );
        assert_eq!(FrameFormat::from_name("docx"), None);
        for f in FrameFormat::ALL {
            assert_eq!(FrameFormat::from_name(f.name()), Some(f), "{f}");
            for ext in f.extensions() {
                assert_eq!(FrameFormat::from_name(ext), Some(f), "{ext}");
            }
        }
    }

    #[test]
    fn paths_resolve_by_extension_and_by_vasp_file_name() {
        assert_eq!(
            FrameFormat::from_path(Path::new("dir/Water.XYZ")),
            Some(FrameFormat::Xyz)
        );
        assert_eq!(
            FrameFormat::from_path(Path::new("run/CONTCAR")),
            Some(FrameFormat::Poscar)
        );
        assert_eq!(FrameFormat::from_path(Path::new("notes.txt")), None);
    }

    #[test]
    fn an_explicit_format_wins_over_the_extension() {
        let fmt = FrameFormat::resolve(Path::new("x.txt"), Some("xyz")).unwrap();
        assert_eq!(fmt, FrameFormat::Xyz);
        let err = FrameFormat::resolve(Path::new("x.xyz"), Some("nope")).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
        let err = FrameFormat::resolve(Path::new("x.txt"), None).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn every_writable_text_format_round_trips_its_atom_count() {
        let dir = std::env::temp_dir().join(format!("molrs-io-format-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (name, frame) in [
            ("w.pdb", water()),
            ("w.xyz", water()),
            ("w.mol2", water()),
            ("w.lammpstrj", water()),
        ] {
            let path = dir.join(name);
            write_frame(&path, &frame, None).unwrap_or_else(|e| panic!("{name}: {e}"));
            let back = read_frame(&path, None).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(back["atoms"].nrows(), Some(3), "{name}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn writing_a_read_only_format_is_refused() {
        let err = write_frame("w.sdf", &water(), None).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::InvalidInput);
    }

    #[test]
    fn a_dump_reads_its_first_snapshot() {
        let path =
            std::env::temp_dir().join(format!("molrs-io-format-{}.lammpstrj", std::process::id()));
        let mut second = water();
        second
            .get_mut("atoms")
            .unwrap()
            .insert("x", Array1::from_vec(vec![5.0 as F, 6.0, 7.0]).into_dyn())
            .unwrap();
        crate::io::trajectory::lammps_dump::write_lammps_dump(&path, &[water(), second], None)
            .unwrap();
        let frame = read_frame(&path, None);
        let _ = std::fs::remove_file(&path);
        let x = frame.unwrap()["atoms"]
            .get("x")
            .and_then(|c| c.as_float())
            .unwrap()
            .iter()
            .copied()
            .collect::<Vec<F>>();
        assert_eq!(x, [0.0, 0.9572, -0.24]);
    }
}
