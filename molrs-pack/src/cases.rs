use std::error::Error;
use std::path::{Path, PathBuf};

use molrs_io::pdb::read_pdb_frame;

use crate::target::Target;
use crate::{
    AbovePlaneRestraint, BelowPlaneRestraint, InsideBoxRestraint, InsideSphereRestraint,
    OutsideSphereRestraint,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExampleCase {
    Mixture,
    Bilayer,
    Interface,
    Solvprotein,
    Spherical,
}

impl ExampleCase {
    pub const ALL: [ExampleCase; 5] = [
        ExampleCase::Mixture,
        ExampleCase::Bilayer,
        ExampleCase::Interface,
        ExampleCase::Solvprotein,
        ExampleCase::Spherical,
    ];

    pub fn name(self) -> &'static str {
        match self {
            ExampleCase::Mixture => "pack_mixture",
            ExampleCase::Bilayer => "pack_bilayer",
            ExampleCase::Interface => "pack_interface",
            ExampleCase::Solvprotein => "pack_solvprotein",
            ExampleCase::Spherical => "pack_spherical",
        }
    }

    pub fn output_xyz(self) -> &'static str {
        match self {
            ExampleCase::Mixture => "mixture.xyz",
            ExampleCase::Bilayer => "bilayer.xyz",
            ExampleCase::Interface => "interface.xyz",
            ExampleCase::Solvprotein => "solvprotein.xyz",
            ExampleCase::Spherical => "spherical.xyz",
        }
    }

    pub fn max_loops(self) -> usize {
        match self {
            ExampleCase::Mixture => 400,
            ExampleCase::Bilayer => 800,
            ExampleCase::Interface => 400,
            ExampleCase::Solvprotein => 800,
            ExampleCase::Spherical => 800,
        }
    }

    pub fn seed(self) -> u64 {
        match self {
            ExampleCase::Mixture => 1_234_567,
            ExampleCase::Bilayer => 1_234_567,
            ExampleCase::Interface => 1_234_567,
            ExampleCase::Solvprotein => 1_234_567,
            ExampleCase::Spherical => 1_234_567,
        }
    }
}

pub fn example_dir_from_manifest(case: ExampleCase) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join(case.name())
}

pub fn build_targets(case: ExampleCase, base: &Path) -> Result<Vec<Target>, Box<dyn Error>> {
    let targets = match case {
        ExampleCase::Mixture => {
            let water = read_pdb_frame(base.join("water.pdb"))?;
            let urea = read_pdb_frame(base.join("urea.pdb"))?;
            let box_restraint = InsideBoxRestraint::cube_from_origin([0.0, 0.0, 0.0], 40.0);
            vec![
                Target::new(water, 1000)
                    .with_restraint(box_restraint)
                    .with_name("water"),
                Target::new(urea, 400)
                    .with_restraint(box_restraint)
                    .with_name("urea"),
            ]
        }
        ExampleCase::Bilayer => {
            let water = read_pdb_frame(base.join("water.pdb"))?;
            let lipid = read_pdb_frame(base.join("palmitoil.pdb"))?;
            vec![
                Target::new(water.clone(), 50)
                    .with_restraint(InsideBoxRestraint::new(
                        [0.0, 0.0, -10.0],
                        [40.0, 40.0, 0.0],
                    ))
                    .with_name("water_low"),
                Target::new(water, 50)
                    .with_restraint(InsideBoxRestraint::new(
                        [0.0, 0.0, 28.0],
                        [40.0, 40.0, 38.0],
                    ))
                    .with_name("water_high"),
                Target::new(lipid.clone(), 10)
                    .with_restraint(InsideBoxRestraint::new([0.0, 0.0, 0.0], [40.0, 40.0, 14.0]))
                    .with_restraint_for_atoms(
                        &[31, 32],
                        BelowPlaneRestraint::new([0.0, 0.0, 1.0], 2.0),
                    )
                    .with_restraint_for_atoms(
                        &[1, 2],
                        AbovePlaneRestraint::new([0.0, 0.0, 1.0], 12.0),
                    )
                    .with_name("lipid_low"),
                Target::new(lipid, 10)
                    .with_restraint(InsideBoxRestraint::new(
                        [0.0, 0.0, 14.0],
                        [40.0, 40.0, 28.0],
                    ))
                    .with_restraint_for_atoms(
                        &[1, 2],
                        BelowPlaneRestraint::new([0.0, 0.0, 1.0], 16.0),
                    )
                    .with_restraint_for_atoms(
                        &[31, 32],
                        AbovePlaneRestraint::new([0.0, 0.0, 1.0], 26.0),
                    )
                    .with_name("lipid_high"),
            ]
        }
        ExampleCase::Interface => {
            let water = read_pdb_frame(base.join("water.pdb"))?;
            let chloroform = read_pdb_frame(base.join("chloroform.pdb"))?;
            let t3 = read_pdb_frame(base.join("t3.pdb"))?;
            vec![
                Target::new(water, 100)
                    .with_restraint(InsideBoxRestraint::new(
                        [-20.0, 0.0, 0.0],
                        [0.0, 39.0, 39.0],
                    ))
                    .with_name("water"),
                Target::new(chloroform, 30)
                    .with_restraint(InsideBoxRestraint::new([0.0, 0.0, 0.0], [21.0, 39.0, 39.0]))
                    .with_name("chloroform"),
                Target::new(t3, 1)
                    .with_name("t3")
                    .with_center()
                    .fixed_at_with_euler([0.0, 20.0, 20.0], [1.57, 1.57, 1.57]),
            ]
        }
        ExampleCase::Solvprotein => {
            let protein = read_pdb_frame(base.join("protein.pdb"))?;
            let water = read_pdb_frame(base.join("water.pdb"))?;
            let sodium = read_pdb_frame(base.join("sodium.pdb"))?;
            let chloride = read_pdb_frame(base.join("chloride.pdb"))?;
            let sphere = InsideSphereRestraint::new([0.0, 0.0, 0.0], 50.0);
            vec![
                Target::new(protein, 1)
                    .with_name("protein")
                    .with_center()
                    .fixed_at([0.0, 0.0, 0.0]),
                Target::new(water, 1000)
                    .with_restraint(sphere)
                    .with_name("water"),
                Target::new(sodium, 30)
                    .with_restraint(sphere)
                    .with_name("sodium"),
                Target::new(chloride, 20)
                    .with_restraint(sphere)
                    .with_name("chloride"),
            ]
        }
        ExampleCase::Spherical => {
            let water = read_pdb_frame(base.join("water.pdb"))?;
            let lipid = read_pdb_frame(base.join("palmitoil.pdb"))?;
            let origin = [0.0, 0.0, 0.0];
            vec![
                Target::new(water.clone(), 308)
                    .with_restraint(InsideSphereRestraint::new(origin, 13.0))
                    .with_name("water_inner"),
                Target::new(lipid.clone(), 90)
                    .with_restraint_for_atoms(&[37], InsideSphereRestraint::new(origin, 14.0))
                    .with_restraint_for_atoms(&[5], OutsideSphereRestraint::new(origin, 26.0))
                    .with_name("lipid_inner"),
                Target::new(lipid, 300)
                    .with_restraint_for_atoms(&[5], InsideSphereRestraint::new(origin, 29.0))
                    .with_restraint_for_atoms(&[37], OutsideSphereRestraint::new(origin, 41.0))
                    .with_name("lipid_outer"),
                Target::new(water, 17536)
                    .with_restraint(InsideBoxRestraint::new(
                        [-47.5, -47.5, -47.5],
                        [47.5, 47.5, 47.5],
                    ))
                    .with_restraint(OutsideSphereRestraint::new(origin, 43.0))
                    .with_name("water_outer"),
            ]
        }
    };

    Ok(targets)
}

pub fn render_packmol_input(case: ExampleCase, base: &Path, output: &Path, seed: u64) -> String {
    let water = base.join("water.pdb");
    let lipid = base.join("palmitoil.pdb");
    let urea = base.join("urea.pdb");
    let chloro = base.join("chloroform.pdb");
    let t3 = base.join("t3.pdb");
    let protein = base.join("protein.pdb");
    let sodium = base.join("sodium.pdb");
    let chloride = base.join("chloride.pdb");

    match case {
        ExampleCase::Mixture => format!(
            "tolerance 2.0\nseed {seed}\nfiletype pdb\noutput {}\n\n\
structure {}\n  number 1000\n  inside box 0. 0. 0. 40. 40. 40.\nend structure\n\n\
structure {}\n  number 400\n  inside box 0. 0. 0. 40. 40. 40.\nend structure\n",
            output.display(),
            water.display(),
            urea.display()
        ),
        ExampleCase::Bilayer => format!(
            "tolerance 2.0\nseed {seed}\nfiletype pdb\noutput {}\n\n\
structure {}\n  number 50\n  inside box 0. 0. -10. 40. 40. 0.\nend structure\n\n\
structure {}\n  number 50\n  inside box 0. 0. 28. 40. 40. 38.\nend structure\n\n\
structure {}\n  number 10\n  inside box 0. 0. 0. 40. 40. 14.\n  atoms 31 32\n    below plane 0. 0. 1. 2.\n  end atoms\n  atoms 1 2\n    over plane 0. 0. 1. 12.\n  end atoms\nend structure\n\n\
structure {}\n  number 10\n  inside box 0. 0. 14. 40. 40. 28.\n  atoms 1 2\n    below plane 0. 0. 1. 16.\n  end atoms\n  atoms 31 32\n    over plane 0. 0. 1. 26.\n  end atoms\nend structure\n",
            output.display(),
            water.display(),
            water.display(),
            lipid.display(),
            lipid.display()
        ),
        ExampleCase::Interface => format!(
            "tolerance 2.0\nseed {seed}\nfiletype pdb\noutput {}\n\n\
structure {}\n  number 100\n  inside box -20. 0. 0. 0. 39. 39.\nend structure\n\n\
structure {}\n  number 30\n  inside box 0. 0. 0. 21. 39. 39.\nend structure\n\n\
structure {}\n  number 1\n  center\n  fixed 0. 20. 20. 1.57 1.57 1.57\nend structure\n",
            output.display(),
            water.display(),
            chloro.display(),
            t3.display()
        ),
        ExampleCase::Solvprotein => format!(
            "tolerance 2.0\nseed {seed}\nfiletype pdb\noutput {}\n\n\
structure {}\n  number 1\n  center\n  fixed 0. 0. 0. 0. 0. 0.\nend structure\n\n\
structure {}\n  number 1000\n  inside sphere 0. 0. 0. 50.\nend structure\n\n\
structure {}\n  number 20\n  inside sphere 0. 0. 0. 50.\nend structure\n\n\
structure {}\n  number 30\n  inside sphere 0. 0. 0. 50.\nend structure\n\
avoid_overlap no\n",
            output.display(),
            protein.display(),
            water.display(),
            chloride.display(),
            sodium.display()
        ),
        ExampleCase::Spherical => format!(
            "tolerance 2.0\nseed {seed}\nfiletype pdb\noutput {}\n\n\
structure {}\n  number 308\n  inside sphere 0. 0. 0. 13.\nend structure\n\n\
structure {}\n  number 90\n  atoms 37\n    inside sphere 0. 0. 0. 14.\n  end atoms\n  atoms 5\n    outside sphere 0. 0. 0. 26.\n  end atoms\nend structure\n\n\
structure {}\n  number 300\n  atoms 5\n    inside sphere 0. 0. 0. 29.\n  end atoms\n  atoms 37\n    outside sphere 0. 0. 0. 41.\n  end atoms\nend structure\n\n\
structure {}\n  number 17536\n  inside box -47.5 -47.5 -47.5 47.5 47.5 47.5\n  outside sphere 0. 0. 0. 43.\nend structure\n",
            output.display(),
            water.display(),
            lipid.display(),
            lipid.display(),
            water.display()
        ),
    }
}
