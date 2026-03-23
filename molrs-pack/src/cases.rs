use std::error::Error;
use std::path::{Path, PathBuf};

use molrs::io::pdb::read_pdb_frame;

use crate::constraint::RegionConstraint;
use crate::target::Target;
use crate::{
    AbovePlaneConstraint, BelowPlaneConstraint, InsideBoxConstraint, InsideSphereConstraint,
    OutsideSphereConstraint,
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
            ExampleCase::Mixture => 200,
            ExampleCase::Bilayer => 300,
            ExampleCase::Interface => 200,
            ExampleCase::Solvprotein => 200,
            ExampleCase::Spherical => 300,
        }
    }

    pub fn seed(self) -> u64 {
        match self {
            ExampleCase::Mixture => 7,
            ExampleCase::Bilayer => 2026,
            ExampleCase::Interface => 42,
            ExampleCase::Solvprotein => 42,
            ExampleCase::Spherical => 2026,
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
            let box_constraint = InsideBoxConstraint::cube_from_origin(40.0, [0.0, 0.0, 0.0]);
            vec![
                Target::new(water, 1000)
                    .with_constraint(box_constraint.clone())
                    .with_name("water"),
                Target::new(urea, 400)
                    .with_constraint(box_constraint)
                    .with_name("urea"),
            ]
        }
        ExampleCase::Bilayer => {
            let water = read_pdb_frame(base.join("water.pdb"))?;
            let lipid = read_pdb_frame(base.join("palmitoil.pdb"))?;
            vec![
                Target::new(water.clone(), 500)
                    .with_constraint(InsideBoxConstraint::new(
                        [0.0, 0.0, -10.0],
                        [40.0, 40.0, 0.0],
                    ))
                    .with_name("water_low"),
                Target::new(water, 500)
                    .with_constraint(InsideBoxConstraint::new(
                        [0.0, 0.0, 28.0],
                        [40.0, 40.0, 38.0],
                    ))
                    .with_name("water_high"),
                Target::new(lipid.clone(), 50)
                    .with_constraint(InsideBoxConstraint::new(
                        [0.0, 0.0, 0.0],
                        [40.0, 40.0, 14.0],
                    ))
                    .with_constraint_for_atoms(
                        &[31, 32],
                        BelowPlaneConstraint::new([0.0, 0.0, 1.0], 2.0),
                    )
                    .with_constraint_for_atoms(
                        &[1, 2],
                        AbovePlaneConstraint::new([0.0, 0.0, 1.0], 12.0),
                    )
                    .with_name("lipid_low"),
                Target::new(lipid, 50)
                    .with_constraint(InsideBoxConstraint::new(
                        [0.0, 0.0, 14.0],
                        [40.0, 40.0, 28.0],
                    ))
                    .with_constraint_for_atoms(
                        &[1, 2],
                        BelowPlaneConstraint::new([0.0, 0.0, 1.0], 16.0),
                    )
                    .with_constraint_for_atoms(
                        &[31, 32],
                        AbovePlaneConstraint::new([0.0, 0.0, 1.0], 26.0),
                    )
                    .with_name("lipid_high"),
            ]
        }
        ExampleCase::Interface => {
            let water = read_pdb_frame(base.join("water.pdb"))?;
            let chloroform = read_pdb_frame(base.join("chloroform.pdb"))?;
            let t3 = read_pdb_frame(base.join("t3.pdb"))?;
            vec![
                Target::new(water, 200)
                    .with_constraint(InsideBoxConstraint::new(
                        [-20.0, 0.0, 0.0],
                        [0.0, 39.0, 39.0],
                    ))
                    .with_name("water"),
                Target::new(chloroform, 50)
                    .with_constraint(InsideBoxConstraint::new(
                        [0.0, 0.0, 0.0],
                        [21.0, 39.0, 39.0],
                    ))
                    .with_name("chloroform"),
                Target::new(t3, 1)
                    .with_name("t3")
                    .with_center_of_mass()
                    .fixed_at_with_euler([0.0, 20.0, 20.0], [0.0, 0.0, 0.0]),
            ]
        }
        ExampleCase::Solvprotein => {
            let protein = read_pdb_frame(base.join("protein.pdb"))?;
            let water = read_pdb_frame(base.join("water.pdb"))?;
            let sodium = read_pdb_frame(base.join("sodium.pdb"))?;
            let chloride = read_pdb_frame(base.join("chloride.pdb"))?;
            let sphere = InsideSphereConstraint::new(50.0, [0.0, 0.0, 0.0]);
            vec![
                Target::new(protein, 1)
                    .with_name("protein")
                    .with_center_of_mass()
                    .fixed_at([0.0, 0.0, 0.0]),
                Target::new(water, 500)
                    .with_constraint(sphere.clone())
                    .with_name("water"),
                Target::new(sodium, 30)
                    .with_constraint(sphere.clone())
                    .with_name("sodium"),
                Target::new(chloride, 20)
                    .with_constraint(sphere)
                    .with_name("chloride"),
            ]
        }
        ExampleCase::Spherical => {
            let water = read_pdb_frame(base.join("water.pdb"))?;
            let lipid = read_pdb_frame(base.join("palmitoil.pdb"))?;
            let origin = [0.0, 0.0, 0.0];
            vec![
                Target::new(water.clone(), 308)
                    .with_constraint(InsideSphereConstraint::new(13.0, origin))
                    .with_name("water_inner"),
                Target::new(lipid.clone(), 90)
                    .with_constraint_for_atoms(&[37], InsideSphereConstraint::new(14.0, origin))
                    .with_constraint_for_atoms(&[5], OutsideSphereConstraint::new(26.0, origin))
                    .with_name("lipid_inner"),
                Target::new(lipid, 300)
                    .with_constraint_for_atoms(&[5], InsideSphereConstraint::new(29.0, origin))
                    .with_constraint_for_atoms(&[37], OutsideSphereConstraint::new(41.0, origin))
                    .with_name("lipid_outer"),
                Target::new(water, 17536)
                    .with_constraint(
                        InsideBoxConstraint::new([-47.5, -47.5, -47.5], [47.5, 47.5, 47.5])
                            .and(OutsideSphereConstraint::new(43.0, origin)),
                    )
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
structure {}\n  number 500\n  inside box 0. 0. -10. 40. 40. 0.\nend structure\n\n\
structure {}\n  number 500\n  inside box 0. 0. 28. 40. 40. 38.\nend structure\n\n\
structure {}\n  number 50\n  inside box 0. 0. 0. 40. 40. 14.\n  atoms 31 32\n    below plane 0. 0. 1. 2.\n  end atoms\n  atoms 1 2\n    over plane 0. 0. 1. 12.\n  end atoms\nend structure\n\n\
structure {}\n  number 50\n  inside box 0. 0. 14. 40. 40. 28.\n  atoms 1 2\n    below plane 0. 0. 1. 16.\n  end atoms\n  atoms 31 32\n    over plane 0. 0. 1. 26.\n  end atoms\nend structure\n",
            output.display(),
            water.display(),
            water.display(),
            lipid.display(),
            lipid.display()
        ),
        ExampleCase::Interface => format!(
            "tolerance 2.0\nseed {seed}\nfiletype pdb\noutput {}\n\n\
structure {}\n  number 200\n  inside box -20. 0. 0. 0. 39. 39.\nend structure\n\n\
structure {}\n  number 50\n  inside box 0. 0. 0. 21. 39. 39.\nend structure\n\n\
structure {}\n  number 1\n  centerofmass\n  fixed 0. 20. 20. 0. 0. 0.\nend structure\n",
            output.display(),
            water.display(),
            chloro.display(),
            t3.display()
        ),
        ExampleCase::Solvprotein => format!(
            "tolerance 2.0\nseed {seed}\nfiletype pdb\noutput {}\n\n\
structure {}\n  number 500\n  inside sphere 0. 0. 0. 50.\nend structure\n\n\
structure {}\n  number 30\n  inside sphere 0. 0. 0. 50.\nend structure\n\n\
structure {}\n  number 20\n  inside sphere 0. 0. 0. 50.\nend structure\n\n\
structure {}\n  number 1\n  centerofmass\n  fixed 0. 0. 0. 0. 0. 0.\nend structure\n",
            output.display(),
            water.display(),
            sodium.display(),
            chloride.display(),
            protein.display()
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
