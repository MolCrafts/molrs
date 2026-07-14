//! The shipped MMFF parameter set, assembled from the compiled table.
//!
//! `MMFF94Typifier::new()` and `MMFF94STypifier::new()` used to `include_str!` a
//! 68 KB XML each and re-parse it on every construction. Both now read the one
//! compiled table ([`crate::ff::params::mmff`]) and differ by exactly two things:
//! the force-field **name** they build under, and the
//! [`MmffVariant`](crate::ff::mmff::MmffVariant) their front door pins. Nothing
//! here is a number: every value comes from the table.
//!
//! The XML reader is not gone — [`MMFF94Typifier::from_xml_str`] still parses a
//! caller-supplied parameter set — but the *shipped* set is no longer text.

use std::collections::HashMap;

use crate::ff::forcefield::{ForceField, SpecialBonds};
use crate::ff::params::mmff::{MMFF_ELE_STYLE, MMFF_PROP, MMFF_STYLES, MMFF_VDW, MMFF_VDW_STYLE};
use crate::ff::potential::pair::mmff::encode_da_byte;

use super::params::{MMFFAtomProp, MMFFParams};

/// MMFF's vdW 1-4 weight.
///
/// Not a column of `<ElectrostaticParams>` and not an oversight: MMFF scales 1-4
/// **electrostatics** (by [`MmffEleStyle::scale14`](crate::ff::params::mmff::MmffEleStyle::scale14))
/// and leaves 1-4 vdW at full strength, because its torsion parameters were
/// fitted against unscaled 1-4 vdW. The next reader tempted to "fix" this to 0.5
/// by analogy with Amber would corrupt every MMFF vdW energy with a 1-4 pair.
const VDW_SCALE_14: f64 = 1.0;

/// Build the shipped [`ForceField`] under `name` (`"MMFF94"` or `"MMFF94s"`).
///
/// The style *order* is the table's ([`MMFF_STYLES`]) because `ForceField`
/// lookups take the first matching style; the numbers are the table's too.
pub(super) fn force_field(name: &str) -> ForceField {
    let mut ff = ForceField::new(name);

    for style in MMFF_STYLES {
        match style.category {
            "bond" => {
                ff.def_bondstyle(style.name);
            }
            "angle" => {
                ff.def_anglestyle(style.name);
            }
            "dihedral" => {
                ff.def_dihedralstyle(style.name);
            }
            "improper" => {
                ff.def_improperstyle(style.name);
            }
            // The two pair styles are the only ones carrying style-level params,
            // and `mmff_vdw` is the only one with per-type rows (the other five
            // are `ParamSource::PerInstance`).
            "pair" if style.name == "mmff_vdw" => {
                let vdw = ff.def_pairstyle(
                    style.name,
                    &[
                        ("B", MMFF_VDW_STYLE.b),
                        ("Beta", MMFF_VDW_STYLE.beta),
                        ("DARAD", MMFF_VDW_STYLE.darad),
                        ("DAEPS", MMFF_VDW_STYLE.daeps),
                    ],
                );
                for row in MMFF_VDW {
                    let atom_type = row.atom_type.to_string();
                    vdw.def_pairtype(
                        &atom_type,
                        None,
                        &[
                            ("alpha", row.alpha_i),
                            ("n_eff", row.n_i),
                            ("a_i", row.a_i),
                            ("g_i", row.g_i),
                            ("da", encode_da_byte(row.da)),
                            ("type", f64::from(row.atom_type)),
                        ],
                    );
                }
            }
            "pair" => {
                ff.def_pairstyle(
                    style.name,
                    &[
                        ("dielectric", MMFF_ELE_STYLE.dielectric),
                        ("delta", MMFF_ELE_STYLE.delta),
                    ],
                );
            }
            other => unreachable!("MMFF declares no `{other}` style"),
        }
    }

    ff.set_special_bonds(SpecialBonds {
        lj: [0.0, 0.0, VDW_SCALE_14],
        coul: [0.0, 0.0, MMFF_ELE_STYLE.scale14],
    });
    ff
}

/// The typing metadata: MMFF's 95 nine-column atom-property rows.
///
/// `sbmb` picks the stretch-bend row, `arom` / `pilp` / `mltb` drive bond
/// classification, `crd` / `val` gate the type assignment. None of them appears
/// in an energy.
pub(super) fn typing_params() -> MMFFParams {
    let props: HashMap<u32, MMFFAtomProp> = MMFF_PROP
        .iter()
        .map(|p| {
            let prop = MMFFAtomProp {
                type_id: u32::from(p.atom_type),
                atno: u32::from(p.atno),
                crd: u32::from(p.crd),
                val: u32::from(p.val),
                pilp: u32::from(p.pilp),
                mltb: u32::from(p.mltb),
                arom: u32::from(p.arom),
                linh: u32::from(p.linh),
                sbmb: u32::from(p.sbmb),
            };
            (prop.type_id, prop)
        })
        .collect();
    MMFFParams::new(props)
}
