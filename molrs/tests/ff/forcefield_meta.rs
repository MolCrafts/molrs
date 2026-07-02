//! End-to-end tests for `forcefield_meta`: force-field method metadata JSON.

use molrs::ff::ForceField;
use molrs::ff::forcefield_method_json;

fn demo_ff() -> ForceField {
    let mut ff = ForceField::new("OPLS-demo");
    ff.def_bondstyle("harmonic")
        .def_type("CT-CT", &[("k", 268.0), ("r0", 1.529)]);
    ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
        .def_type("CT", &[("epsilon", 0.066), ("sigma", 3.5)]);
    ff
}

#[test]
fn forcefield_method_json_records_method_and_styles() {
    let ff = demo_ff();

    let method = forcefield_method_json(&ff);

    assert_eq!(method["type"], "classical");
    let fffield = &method["classical"]["force_field"];
    assert_eq!(fffield["name"], "OPLS-demo");
    let styles = fffield["styles"].as_array().expect("styles array");
    assert_eq!(styles.len(), 2);
    let categories: Vec<&str> = styles
        .iter()
        .map(|s| s["category"].as_str().unwrap())
        .collect();
    assert!(categories.contains(&"bond"));
    assert!(categories.contains(&"pair"));
}

#[test]
fn empty_forcefield_yields_empty_style_list() {
    let ff = ForceField::new("bare");
    let method = forcefield_method_json(&ff);
    let styles = method["classical"]["force_field"]["styles"]
        .as_array()
        .expect("styles array");
    assert!(styles.is_empty());
}
