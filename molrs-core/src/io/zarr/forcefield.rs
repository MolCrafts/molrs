//! ForceField ↔ Zarr serialization.
//!
//! Each [`Style`] maps to a Zarr group under `{prefix}/forcefield/styles/`.
//! The group name is `{category}__{sanitized_style_name}`.

use zarrs::array::{Array, ArraySubset};
#[cfg(feature = "filesystem")]
use zarrs::array::{ArrayBuilder, data_type};
#[cfg(feature = "filesystem")]
use zarrs::group::GroupBuilder;
use zarrs::node::{Node, NodeMetadata};
use zarrs::storage::ReadableWritableListableStorage;

use crate::error::MolRsError;
use crate::forcefield::ForceField;
#[cfg(feature = "filesystem")]
use crate::forcefield::{Style, StyleDefs};

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------

#[cfg(feature = "filesystem")]
pub(crate) fn write_forcefield(
    store: &ReadableWritableListableStorage,
    prefix: &str,
    ff: &ForceField,
) -> Result<(), MolRsError> {
    let ff_root = super::frame_io::join_path(prefix, "forcefield");

    let mut root_attrs = serde_json::Map::new();
    root_attrs.insert("name".into(), serde_json::Value::String(ff.name.clone()));
    GroupBuilder::new()
        .attributes(root_attrs)
        .build(store.clone(), &ff_root)?
        .store_metadata()?;

    let styles_root = format!("{}/styles", ff_root);
    GroupBuilder::new()
        .build(store.clone(), &styles_root)?
        .store_metadata()?;

    for style in ff.styles() {
        write_style(store, &styles_root, style)?;
    }

    Ok(())
}

#[cfg(feature = "filesystem")]
fn write_style(
    store: &ReadableWritableListableStorage,
    styles_root: &str,
    style: &Style,
) -> Result<(), MolRsError> {
    let group_name = format!(
        "{}/{}__{}",
        styles_root,
        style.category(),
        sanitize_name(&style.name)
    );

    // Style-level attrs
    let mut attrs = serde_json::Map::new();
    attrs.insert(
        "category".into(),
        serde_json::Value::String(style.category().to_owned()),
    );
    attrs.insert(
        "style_name".into(),
        serde_json::Value::String(style.name.clone()),
    );

    // style_params as JSON object
    let mut sp = serde_json::Map::new();
    for (k, v) in style.params.iter() {
        sp.insert(k.to_owned(), serde_json::Value::from(v));
    }
    attrs.insert("style_params".into(), serde_json::Value::Object(sp));

    GroupBuilder::new()
        .attributes(attrs)
        .build(store.clone(), &group_name)?
        .store_metadata()?;

    // Type definitions → columnar arrays
    let type_params = style.defs.collect_type_params();
    if type_params.is_empty() {
        return Ok(());
    }

    let n = type_params.len() as u64;

    // name array
    let names: Vec<String> = type_params.iter().map(|(n, _)| n.clone()).collect();
    write_string_array(store, &format!("{}/name", group_name), &names)?;

    // atom-ref columns (itom, jtom, ktom, ltom) — depends on category
    write_atom_ref_columns(store, &group_name, &style.defs)?;

    // param columns — union of all param keys across types
    let all_keys = collect_param_keys(&type_params);
    for key in &all_keys {
        let values: Vec<f64> = type_params
            .iter()
            .map(|(_, p)| p.get(key).unwrap_or(0.0))
            .collect();
        let path = format!("{}/{}", group_name, key);
        let arr = ArrayBuilder::new(vec![n], vec![n], data_type::float64(), 0.0f64)
            .build(store.clone(), &path)?;
        arr.store_metadata()?;
        arr.store_array_subset(&ArraySubset::new_with_shape(vec![n]), &values)?;
    }

    Ok(())
}

#[cfg(feature = "filesystem")]
fn write_atom_ref_columns(
    store: &ReadableWritableListableStorage,
    group: &str,
    defs: &StyleDefs,
) -> Result<(), MolRsError> {
    match defs {
        StyleDefs::Atom(_) => {} // no atom-ref columns
        StyleDefs::Bond(types) => {
            let itoms: Vec<String> = types.iter().map(|t| t.itom.clone()).collect();
            let jtoms: Vec<String> = types.iter().map(|t| t.jtom.clone()).collect();
            write_string_array(store, &format!("{}/itom", group), &itoms)?;
            write_string_array(store, &format!("{}/jtom", group), &jtoms)?;
        }
        StyleDefs::Angle(types) => {
            let itoms: Vec<String> = types.iter().map(|t| t.itom.clone()).collect();
            let jtoms: Vec<String> = types.iter().map(|t| t.jtom.clone()).collect();
            let ktoms: Vec<String> = types.iter().map(|t| t.ktom.clone()).collect();
            write_string_array(store, &format!("{}/itom", group), &itoms)?;
            write_string_array(store, &format!("{}/jtom", group), &jtoms)?;
            write_string_array(store, &format!("{}/ktom", group), &ktoms)?;
        }
        StyleDefs::Dihedral(types) => {
            let itoms: Vec<String> = types.iter().map(|t| t.itom.clone()).collect();
            let jtoms: Vec<String> = types.iter().map(|t| t.jtom.clone()).collect();
            let ktoms: Vec<String> = types.iter().map(|t| t.ktom.clone()).collect();
            let ltoms: Vec<String> = types.iter().map(|t| t.ltom.clone()).collect();
            write_string_array(store, &format!("{}/itom", group), &itoms)?;
            write_string_array(store, &format!("{}/jtom", group), &jtoms)?;
            write_string_array(store, &format!("{}/ktom", group), &ktoms)?;
            write_string_array(store, &format!("{}/ltom", group), &ltoms)?;
        }
        StyleDefs::Improper(types) => {
            let itoms: Vec<String> = types.iter().map(|t| t.itom.clone()).collect();
            let jtoms: Vec<String> = types.iter().map(|t| t.jtom.clone()).collect();
            let ktoms: Vec<String> = types.iter().map(|t| t.ktom.clone()).collect();
            let ltoms: Vec<String> = types.iter().map(|t| t.ltom.clone()).collect();
            write_string_array(store, &format!("{}/itom", group), &itoms)?;
            write_string_array(store, &format!("{}/jtom", group), &jtoms)?;
            write_string_array(store, &format!("{}/ktom", group), &ktoms)?;
            write_string_array(store, &format!("{}/ltom", group), &ltoms)?;
        }
        StyleDefs::Pair(types) => {
            let itoms: Vec<String> = types.iter().map(|t| t.itom.clone()).collect();
            let jtoms: Vec<String> = types.iter().map(|t| t.jtom.clone()).collect();
            write_string_array(store, &format!("{}/itom", group), &itoms)?;
            write_string_array(store, &format!("{}/jtom", group), &jtoms)?;
        }
        StyleDefs::KSpace => {} // no type arrays
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Read
// ---------------------------------------------------------------------------

pub(crate) fn read_forcefield(
    store: &ReadableWritableListableStorage,
    prefix: &str,
) -> Result<Option<ForceField>, MolRsError> {
    let ff_root = super::frame_io::join_path(prefix, "forcefield");

    let ff_group = match zarrs::group::Group::open(store.clone(), &ff_root) {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };

    let name = ff_group
        .attributes()
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("unnamed")
        .to_owned();

    let mut ff = ForceField::new(&name);

    let styles_root = format!("{}/styles", ff_root);
    let styles_node = match Node::open(store, &styles_root) {
        Ok(n) => n,
        Err(_) => return Ok(Some(ff)),
    };

    for child in styles_node.children() {
        if !matches!(child.metadata(), NodeMetadata::Group(_)) {
            continue;
        }
        read_style(store, child.path().as_str(), &mut ff)?;
    }

    Ok(Some(ff))
}

fn read_style(
    store: &ReadableWritableListableStorage,
    group_path: &str,
    ff: &mut ForceField,
) -> Result<(), MolRsError> {
    let group = zarrs::group::Group::open(store.clone(), group_path)?;
    let attrs = group.attributes();

    let category = attrs
        .get("category")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_owned();
    let style_name = attrs
        .get("style_name")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_owned();

    // Parse style_params
    let style_params: Vec<(String, f64)> = attrs
        .get("style_params")
        .and_then(|v| v.as_object())
        .map(|obj| {
            obj.iter()
                .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
                .collect()
        })
        .unwrap_or_default();
    let sp_refs: Vec<(&str, f64)> = style_params.iter().map(|(k, v)| (k.as_str(), *v)).collect();

    // Read type names (if any)
    let names_path = format!("{}/name", group_path);
    let names = read_string_array(store, &names_path).unwrap_or_default();
    if names.is_empty() {
        // KSpace or empty style — just register style
        match category.as_str() {
            "kspace" => {
                ff.def_kspacestyle(&style_name, &sp_refs);
            }
            "atom" => {
                ff.def_atomstyle(&style_name);
            }
            "bond" => {
                ff.def_bondstyle(&style_name);
            }
            "angle" => {
                ff.def_anglestyle(&style_name);
            }
            "dihedral" => {
                ff.def_dihedralstyle(&style_name);
            }
            "improper" => {
                ff.def_improperstyle(&style_name);
            }
            "pair" => {
                ff.def_pairstyle(&style_name, &sp_refs);
            }
            _ => {}
        }
        return Ok(());
    }

    // Discover param keys from array children (skip name, itom, jtom, ktom, ltom)
    let skip = ["name", "itom", "jtom", "ktom", "ltom"];
    let node = Node::open(store, group_path)?;
    let mut param_keys = Vec::new();
    for child in node.children() {
        if !matches!(child.metadata(), NodeMetadata::Array(_)) {
            continue;
        }
        let col_name = child.path().as_str().rsplit('/').next().unwrap_or("");
        if !skip.contains(&col_name) {
            param_keys.push(col_name.to_owned());
        }
    }

    // Read param columns
    let mut param_cols: Vec<(String, Vec<f64>)> = Vec::new();
    for key in &param_keys {
        let path = format!("{}/{}", group_path, key);
        let arr = Array::open(store.clone(), &path)?;
        let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
        let data: Vec<f64> = arr.retrieve_array_subset(&subset)?;
        param_cols.push((key.clone(), data));
    }

    // Build type params for each type
    let n = names.len();
    let itoms = read_string_array(store, &format!("{}/itom", group_path)).unwrap_or_default();
    let jtoms = read_string_array(store, &format!("{}/jtom", group_path)).unwrap_or_default();
    let ktoms = read_string_array(store, &format!("{}/ktom", group_path)).unwrap_or_default();
    let ltoms = read_string_array(store, &format!("{}/ltom", group_path)).unwrap_or_default();

    #[allow(clippy::needless_range_loop)]
    match category.as_str() {
        "atom" => {
            let style = ff.def_atomstyle(&style_name);
            for i in 0..n {
                let params = build_params(i, &param_cols);
                style.def_atomtype(&names[i], &params);
            }
        }
        "bond" => {
            let style = ff.def_bondstyle(&style_name);
            for i in 0..n {
                let params = build_params(i, &param_cols);
                style.def_bondtype(&itoms[i], &jtoms[i], &params);
            }
        }
        "angle" => {
            let style = ff.def_anglestyle(&style_name);
            for i in 0..n {
                let params = build_params(i, &param_cols);
                style.def_angletype(&itoms[i], &jtoms[i], &ktoms[i], &params);
            }
        }
        "dihedral" => {
            let style = ff.def_dihedralstyle(&style_name);
            for i in 0..n {
                let params = build_params(i, &param_cols);
                style.def_dihedraltype(&itoms[i], &jtoms[i], &ktoms[i], &ltoms[i], &params);
            }
        }
        "improper" => {
            let style = ff.def_improperstyle(&style_name);
            for i in 0..n {
                let params = build_params(i, &param_cols);
                style.def_impropertype(&itoms[i], &jtoms[i], &ktoms[i], &ltoms[i], &params);
            }
        }
        "pair" => {
            let style = ff.def_pairstyle(&style_name, &sp_refs);
            for i in 0..n {
                let params = build_params(i, &param_cols);
                let jt = if itoms[i] == jtoms[i] {
                    None
                } else {
                    Some(jtoms[i].as_str())
                };
                style.def_pairtype(&itoms[i], jt, &params);
            }
        }
        _ => {}
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "filesystem")]
fn sanitize_name(name: &str) -> String {
    name.replace('/', "_")
}

#[cfg(feature = "filesystem")]
fn collect_param_keys(type_params: &[(String, crate::forcefield::Params)]) -> Vec<String> {
    let mut keys = std::collections::BTreeSet::new();
    for (_, p) in type_params {
        for (k, _) in p.iter() {
            keys.insert(k.to_owned());
        }
    }
    keys.into_iter().collect()
}

fn build_params(idx: usize, cols: &[(String, Vec<f64>)]) -> Vec<(&str, f64)> {
    cols.iter()
        .map(|(k, vals)| (k.as_str(), vals[idx]))
        .collect()
}

#[cfg(feature = "filesystem")]
fn write_string_array(
    store: &ReadableWritableListableStorage,
    path: &str,
    data: &[String],
) -> Result<(), MolRsError> {
    let n = data.len() as u64;
    let arr =
        ArrayBuilder::new(vec![n], vec![n], data_type::string(), "").build(store.clone(), path)?;
    arr.store_metadata()?;
    arr.store_array_subset(&ArraySubset::new_with_shape(vec![n]), data)?;
    Ok(())
}

fn read_string_array(
    store: &ReadableWritableListableStorage,
    path: &str,
) -> Result<Vec<String>, MolRsError> {
    let arr = Array::open(store.clone(), path)?;
    let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
    let data: Vec<String> = arr.retrieve_array_subset(&subset)?;
    Ok(data)
}
