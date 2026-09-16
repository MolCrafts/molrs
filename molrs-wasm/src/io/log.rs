//! Solver log parsing for the WASM API.
//!
//! A thermo table is the one thing a running simulation emits that a chart
//! wants, and it arrives as text — so the browser can read it directly rather
//! than asking a server to convert it first. `parse_lammps_log_text` takes a
//! `&str` and touches no filesystem, which is what makes this WASM-clean.
//!
//! | JS function | Format |
//! |-------------|--------|
//! | `readLammpsLogThermo(text)` | LAMMPS log — one entry per `run` block |
//!
//! The tables come back whole. Downsampling belongs to whoever is drawing:
//! it knows how many points the chart can show, and slicing a `Float64Array`
//! in JS costs nothing next to re-parsing.

use molrs::io::log::lammps::parse_lammps_log_text;
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// One `run` block's thermo table.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ThermoTable {
    /// Position of the block in the log; a restart appends rather than resets,
    /// so this is what separates one continuation from the next.
    pub run_index: usize,
    /// Column names exactly as LAMMPS printed them (`Step`, `Temp`, …).
    pub columns: Vec<String>,
    /// Row-major values, one inner vector per thermo row.
    pub rows: Vec<Vec<f64>>,
}

/// Parse a LAMMPS log's thermo tables.
///
/// Returns one [`ThermoTable`] per `run` block that produced thermo output;
/// blocks that never reached a thermo section are omitted rather than
/// reported as empty, so a caller can treat an empty result as "nothing to
/// plot" without inspecting each entry.
///
/// # Example (JavaScript)
///
/// ```js
/// const tables = readLammpsLogThermo(await file.text());
/// const temp = tables[0].columns.indexOf("Temp");
/// const series = tables[0].rows.map((row) => row[temp]);
/// ```
#[wasm_bindgen(js_name = readLammpsLogThermo)]
pub fn read_lammps_log_thermo(text: &str) -> Result<JsValue, JsValue> {
    let parsed = parse_lammps_log_text(text, "", "default");
    let tables: Vec<ThermoTable> = parsed
        .runs
        .into_iter()
        .enumerate()
        .filter_map(|(run_index, run)| {
            let thermo = run.thermo?;
            if thermo.columns.is_empty() || thermo.rows.is_empty() {
                return None;
            }
            Some(ThermoTable {
                run_index,
                columns: thermo.columns,
                rows: thermo.rows,
            })
        })
        .collect();
    serde_wasm_bindgen::to_value(&tables)
        .map_err(|e| JsValue::from_str(&format!("LAMMPS log serialization error: {e}")))
}

/// True when *text* looks like a LAMMPS log carrying a thermo table.
///
/// The banner alone is not enough: a log whose run died during setup has
/// nothing to plot, so the thermo marker is what decides. Cheap enough to run
/// on a file's first bytes before deciding to read the whole thing.
#[wasm_bindgen(js_name = isLammpsLog)]
pub fn is_lammps_log(text: &str) -> bool {
    text.contains("Per MPI rank memory allocation")
}

#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    use super::*;
    use js_sys::{Array, Reflect};
    use wasm_bindgen_test::wasm_bindgen_test;

    #[wasm_bindgen_test]
    fn thermo_tables_preserve_run_indices_and_numeric_rows() {
        let text = "\
LAMMPS (30 Mar 2026)
Per MPI rank memory allocation (min/avg/max) = 1 | 1 | 1 Mbytes
Total wall time: 0:00:01
Per MPI rank memory allocation (min/avg/max) = 1 | 1 | 1 Mbytes
Step Temp
1000 298.5
2000 301.25
Loop time of 1.0 on 1 procs for 1000 steps with 10 atoms
Total wall time: 0:00:02
";
        assert!(is_lammps_log(text));
        let tables = Array::from(&read_lammps_log_thermo(text).unwrap());
        assert_eq!(tables.length(), 1);
        let table = tables.get(0);
        assert_eq!(
            Reflect::get(&table, &"runIndex".into()).unwrap().as_f64(),
            Some(1.0)
        );
        let columns = Array::from(&Reflect::get(&table, &"columns".into()).unwrap());
        assert_eq!(columns.get(1).as_string().as_deref(), Some("Temp"));
        let rows = Array::from(&Reflect::get(&table, &"rows".into()).unwrap());
        assert_eq!(rows.length(), 2);
        assert_eq!(Array::from(&rows.get(1)).get(1).as_f64(), Some(301.25));
        assert_eq!(
            Array::from(&read_lammps_log_thermo("LAMMPS").unwrap()).length(),
            0
        );
    }
}
