//! LAMMPS run logs for the WASM API — the face of `molrs::io::lammps::log`
//! (`io::read_lammps_log_str`, `io::lammps::is_lammps_log`).
//!
//! A thermo table is the one thing a running simulation emits that a chart
//! wants, and it arrives as text — so the browser reads it directly rather
//! than asking a server to convert it first.
//!
//! | JS function | molrs |
//! |-------------|-------|
//! | `readLammpsLogStr(text, style?)` | `io::read_lammps_log_str` → `LammpsLog` |
//! | `isLammpsLog(text)` | `io::lammps::is_lammps_log` |
//!
//! The tables come back whole. Downsampling belongs to whoever is drawing:
//! it knows how many points the chart can show, and slicing an array in JS
//! costs nothing next to re-parsing.

use molrs::io::lammps::is_lammps_log as is_lammps_log_rs;
use molrs::io::read_lammps_log_str;
use wasm_bindgen::prelude::*;

/// Parse a LAMMPS log's text into molrs's `LammpsLog` record.
///
/// The object has the Rust and Python record's fields, under the same
/// (snake_case) names: `version`, `header`, `runs`, `total_wall_time`,
/// `warnings`, `raw_text`, `style`, `path` (always `""`). Each run has
/// `index`, `memory`, `thermo` (`{ columns, rows, raw_lines }`, or `undefined` for
/// a run that printed none), `loop_time`, `performance`, `CPU_use`,
/// `MPI_task_timing`, `thread_timing`, `load_balance`,
/// `neighbor_statistics`, `warnings`, `setup_log`, `unparsed_log` and
/// `raw_text`. `style` is the thermo style the run printed; only `"default"`
/// tables are parsed.
///
/// # Example (JavaScript)
///
/// ```js
/// const log = readLammpsLogStr(await file.text());
/// const thermo = log.runs[0].thermo;
/// const temp = thermo.columns.indexOf("Temp");
/// const series = thermo.rows.map((row) => row[temp]);
/// ```
#[wasm_bindgen(js_name = readLammpsLogStr)]
pub fn read_lammps_log_str_export(text: &str, style: Option<String>) -> Result<JsValue, JsValue> {
    let log = read_lammps_log_str(text, "", style.as_deref().unwrap_or("default"));
    serde_wasm_bindgen::to_value(&log)
        .map_err(|e| JsValue::from_str(&format!("LAMMPS log serialization error: {e}")))
}

/// True when `text` holds a LAMMPS run — molrs `io::lammps::is_lammps_log`.
///
/// Keys on the `Per MPI rank memory allocation` line each run opens with, as
/// `readLammpsLogStr` does: a banner alone (a run that died in setup) has
/// nothing to read. Cheap enough to run on a file's first bytes.
#[wasm_bindgen(js_name = isLammpsLog)]
pub fn is_lammps_log(text: &str) -> bool {
    is_lammps_log_rs(text)
}

#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    use super::*;
    use js_sys::{Array, Reflect};
    use wasm_bindgen_test::wasm_bindgen_test;

    fn get(value: &JsValue, key: &str) -> JsValue {
        Reflect::get(value, &key.into()).unwrap()
    }

    #[wasm_bindgen_test]
    fn the_log_record_keeps_run_indices_and_numeric_rows() {
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
        let log = read_lammps_log_str_export(text, None).unwrap();
        let runs = Array::from(&get(&log, "runs"));
        assert_eq!(runs.length(), 2);
        assert!(get(&runs.get(0), "thermo").is_undefined());
        let run = runs.get(1);
        assert_eq!(get(&run, "index").as_f64(), Some(1.0));
        let thermo = get(&run, "thermo");
        let columns = Array::from(&get(&thermo, "columns"));
        assert_eq!(columns.get(1).as_string().as_deref(), Some("Temp"));
        let rows = Array::from(&get(&thermo, "rows"));
        assert_eq!(rows.length(), 2);
        assert_eq!(Array::from(&rows.get(1)).get(1).as_f64(), Some(301.25));
        assert!(!is_lammps_log("LAMMPS"));
        let empty = read_lammps_log_str_export("LAMMPS", None).unwrap();
        assert_eq!(Array::from(&get(&empty, "runs")).length(), 0);
    }
}
