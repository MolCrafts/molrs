//! WASM bindings for inspecting the Frame schema.
//!
//! Projected from the compiled-in Rust tables, so the JS view of the contract
//! cannot describe something the Rust enforcement does not.
//!
//! `schemaDocument()` hands back a real JS object via `serde-wasm-bindgen`
//! rather than a string the caller has to `JSON.parse` — a browser consumer
//! wants the object. `schemaJson()` is kept for callers that want the exact
//! bytes to persist or diff.

use wasm_bindgen::prelude::*;

use molrs::store::schema;

/// The whole Frame vocabulary as a JS object.
///
/// ```js
/// const doc = molrs.schemaDocument();
/// doc.columns.find(c => c.key === "atomi").dtype;  // "uint"
/// ```
#[wasm_bindgen(js_name = schemaDocument)]
pub fn schema_document() -> Result<JsValue, JsValue> {
    serde_wasm_bindgen::to_value(&schema::document())
        .map_err(|e| JsValue::from_str(&format!("schema: {e}")))
}

/// The vocabulary as canonical JSON — stable across runs, so two releases can
/// be diffed byte for byte.
#[wasm_bindgen(js_name = schemaJson)]
pub fn schema_json() -> String {
    schema::document().to_json()
}

/// Vocabulary version — what the names and dtypes *mean*.
///
/// Distinct from the serialization envelope version; a consumer persisting
/// frames should record this alongside the data.
#[wasm_bindgen(js_name = schemaVocabVersion)]
pub fn schema_vocab_version() -> u32 {
    schema::FRAME_VOCAB_VERSION
}

/// Declared dtype of a canonical column.
///
/// Returns `undefined` when the key carries no declared dtype. That means the
/// key is **unconstrained**, not invalid: the column vocabulary is closed but
/// unspecified keys are the documented extension point.
#[wasm_bindgen(js_name = schemaColumnDtype)]
pub fn schema_column_dtype(key: &str) -> Option<String> {
    schema::column(key).map(|spec| spec.dtype.name().to_string())
}

/// Whether a block name is part of the canonical vocabulary.
///
/// `false` does not mean the block is illegal — the block set is open, and a
/// frame may carry blocks the vocabulary does not name.
#[wasm_bindgen(js_name = schemaHasBlock)]
pub fn schema_has_block(name: &str) -> bool {
    schema::block(name).is_some()
}

/// Every canonical column key, in vocabulary order.
#[wasm_bindgen(js_name = schemaColumnKeys)]
pub fn schema_column_keys() -> Vec<String> {
    schema::SCHEMA_COLUMNS
        .iter()
        .map(|c| c.key.to_string())
        .collect()
}

/// Every canonical block name, in vocabulary order.
#[wasm_bindgen(js_name = schemaBlockNames)]
pub fn schema_block_names() -> Vec<String> {
    schema::SCHEMA_BLOCKS
        .iter()
        .map(|b| b.name.to_string())
        .collect()
}
