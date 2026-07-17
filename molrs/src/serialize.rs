//! `serde` support for the core model, enabled by the `serde` feature.
//!
//! The real [`Frame`], [`Block`], [`Column`], and [`SimBox`] implement
//! `Serialize`/`Deserialize` **directly** — there is no parallel "wire" type.
//! Transport encodings (MessagePack / JSON) live in [`crate::stream`], which
//! turns this feature on.
//!
//! On-wire shape (the general MolRec model, no privileged fields):
//!
//! - `Frame`  -> `{ version: 2, blocks: { <name>: Block }, meta: { k: {dtype,value} }, box?: SimBox }`
//! - `Block`  -> `{ shape: [usize], columns: { <name>: Column } }`
//! - `Column` -> `{ dtype, shape: [usize], data }` — `data` is raw
//!   little-endian bytes for numeric dtypes, or a string list for `string`.
//! - `SimBox` -> `{ vectors: [[f64;3];3], origin, boundary, cell_defined }`
//!
//! The small private `*Repr` structs and visitors below are serde
//! deserialization scaffolding (derive needs owned fields); they are not part
//! of the public model.

use std::collections::BTreeMap;

use ndarray::{ArrayD, IxDyn};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::core::spatial::region::simbox::SimBox;
use crate::core::store::block::{Block, Column, DType};
use crate::core::store::frame::{FRAME_SCHEMA_VERSION, Frame};
use crate::core::store::meta::{MetaMap, MetaValue};

// ===== MetaValue ===========================================================

impl Serialize for MetaValue {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_json_value().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MetaValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        MetaValue::from_json_value(&value).map_err(de::Error::custom)
    }
}

// ===== Column ===============================================================

/// Serialize a byte slice via serde's `bytes` (a compact `bin` in MessagePack;
/// an integer array in JSON).
struct RawBytes<'a>(&'a [u8]);

impl Serialize for RawBytes<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_bytes(self.0)
    }
}

impl Serialize for Column {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut st = s.serialize_struct("Column", 3)?;
        st.serialize_field("dtype", self.dtype().name())?;
        st.serialize_field("shape", self.shape())?;
        match self {
            Column::String(h) => {
                let v: Vec<&String> = h.array().iter().collect();
                st.serialize_field("data", &v)?;
            }
            _ => {
                let bytes = self
                    .raw_bytes()
                    .expect("a non-string column always yields raw_bytes");
                st.serialize_field("data", &RawBytes(&bytes))?;
            }
        }
        st.end()
    }
}

fn dtype_from_tag(tag: &str) -> Option<DType> {
    Some(match tag {
        "float" => DType::Float,
        "int" => DType::Int,
        "bool" => DType::Bool,
        "uint" => DType::UInt,
        "u8" => DType::U8,
        "string" => DType::String,
        _ => return None,
    })
}

/// Decoded column payload, chosen by dtype.
enum ColData {
    Bytes(Vec<u8>),
    Strings(Vec<String>),
}

/// Untyped payload as it appears on the wire. The dtype field may arrive before
/// or after data, so we defer interpretation until the whole Column map is read.
enum RawData {
    Bytes(Vec<u8>),
    Strings(Vec<String>),
}

impl RawData {
    fn into_typed(self, dtype: DType) -> Result<ColData, String> {
        match (dtype, self) {
            (DType::String, RawData::Strings(s)) => Ok(ColData::Strings(s)),
            (DType::String, RawData::Bytes(b)) if b.is_empty() => Ok(ColData::Strings(Vec::new())),
            (DType::String, RawData::Bytes(_)) => {
                Err("string column payload must be a string array".to_string())
            }
            (_, RawData::Bytes(b)) => Ok(ColData::Bytes(b)),
            (_, RawData::Strings(_)) => {
                Err("numeric column payload must be a byte buffer".to_string())
            }
        }
    }
}

enum DataElem {
    Byte(u8),
    String(String),
}

impl<'de> Deserialize<'de> for DataElem {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct ElemVisitor;
        impl Visitor<'_> for ElemVisitor {
            type Value = DataElem;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a byte value or a string")
            }
            fn visit_u64<E: de::Error>(self, v: u64) -> Result<DataElem, E> {
                u8::try_from(v)
                    .map(DataElem::Byte)
                    .map_err(|_| de::Error::custom(format!("byte value out of range: {v}")))
            }
            fn visit_i64<E: de::Error>(self, v: i64) -> Result<DataElem, E> {
                u8::try_from(v)
                    .map(DataElem::Byte)
                    .map_err(|_| de::Error::custom(format!("byte value out of range: {v}")))
            }
            fn visit_str<E: de::Error>(self, v: &str) -> Result<DataElem, E> {
                Ok(DataElem::String(v.to_string()))
            }
            fn visit_string<E: de::Error>(self, v: String) -> Result<DataElem, E> {
                Ok(DataElem::String(v))
            }
        }
        d.deserialize_any(ElemVisitor)
    }
}

/// Accept numeric payloads as MessagePack `bin` (`visit_bytes`/`visit_byte_buf`)
/// or JSON integer arrays, and string payloads as string arrays.
struct RawDataVisitor;

impl<'de> Visitor<'de> for RawDataVisitor {
    type Value = RawData;
    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("a byte buffer or a string array")
    }
    fn visit_bytes<E: de::Error>(self, v: &[u8]) -> Result<RawData, E> {
        Ok(RawData::Bytes(v.to_vec()))
    }
    fn visit_byte_buf<E: de::Error>(self, v: Vec<u8>) -> Result<RawData, E> {
        Ok(RawData::Bytes(v))
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<RawData, A::Error> {
        let Some(first) = seq.next_element::<DataElem>()? else {
            return Ok(RawData::Bytes(Vec::new()));
        };
        match first {
            DataElem::Byte(b) => {
                let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0) + 1);
                out.push(b);
                while let Some(next) = seq.next_element::<DataElem>()? {
                    match next {
                        DataElem::Byte(b) => out.push(b),
                        DataElem::String(_) => {
                            return Err(de::Error::custom(
                                "mixed string and byte values in column payload",
                            ));
                        }
                    }
                }
                Ok(RawData::Bytes(out))
            }
            DataElem::String(s) => {
                let mut out = Vec::with_capacity(seq.size_hint().unwrap_or(0) + 1);
                out.push(s);
                while let Some(next) = seq.next_element::<DataElem>()? {
                    match next {
                        DataElem::String(s) => out.push(s),
                        DataElem::Byte(_) => {
                            return Err(de::Error::custom(
                                "mixed string and byte values in column payload",
                            ));
                        }
                    }
                }
                Ok(RawData::Strings(out))
            }
        }
    }
}

impl<'de> Deserialize<'de> for RawData {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(RawDataVisitor)
    }
}

impl<'de> Deserialize<'de> for Column {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Column, D::Error> {
        struct ColumnVisitor;
        impl<'de> Visitor<'de> for ColumnVisitor {
            type Value = Column;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("a column map {dtype, shape, data}")
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Column, A::Error> {
                let mut dtype: Option<DType> = None;
                let mut shape: Option<Vec<usize>> = None;
                let mut data: Option<RawData> = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "dtype" => {
                            let tag: String = map.next_value()?;
                            dtype = Some(dtype_from_tag(&tag).ok_or_else(|| {
                                de::Error::custom(format!("unknown dtype {tag:?}"))
                            })?);
                        }
                        "shape" => shape = Some(map.next_value()?),
                        "data" => data = Some(map.next_value()?),
                        _ => {
                            let _: de::IgnoredAny = map.next_value()?;
                        }
                    }
                }
                let dtype = dtype.ok_or_else(|| de::Error::missing_field("dtype"))?;
                let shape = shape.ok_or_else(|| de::Error::missing_field("shape"))?;
                let data = data
                    .ok_or_else(|| de::Error::missing_field("data"))?
                    .into_typed(dtype)
                    .map_err(de::Error::custom)?;
                build_column(dtype, &shape, data).map_err(de::Error::custom)
            }
        }
        d.deserialize_struct("Column", &["dtype", "shape", "data"], ColumnVisitor)
    }
}

fn build_column(dtype: DType, shape: &[usize], data: ColData) -> Result<Column, String> {
    let n: usize = shape.iter().product();
    let ix = IxDyn(shape);
    let shape_err = |ty: &str| format!("{ty} column: element count does not match shape {shape:?}");
    match (dtype, data) {
        (DType::Float, ColData::Bytes(b)) => {
            let v = le::<8, _>(&b, n, f64::from_le_bytes)?;
            Ok(Column::from_float(
                ArrayD::from_shape_vec(ix, v).map_err(|e| e.to_string())?,
            ))
        }
        (DType::Int, ColData::Bytes(b)) => {
            let v = le::<4, _>(&b, n, i32::from_le_bytes)?;
            Ok(Column::from_int(
                ArrayD::from_shape_vec(ix, v).map_err(|e| e.to_string())?,
            ))
        }
        (DType::UInt, ColData::Bytes(b)) => {
            let v = le::<4, _>(&b, n, u32::from_le_bytes)?;
            Ok(Column::from_uint(
                ArrayD::from_shape_vec(ix, v).map_err(|e| e.to_string())?,
            ))
        }
        (DType::U8, ColData::Bytes(b)) => {
            if b.len() != n {
                return Err(shape_err("u8"));
            }
            Ok(Column::from_u8(
                ArrayD::from_shape_vec(ix, b).map_err(|e| e.to_string())?,
            ))
        }
        (DType::Bool, ColData::Bytes(b)) => {
            if b.len() != n {
                return Err(shape_err("bool"));
            }
            let v: Vec<bool> = b.iter().map(|&x| x != 0).collect();
            Ok(Column::from_bool(
                ArrayD::from_shape_vec(ix, v).map_err(|e| e.to_string())?,
            ))
        }
        (DType::String, ColData::Strings(s)) => {
            if s.len() != n {
                return Err(shape_err("string"));
            }
            Ok(Column::from_string(
                ArrayD::from_shape_vec(ix, s).map_err(|e| e.to_string())?,
            ))
        }
        _ => Err("dtype does not match its data payload".to_string()),
    }
}

/// Parse `n` little-endian values of `N` bytes each from `bytes`.
fn le<const N: usize, T>(
    bytes: &[u8],
    n: usize,
    read: impl Fn([u8; N]) -> T,
) -> Result<Vec<T>, String> {
    if bytes.len() != n * N {
        return Err(format!(
            "numeric payload has {} byte(s), expected {}",
            bytes.len(),
            n * N
        ));
    }
    Ok(bytes
        .chunks_exact(N)
        .map(|c| read(c.try_into().unwrap()))
        .collect())
}

// ===== Block ================================================================

impl Serialize for Block {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut st = s.serialize_struct("Block", 2)?;
        st.serialize_field("shape", &self.shape())?;
        let columns: BTreeMap<&str, &Column> = self.iter().collect();
        st.serialize_field("columns", &columns)?;
        st.end()
    }
}

#[derive(Deserialize)]
struct BlockRepr {
    #[serde(default)]
    shape: Vec<usize>,
    #[serde(default)]
    columns: BTreeMap<String, Column>,
}

impl<'de> Deserialize<'de> for Block {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Block, D::Error> {
        let r = BlockRepr::deserialize(d)?;
        let shape = r.shape;
        let has_columns = !r.columns.is_empty();
        let mut block = Block::with_capacity(r.columns.len());
        for (name, col) in r.columns {
            block.insert_column(name, col).map_err(de::Error::custom)?;
        }
        // Restore an N-D structural shape (e.g. a volumetric block); a plain
        // table's `[count]` is already established by the inserts, but an empty
        // schema-only table needs its explicit row count restored.
        if !shape.is_empty() && (!has_columns || shape.len() > 1) {
            block.set_shape(&shape).map_err(de::Error::custom)?;
        }
        Ok(block)
    }
}

// ===== SimBox ===============================================================

impl Serialize for SimBox {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let h = self.h_view();
        let vectors = [
            [h[[0, 0]], h[[0, 1]], h[[0, 2]]],
            [h[[1, 0]], h[[1, 1]], h[[1, 2]]],
            [h[[2, 0]], h[[2, 1]], h[[2, 2]]],
        ];
        let o = self.origin_view();
        let mut st = s.serialize_struct("SimBox", 4)?;
        st.serialize_field("vectors", &vectors)?;
        st.serialize_field("origin", &[o[0], o[1], o[2]])?;
        st.serialize_field("boundary", &self.pbc())?;
        st.serialize_field("cell_defined", &self.is_cell_defined())?;
        st.end()
    }
}

#[derive(Deserialize)]
struct SimBoxRepr {
    vectors: [[f64; 3]; 3],
    origin: [f64; 3],
    boundary: [bool; 3],
    cell_defined: bool,
}

impl<'de> Deserialize<'de> for SimBox {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<SimBox, D::Error> {
        let r = SimBoxRepr::deserialize(d)?;
        SimBox::new_cell(
            ndarray::arr2(&r.vectors),
            ndarray::arr1(&r.origin),
            r.boundary,
            r.cell_defined,
        )
        .map_err(|e| de::Error::custom(format!("{e:?}")))
    }
}

// ===== Frame ================================================================

impl Serialize for Frame {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let has_box = self.simbox.is_some();
        let mut st = s.serialize_struct("Frame", 3 + has_box as usize)?;
        st.serialize_field("version", &FRAME_SCHEMA_VERSION)?;
        let blocks: BTreeMap<&str, &Block> = self.iter().collect();
        st.serialize_field("blocks", &blocks)?;
        let meta: BTreeMap<&str, &MetaValue> =
            self.meta.iter().map(|(k, v)| (k.as_str(), v)).collect();
        st.serialize_field("meta", &meta)?;
        if let Some(sb) = &self.simbox {
            st.serialize_field("box", sb)?;
        }
        st.end()
    }
}

#[derive(Deserialize)]
struct FrameRepr {
    version: u32,
    #[serde(default)]
    blocks: BTreeMap<String, Block>,
    #[serde(default)]
    meta: BTreeMap<String, MetaValue>,
    #[serde(default, rename = "box")]
    simbox: Option<SimBox>,
}

impl<'de> Deserialize<'de> for Frame {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Frame, D::Error> {
        let r = FrameRepr::deserialize(d)?;
        if r.version != FRAME_SCHEMA_VERSION {
            return Err(de::Error::custom(format!(
                "unsupported frame schema version {}; expected {}",
                r.version, FRAME_SCHEMA_VERSION
            )));
        }
        let mut frame = Frame::with_capacity(r.blocks.len());
        for (name, block) in r.blocks {
            frame.insert(name, block);
        }
        frame.meta = MetaMap::with_capacity(r.meta.len());
        frame.meta.extend(r.meta);
        frame.simbox = r.simbox;
        Ok(frame)
    }
}
