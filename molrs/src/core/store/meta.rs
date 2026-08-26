//! Typed frame metadata.
//!
//! [`MetaMap`] is the only frame-metadata container.  Every value carries an
//! exact scalar or fixed-vector dtype; metadata is never routed through a
//! string representation.

use std::collections::HashMap;

/// Exact metadata value stored on a frame.
#[derive(Clone, Debug, PartialEq)]
pub enum MetaValue {
    Bool(bool),
    I32(i32),
    I64(i64),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    String(String),
    Bool3([bool; 3]),
    I32x3([i32; 3]),
    I64x3([i64; 3]),
    U32x3([u32; 3]),
    U64x3([u64; 3]),
    F32x3([f32; 3]),
    F64x3([f64; 3]),
    /// Symmetric stress tensor in `(xx, yy, zz, xy, xz, yz)` order.
    F32x6([f32; 6]),
    /// Symmetric stress tensor in `(xx, yy, zz, xy, xz, yz)` order.
    F64x6([f64; 6]),
    /// Row-major 3x3 tensor.
    F32x9([f32; 9]),
    /// Row-major 3x3 tensor.
    F64x9([f64; 9]),
    /// Nested JSON document value. Frame group attributes are a document,
    /// not a typed scalar map: objects, arrays, and nulls land here.
    Json(serde_json::Value),
}

impl MetaValue {
    /// Stable dtype tag used by every serialized representation and binding.
    pub const fn dtype(&self) -> &'static str {
        match self {
            Self::Bool(_) => "bool",
            Self::I32(_) => "i32",
            Self::I64(_) => "i64",
            Self::U32(_) => "u32",
            Self::U64(_) => "u64",
            Self::F32(_) => "f32",
            Self::F64(_) => "f64",
            Self::String(_) => "string",
            Self::Bool3(_) => "bool3",
            Self::I32x3(_) => "i32x3",
            Self::I64x3(_) => "i64x3",
            Self::U32x3(_) => "u32x3",
            Self::U64x3(_) => "u64x3",
            Self::F32x3(_) => "f32x3",
            Self::F64x3(_) => "f64x3",
            Self::F32x6(_) => "f32x6",
            Self::F64x6(_) => "f64x6",
            Self::F32x9(_) => "f32x9",
            Self::F64x9(_) => "f64x9",
            Self::Json(_) => "json",
        }
    }

    pub const fn as_bool(&self) -> Option<bool> {
        if let Self::Bool(value) = self {
            Some(*value)
        } else {
            None
        }
    }
    pub const fn as_i32(&self) -> Option<i32> {
        if let Self::I32(value) = self {
            Some(*value)
        } else {
            None
        }
    }
    pub const fn as_i64(&self) -> Option<i64> {
        if let Self::I64(value) = self {
            Some(*value)
        } else {
            None
        }
    }
    pub const fn as_u32(&self) -> Option<u32> {
        if let Self::U32(value) = self {
            Some(*value)
        } else {
            None
        }
    }
    pub const fn as_u64(&self) -> Option<u64> {
        if let Self::U64(value) = self {
            Some(*value)
        } else {
            None
        }
    }
    pub const fn as_f32(&self) -> Option<f32> {
        if let Self::F32(value) = self {
            Some(*value)
        } else {
            None
        }
    }
    pub const fn as_f64(&self) -> Option<f64> {
        if let Self::F64(value) = self {
            Some(*value)
        } else {
            None
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        if let Self::String(value) = self {
            Some(value)
        } else {
            None
        }
    }

    /// Lossless JSON object used by Zarr attributes.
    pub fn to_json_value(&self) -> serde_json::Value {
        use serde_json::{Value, json};
        let value = match self {
            Self::Bool(v) => json!(v),
            Self::I32(v) => json!(v),
            Self::I64(v) => json!(v),
            Self::U32(v) => json!(v),
            Self::U64(v) => json!(v),
            Self::F32(v) => json!(v),
            Self::F64(v) => json!(v),
            Self::String(v) => Value::String(v.clone()),
            Self::Bool3(v) => json!(v),
            Self::I32x3(v) => json!(v),
            Self::I64x3(v) => json!(v),
            Self::U32x3(v) => json!(v),
            Self::U64x3(v) => json!(v),
            Self::F32x3(v) => json!(v),
            Self::F64x3(v) => json!(v),
            Self::F32x6(v) => json!(v),
            Self::F64x6(v) => json!(v),
            Self::F32x9(v) => json!(v),
            Self::F64x9(v) => json!(v),
            Self::Json(v) => v.clone(),
        };
        json!({ "dtype": self.dtype(), "value": value })
    }

    /// JSON as stored on a Zarr group attribute: the payload, not the typed
    /// `{dtype, value}` envelope. The MolRec document contract is raw JSON.
    pub fn to_attr_value(&self) -> serde_json::Value {
        use serde_json::{Value, json};
        match self {
            Self::Bool(v) => json!(v),
            Self::I32(v) => json!(v),
            Self::I64(v) => json!(v),
            Self::U32(v) => json!(v),
            Self::U64(v) => json!(v),
            Self::F32(v) => json!(v),
            Self::F64(v) => json!(v),
            Self::String(v) => Value::String(v.clone()),
            Self::Bool3(v) => json!(v),
            Self::I32x3(v) => json!(v),
            Self::I64x3(v) => json!(v),
            Self::U32x3(v) => json!(v),
            Self::U64x3(v) => json!(v),
            Self::F32x3(v) => json!(v),
            Self::F64x3(v) => json!(v),
            Self::F32x6(v) => json!(v),
            Self::F64x6(v) => json!(v),
            Self::F32x9(v) => json!(v),
            Self::F64x9(v) => json!(v),
            Self::Json(v) => v.clone(),
        }
    }

    /// Decode a document attribute. Accepts the typed `{dtype, value}`
    /// envelope and raw JSON alike.
    pub fn from_attr_value(value: &serde_json::Value) -> Self {
        if let Ok(typed) = Self::from_json_value(value) {
            return typed;
        }
        match value {
            serde_json::Value::Bool(v) => Self::Bool(*v),
            serde_json::Value::Number(n) if n.is_i64() => Self::I64(n.as_i64().unwrap()),
            serde_json::Value::Number(n) if n.is_u64() => Self::U64(n.as_u64().unwrap()),
            serde_json::Value::Number(n) => Self::F64(n.as_f64().unwrap_or(f64::NAN)),
            serde_json::Value::String(v) => Self::String(v.clone()),
            other => Self::Json(other.clone()),
        }
    }

    /// Decode the exact JSON object emitted by [`Self::to_json_value`].
    pub fn from_json_value(value: &serde_json::Value) -> Result<Self, String> {
        let object = value
            .as_object()
            .ok_or_else(|| "typed metadata must be an object".to_string())?;
        if object.len() != 2 || !object.contains_key("dtype") || !object.contains_key("value") {
            return Err("typed metadata object must contain exactly `dtype` and `value`".into());
        }
        let dtype = object["dtype"]
            .as_str()
            .ok_or_else(|| "metadata dtype must be a string".to_string())?;
        let payload = &object["value"];
        let wrong_type = || format!("metadata `{dtype}` payload has the wrong JSON type");
        macro_rules! array {
            ($ty:ty, $len:expr, $variant:ident) => {{
                let values: Vec<$ty> = serde_json::from_value(payload.clone())
                    .map_err(|e| format!("metadata `{dtype}` payload: {e}"))?;
                let values: [$ty; $len] = values.try_into().map_err(|v: Vec<$ty>| {
                    format!(
                        "metadata `{dtype}` expects {} values, got {}",
                        $len,
                        v.len()
                    )
                })?;
                Ok(Self::$variant(values))
            }};
        }
        match dtype {
            "bool" => payload.as_bool().map(Self::Bool).ok_or_else(wrong_type),
            "i32" => payload.as_i64().ok_or_else(wrong_type).and_then(|raw| {
                i32::try_from(raw)
                    .map(Self::I32)
                    .map_err(|_| "i32 metadata out of range".into())
            }),
            "i64" => payload.as_i64().map(Self::I64).ok_or_else(wrong_type),
            "u32" => payload.as_u64().ok_or_else(wrong_type).and_then(|raw| {
                u32::try_from(raw)
                    .map(Self::U32)
                    .map_err(|_| "u32 metadata out of range".into())
            }),
            "u64" => payload.as_u64().map(Self::U64).ok_or_else(wrong_type),
            "f32" => payload.as_f64().ok_or_else(wrong_type).and_then(|raw| {
                let narrowed = raw as f32;
                if raw.is_finite() && !narrowed.is_finite() {
                    Err("f32 metadata out of range".into())
                } else {
                    Ok(Self::F32(narrowed))
                }
            }),
            "f64" => payload.as_f64().map(Self::F64).ok_or_else(wrong_type),
            "string" => payload
                .as_str()
                .map(|v| Self::String(v.to_owned()))
                .ok_or_else(|| "string metadata payload has the wrong JSON type".into()),
            "bool3" => array!(bool, 3, Bool3),
            "i32x3" => array!(i32, 3, I32x3),
            "i64x3" => array!(i64, 3, I64x3),
            "u32x3" => array!(u32, 3, U32x3),
            "u64x3" => array!(u64, 3, U64x3),
            "f32x3" => array!(f32, 3, F32x3),
            "f64x3" => array!(f64, 3, F64x3),
            "f32x6" => array!(f32, 6, F32x6),
            "f64x6" => array!(f64, 6, F64x6),
            "f32x9" => array!(f32, 9, F32x9),
            "f64x9" => array!(f64, 9, F64x9),
            "json" => Ok(Self::Json(payload.clone())),
            other => Err(format!("unknown metadata dtype `{other}`")),
        }
    }
}

macro_rules! impl_from_meta {
    ($ty:ty, $variant:ident) => {
        impl From<$ty> for MetaValue {
            fn from(value: $ty) -> Self {
                Self::$variant(value)
            }
        }
    };
}
impl_from_meta!(bool, Bool);
impl_from_meta!(i32, I32);
impl_from_meta!(i64, I64);
impl_from_meta!(u32, U32);
impl_from_meta!(u64, U64);
impl_from_meta!(f32, F32);
impl_from_meta!(f64, F64);
impl_from_meta!(String, String);
impl_from_meta!([bool; 3], Bool3);
impl_from_meta!([i32; 3], I32x3);
impl_from_meta!([i64; 3], I64x3);
impl_from_meta!([u32; 3], U32x3);
impl_from_meta!([u64; 3], U64x3);
impl_from_meta!([f32; 3], F32x3);
impl_from_meta!([f64; 3], F64x3);
impl_from_meta!([f32; 6], F32x6);
impl_from_meta!([f64; 6], F64x6);
impl_from_meta!([f32; 9], F32x9);
impl_from_meta!([f64; 9], F64x9);

impl From<&str> for MetaValue {
    fn from(value: &str) -> Self {
        Self::String(value.to_owned())
    }
}

/// The unique metadata map used by owned and borrowed frames.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MetaMap(HashMap<String, MetaValue>);

impl MetaMap {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_capacity(capacity: usize) -> Self {
        Self(HashMap::with_capacity(capacity))
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn clear(&mut self) {
        self.0.clear();
    }
    pub fn contains_key(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }
    pub fn get(&self, key: &str) -> Option<&MetaValue> {
        self.0.get(key)
    }
    pub fn get_mut(&mut self, key: &str) -> Option<&mut MetaValue> {
        self.0.get_mut(key)
    }
    pub fn insert(
        &mut self,
        key: impl Into<String>,
        value: impl Into<MetaValue>,
    ) -> Option<MetaValue> {
        self.0.insert(key.into(), value.into())
    }
    pub fn remove(&mut self, key: &str) -> Option<MetaValue> {
        self.0.remove(key)
    }
    pub fn iter(&self) -> impl Iterator<Item = (&String, &MetaValue)> {
        self.0.iter()
    }
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.0.keys()
    }
    pub fn values(&self) -> impl Iterator<Item = &MetaValue> {
        self.0.values()
    }
    pub fn extend(&mut self, values: impl IntoIterator<Item = (String, MetaValue)>) {
        self.0.extend(values);
    }
}

impl IntoIterator for MetaMap {
    type Item = (String, MetaValue);
    type IntoIter = std::collections::hash_map::IntoIter<String, MetaValue>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a MetaMap {
    type Item = (&'a String, &'a MetaValue);
    type IntoIter = std::collections::hash_map::Iter<'a, String, MetaValue>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip_preserves_every_dtype() {
        let values = [
            MetaValue::Bool(true),
            MetaValue::I32(-3),
            MetaValue::I64(i64::MIN + 7),
            MetaValue::U32(9),
            MetaValue::U64(u64::MAX - 7),
            MetaValue::F32(1.25),
            MetaValue::F64(-2.5),
            MetaValue::String("x".into()),
            MetaValue::F64x3([1.0, 2.0, 3.0]),
            MetaValue::F64x6([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            MetaValue::F64x9([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        ];
        for value in values {
            assert_eq!(
                MetaValue::from_json_value(&value.to_json_value()).unwrap(),
                value
            );
        }
    }

    #[test]
    fn untyped_json_is_rejected() {
        assert!(MetaValue::from_json_value(&serde_json::json!("legacy")).is_err());
    }

    #[test]
    fn overflowing_f32_json_is_rejected() {
        assert!(
            MetaValue::from_json_value(&serde_json::json!({
                "dtype": "f32",
                "value": f64::MAX,
            }))
            .is_err()
        );
    }
}
