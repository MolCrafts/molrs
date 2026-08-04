//! The validator — a standalone judge of a Frame against the vocabulary.
//!
//! `Validator` deliberately does **not** live on `Frame`. It is its own type,
//! it takes a frame as a parameter, and it depends on nothing above
//! `core::store`: it knows nothing of `io`, `ff` or `compute`.
//!
//! It is generic over [`FrameAccess`] rather than tied to `Frame`, so the same
//! validator judges an owned `Frame` and a borrowed `FrameView` without
//! changes — including the `FrameView` that crosses the CXX bridge.

use super::block::RowKind;
use super::violation::{
    InstancePath, MAX_CELL_VIOLATIONS_PER_COLUMN, SchemaReport, Violation, ViolationKind,
};
use super::{block, column, consts};
use crate::store::block::{BlockAccess, DType};
use crate::store::frame_access::FrameAccess;
use std::collections::HashMap;

/// Judges a frame against the canonical vocabulary, plus any caller
/// annotations layered on top.
///
/// # Annotations may extend, never redefine
///
/// A caller can declare keys the vocabulary does not know — format-local
/// columns, perceived facts, per-instance force-field parameters. It cannot
/// redefine a key the vocabulary owns: `x` is `Float` and no annotation makes
/// it otherwise. Were `{"x": Int}` allowed, anyone could route around the
/// convention by declaring their way out of it, and the enforcement would be
/// worth nothing.
#[derive(Debug, Clone, Default)]
pub struct Validator {
    annotations: HashMap<String, DType>,
}

impl Validator {
    /// The canonical validator: the committed vocabulary, no annotations.
    pub fn canonical() -> Self {
        Validator::default()
    }

    /// Layer caller annotations on top.
    ///
    /// Returns [`ViolationKind::AnnotationConflict`] if an annotation names a
    /// key the canonical vocabulary already defines with a different dtype.
    /// Re-declaring a canonical key with its *own* dtype is a harmless no-op.
    pub fn with_annotations<I, S>(mut self, annotations: I) -> Result<Self, Violation>
    where
        I: IntoIterator<Item = (S, DType)>,
        S: Into<String>,
    {
        for (key, dtype) in annotations {
            let key: String = key.into();
            if let Some(spec) = column(&key)
                && spec.dtype != dtype
            {
                return Err(Violation {
                    path: InstancePath::Column {
                        block: "<annotation>".to_string(),
                        col: key,
                    },
                    kind: ViolationKind::AnnotationConflict {
                        canonical: spec.dtype,
                        requested: dtype,
                    },
                });
            }
            self.annotations.insert(key, dtype);
        }
        Ok(self)
    }

    /// Declared dtype for a key: canonical first, then annotations. `None` if
    /// the key is unconstrained.
    pub fn dtype_of(&self, key: &str) -> Option<DType> {
        column(key)
            .map(|s| s.dtype)
            .or_else(|| self.annotations.get(key).copied())
    }

    /// Every violation in `frame`. Never fails; an empty report means the frame
    /// conforms.
    ///
    /// Collects exhaustively — it does not stop at the first problem, because a
    /// caller fixing a file wants the whole list, not one round trip per bad
    /// column.
    pub fn check<FA: FrameAccess + ?Sized>(&self, frame: &FA) -> SchemaReport {
        let mut report = SchemaReport::new();

        let names: Vec<String> = frame.block_keys().iter().map(|s| s.to_string()).collect();
        let mut nrows: HashMap<String, usize> = HashMap::new();
        for name in &names {
            if let Some(n) = frame
                .visit_block(name, |b: &dyn BlockAccess| b.nrows())
                .flatten()
            {
                nrows.insert(name.clone(), n);
            }
        }

        for name in &names {
            self.check_columns(frame, name, &mut report);
            self.check_endpoints(frame, name, &nrows, &mut report);
        }

        report.sort();
        report
    }

    /// `Err` iff [`check`](Self::check) is non-empty.
    pub fn validate<FA: FrameAccess + ?Sized>(&self, frame: &FA) -> Result<(), SchemaReport> {
        self.check(frame).into_result()
    }

    fn check_columns<FA: FrameAccess + ?Sized>(
        &self,
        frame: &FA,
        name: &str,
        report: &mut SchemaReport,
    ) {
        let spec = block(name);
        let found = frame.visit_block(name, |b: &dyn BlockAccess| {
            let keys: Vec<String> = b.column_keys().iter().map(|s| s.to_string()).collect();
            let mut out = Vec::with_capacity(keys.len());
            for k in keys {
                let dtype = b.column_dtype(&k);
                let shape = b.column_shape(&k);
                out.push((k, dtype, shape));
            }
            out
        });
        let Some(cols) = found else { return };

        for (col, dtype, shape) in &cols {
            if let (Some(expected), Some(found)) = (self.dtype_of(col), *dtype)
                && found != expected
            {
                report.push(Violation::column(
                    name,
                    col,
                    ViolationKind::WrongDtype { expected, found },
                ));
            }
            if let (Some(spec), Some(shape)) = (column(col), shape.as_ref())
                && !spec.shape.admits(shape)
            {
                report.push(Violation::column(
                    name,
                    col,
                    ViolationKind::WrongShape {
                        expected: spec.shape,
                        found: shape.clone(),
                    },
                ));
            }
            if let Some(s) = spec
                && !s.open
                && column(col).is_none()
                && !self.annotations.contains_key(col)
            {
                report.push(Violation::column(name, col, ViolationKind::UnknownColumn));
            }
        }

        for key in spec.map(|s| s.required).unwrap_or(&[]) {
            if !cols.iter().any(|(c, _, _)| c == key) {
                report.push(Violation::column(name, *key, ViolationKind::MissingColumn));
            }
        }
    }

    fn check_endpoints<FA: FrameAccess + ?Sized>(
        &self,
        frame: &FA,
        name: &str,
        nrows: &HashMap<String, usize>,
        report: &mut SchemaReport,
    ) {
        let spec = block(name);
        let (endpoint_cols, target): (Vec<&str>, &str) = match spec {
            Some(s) if matches!(s.row_kind, RowKind::Relation { .. }) => {
                let e = s.endpoints.expect("relation spec carries endpoints");
                (e.columns.to_vec(), e.target)
            }
            // A block with no spec is legal (MolGraph mints one per relation
            // kind); infer that it is a relation from the endpoint columns it
            // carries, so those get range-checked too.
            _ => (self.inferred_endpoints(frame, name), "atoms"),
        };
        if endpoint_cols.is_empty() {
            return;
        }
        let Some(&target_rows) = nrows.get(target) else {
            return;
        };
        for col in endpoint_cols {
            self.check_range(frame, name, col, target, target_rows, report);
        }
    }

    fn inferred_endpoints<FA: FrameAccess + ?Sized>(&self, frame: &FA, name: &str) -> Vec<&str> {
        frame
            .visit_block(name, |b: &dyn BlockAccess| {
                consts::ENDPOINTS
                    .iter()
                    .copied()
                    .filter(|k| b.contains_key(k))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    }

    fn check_range<FA: FrameAccess + ?Sized>(
        &self,
        frame: &FA,
        name: &str,
        col: &str,
        target: &str,
        target_rows: usize,
        report: &mut SchemaReport,
    ) {
        // A wrong-dtype endpoint column has already been reported by
        // `check_columns`. Reading it as uint here would return None and
        // silently skip the range check — which is exactly the failure this
        // module exists to remove, so the dtype report is what covers it.
        let Some(values) = frame.get_uint(name, col) else {
            return;
        };
        let mut reported = 0usize;
        let mut extra = 0usize;
        for (row, &v) in values.iter().enumerate() {
            if (v as usize) < target_rows {
                continue;
            }
            if reported < MAX_CELL_VIOLATIONS_PER_COLUMN {
                report.push(Violation {
                    path: InstancePath::Cell {
                        block: name.to_string(),
                        col: col.to_string(),
                        row,
                    },
                    kind: ViolationKind::IndexOutOfRange {
                        value: v as u64,
                        target: target.to_string(),
                        target_nrows: target_rows,
                    },
                });
                reported += 1;
            } else {
                extra += 1;
            }
        }
        if extra > 0 {
            report.push(Violation::column(
                name,
                col,
                ViolationKind::TruncatedCells { extra },
            ));
        }
    }
}
