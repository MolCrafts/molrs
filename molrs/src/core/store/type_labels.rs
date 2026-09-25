//! Type names and the Frame's type-id contract.
//!
//! One grammar for the name of a type (atom, pair, bond, angle, dihedral,
//! improper), [`TypeName`], and one rule for turning a Frame's per-row type
//! labels into dense 1-based ids, [`TypeLabels`]. Both live in `core` so that
//! format writers and force-field code share them without naming each other.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::core::store::frame_access::FrameAccess;
use crate::core::store::keys;
use crate::core::types::Idx;

/// Starts the qualifier of a [`TypeName`]; reserved inside endpoint labels.
const QUALIFIER: char = '@';
/// Separates the fields of a qualifier; reserved inside a qualifier field.
const FIELD: char = '_';
/// Endpoint separator when no endpoint label contains `-`.
const DASH: &str = "-";
/// Endpoint separator when some endpoint label contains `-`.
const WIDE: &str = "::";

/// True if `token` is a signed integer: an optional leading `+` / `-`, then at
/// least one ASCII digit and nothing else.
pub(crate) fn is_int_token(token: &str) -> bool {
    let digits = token.strip_prefix(['+', '-']).unwrap_or(token);
    !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
}

/// The name of a type: its endpoint labels, plus an optional qualifier.
///
/// # Grammar
///
/// `endpoints[@qualifier]`
///
/// - **Endpoints** are joined with `-` (`c3-c3-h1`), or with `::` when any
///   endpoint label itself contains `-` (`tip3p-O::tip3p-H`). Splitting tries
///   `::` first, then `-`. An empty endpoint is a position, not noise:
///   `-CT-CT-` has four endpoints, the outer two empty (a wildcard).
/// - **Qualifier.** The first `@` starts the qualifier: `_`-separated fields
///   holding the values a type depends on that are not a function of its
///   endpoint labels (e.g. bond orders, `C_3-C_R@1.5`).
///
/// # Reserved characters
///
/// `@` may not appear in an endpoint label ([`TypeName::join`] and
/// [`TypeName::pair`] reject it), and neither `_` nor `@` may appear in a
/// qualifier field ([`TypeName::with_qualifier`] rejects them). `-` and `::`
/// are separators as described above.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TypeName(String);

impl TypeName {
    /// Join endpoint labels into a name: `-` between them, or `::` when any
    /// label contains `-`.
    ///
    /// `Err` naming the part when a part contains `@`, which would be read
    /// back as the start of a qualifier.
    pub fn join(parts: &[&str]) -> Result<TypeName, String> {
        if let Some(bad) = parts.iter().find(|p| p.contains(QUALIFIER)) {
            return Err(format!(
                "type-name part {bad:?} contains '{QUALIFIER}', which starts a qualifier"
            ));
        }
        let sep = if parts.iter().any(|p| p.contains(DASH)) {
            WIDE
        } else {
            DASH
        };
        Ok(TypeName(parts.join(sep)))
    }

    /// The name of the pair of atom types `a` and `b`: `a` itself for a
    /// self-pair, else [`TypeName::join`] of the two. Same `@` rule as `join`.
    pub fn pair(a: &str, b: &str) -> Result<TypeName, String> {
        if a == b {
            Self::join(&[a]).map(|_| TypeName(a.to_owned()))
        } else {
            Self::join(&[a, b])
        }
    }

    /// The whole name, qualifier included.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// The endpoint part (before the first `@`) and the qualifier, if any.
    fn split_qualifier(&self) -> (&str, Option<&str>) {
        match self.0.split_once(QUALIFIER) {
            Some((head, qualifier)) => (head, Some(qualifier)),
            None => (&self.0, None),
        }
    }

    /// The endpoint labels, in order: the part before the qualifier, split on
    /// `::` when it contains `::`, else on `-`. Empty positions are kept.
    pub fn endpoints(&self) -> Vec<&str> {
        let (head, _) = self.split_qualifier();
        if head.contains(WIDE) {
            head.split(WIDE).collect()
        } else {
            head.split(DASH).collect()
        }
    }

    /// The qualifier: the text after the first `@`, or `None`.
    pub fn qualifier(&self) -> Option<&str> {
        self.split_qualifier().1
    }

    /// This name with the qualifier `@f1_f2_…` appended.
    ///
    /// `Err` when the name already has a qualifier, when `fields` is empty,
    /// or when a field is empty or contains `_` or `@`.
    pub fn with_qualifier(&self, fields: &[&str]) -> Result<TypeName, String> {
        if self.qualifier().is_some() {
            return Err(format!("type name {:?} already has a qualifier", self.0));
        }
        if fields.is_empty() {
            return Err(format!(
                "qualifier for type name {:?} has no fields",
                self.0
            ));
        }
        if let Some(bad) = fields
            .iter()
            .find(|f| f.is_empty() || f.contains(FIELD) || f.contains(QUALIFIER))
        {
            return Err(format!(
                "qualifier field {bad:?} for type name {:?} is empty or contains \
                 '{FIELD}' or '{QUALIFIER}'",
                self.0
            ));
        }
        let joined = fields.join(&FIELD.to_string());
        Ok(TypeName(format!("{}{QUALIFIER}{joined}", self.0)))
    }

    /// The same term read from the other end.
    ///
    /// The endpoints are reversed, keeping the separator (`-` or `::`) and
    /// empty positions. The qualifier is reordered by the endpoint count,
    /// since `TypeName` carries no category and the count fixes it up to
    /// dihedral / improper:
    ///
    /// - 2 endpoints (bond): the fields are unchanged; they describe the bond
    ///   as a whole.
    /// - 3 endpoints (angle): the first two fields are the i–j and j–k bond
    ///   values, so they swap (`a-b-c@1_1.5_2` → `c-b-a@1.5_1_2`); later
    ///   fields are unchanged. A qualifier with fewer than two fields is
    ///   unchanged.
    /// - 4 endpoints (dihedral, improper): the fields are unchanged; they
    ///   describe the central bond and the term as a whole.
    /// - Any other count: the fields are unchanged.
    pub fn reversed(&self) -> TypeName {
        let (head, qualifier) = self.split_qualifier();
        let sep = if head.contains(WIDE) { WIDE } else { DASH };
        let mut ends: Vec<&str> = head.split(sep).collect();
        ends.reverse();
        let mut out = ends.join(sep);
        if let Some(qualifier) = qualifier {
            let mut fields: Vec<&str> = qualifier.split(FIELD).collect();
            if ends.len() == 3 && fields.len() >= 2 {
                fields.swap(0, 1);
            }
            out.push(QUALIFIER);
            out.push_str(&fields.join(&FIELD.to_string()));
        }
        TypeName(out)
    }

    /// The byte-wise smaller of this name and [`TypeName::reversed`],
    /// qualifier included, so both orientations of one term share it.
    pub fn canonical(&self) -> TypeName {
        let reversed = self.reversed();
        if reversed.0 < self.0 {
            reversed
        } else {
            self.clone()
        }
    }
}

impl fmt::Display for TypeName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for TypeName {
    fn from(name: String) -> Self {
        TypeName(name)
    }
}

/// Resolved type space of one block: per-row ids, ordered labels, type count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockTypes {
    type_ids: Vec<Idx>,
    labels: Option<Vec<String>>,
    n_types: usize,
}

impl BlockTypes {
    /// The 1-based type id of every row, in row order. Empty for a block that
    /// exists only through its inventory meta key.
    pub fn type_ids(&self) -> &[Idx] {
        &self.type_ids
    }

    /// The ordered labels (label `i` has id `i + 1`), or `None` when the ids
    /// carry no labels (pure-integer labels, or a `type_id`-only block with no
    /// inventory).
    pub fn labels(&self) -> Option<&[String]> {
        self.labels.as_deref()
    }

    /// The number of types: the largest id, the number of labels, or at least
    /// 1 for `atoms`.
    pub fn n_types(&self) -> usize {
        self.n_types
    }

    /// Distinct names, sorted byte-wise, or numerically when every name is an
    /// integer (`2` before `10`).
    fn sorted(names: impl IntoIterator<Item = String>) -> Vec<String> {
        let mut items: Vec<String> = names.into_iter().collect();
        items.sort();
        items.dedup();
        if !items.is_empty() && items.iter().all(|s| is_int_token(s)) {
            items.sort_by_key(|s| s.parse::<i64>().unwrap_or(0));
        }
        items
    }

    /// Labels of the inventory meta value `"1:C,2:H"`, ordered by id.
    ///
    /// `Err` naming `key` and the token for a pair without `:`, a non-integer
    /// id, an empty label or a repeated id.
    fn parse_inventory(key: &str, raw: &str) -> Result<Vec<String>, String> {
        if raw.trim().is_empty() {
            return Ok(Vec::new());
        }
        let mut pairs: Vec<(u64, String)> = Vec::new();
        let mut seen: HashSet<u64> = HashSet::new();
        for token in raw.split(',') {
            let Some((id, label)) = token.split_once(':') else {
                return Err(format!(
                    "meta {key:?}: entry {token:?} is not an `id:label` pair"
                ));
            };
            let id: u64 = id.trim().parse().map_err(|_| {
                format!("meta {key:?}: entry {token:?} has a non-integer id {id:?}")
            })?;
            let label = label.trim();
            if label.is_empty() {
                return Err(format!("meta {key:?}: entry {token:?} has an empty label"));
            }
            if !seen.insert(id) {
                return Err(format!("meta {key:?}: entry {token:?} repeats id {id}"));
            }
            pairs.push((id, label.to_owned()));
        }
        pairs.sort_by_key(|(id, _)| *id);
        Ok(pairs.into_iter().map(|(_, label)| label).collect())
    }

    /// Resolve `block`, merging the inventory under `meta_key`.
    fn resolve(
        frame: &impl FrameAccess,
        block: &str,
        meta_key: &str,
        collapse: bool,
    ) -> Result<Option<BlockTypes>, String> {
        let n = frame
            .visit_block(block, |b| b.nrows().unwrap_or(0))
            .unwrap_or(0);
        let inventory = match frame.meta_ref().get(meta_key) {
            None => Vec::new(),
            Some(value) => {
                let raw = value
                    .as_str()
                    .ok_or_else(|| format!("meta {meta_key:?} must be a string"))?;
                Self::parse_inventory(meta_key, raw)?
            }
        };
        let has_inventory = !inventory.is_empty();
        let min_types = if block == "atoms" { 1 } else { 0 };
        let canonical = |name: &str| -> String {
            if collapse {
                TypeName::from(name.to_owned()).canonical().0
            } else {
                name.to_owned()
            }
        };

        if n == 0 {
            if inventory.is_empty() {
                return Ok(None);
            }
            let n_types = inventory.len();
            return Ok(Some(BlockTypes {
                type_ids: Vec::new(),
                labels: Some(inventory),
                n_types,
            }));
        }

        // String labels take precedence over `type_id`.
        if let Some(col) = frame.get_string(block, keys::TYPE) {
            let types: Vec<String> = (0..n).map(|i| canonical(&col[[i]])).collect();
            if types.iter().any(|t| t.trim().is_empty()) {
                return Err(format!(
                    "{block}: empty type label; every row needs a non-empty type"
                ));
            }
            let unique = Self::sorted(types.iter().cloned());
            let pure_int = unique.iter().all(|t| is_int_token(t));

            if pure_int && !has_inventory {
                let mut type_ids = Vec::with_capacity(n);
                let mut max_id: Idx = 0;
                for t in &types {
                    let id: Idx = t
                        .parse()
                        .map_err(|e| format!("{block}: type {t:?} is not a type id: {e}"))?;
                    if id == 0 {
                        return Err(format!("{block}: type id 0 is invalid (ids are 1-based)"));
                    }
                    max_id = max_id.max(id);
                    type_ids.push(id);
                }
                return Ok(Some(BlockTypes {
                    type_ids,
                    labels: None,
                    n_types: (max_id as usize).max(1),
                }));
            }

            let mut all: HashSet<String> = inventory.iter().map(|t| canonical(t)).collect();
            all.extend(unique);
            let ordered = Self::sorted(all);
            let ids: HashMap<&str, Idx> = ordered
                .iter()
                .enumerate()
                .map(|(i, s)| (s.as_str(), (i + 1) as Idx))
                .collect();
            let type_ids = types
                .iter()
                .map(|t| {
                    ids.get(t.as_str())
                        .copied()
                        .ok_or_else(|| format!("{block}: type {t:?} missing from its own ordering"))
                })
                .collect::<Result<Vec<Idx>, String>>()?;
            let n_types = ordered.len().max(min_types);
            return Ok(Some(BlockTypes {
                type_ids,
                labels: Some(ordered),
                n_types,
            }));
        }

        // Numeric ids (unsigned or signed; `type_id`, else a numeric `type`).
        let type_ids: Option<Vec<Idx>> = if let Some(col) = frame.get_uint(block, keys::TYPE_ID) {
            Some((0..n).map(|i| col[[i]]).collect())
        } else if let Some(col) = frame.get_int(block, keys::TYPE_ID) {
            Some((0..n).map(|i| col[[i]] as Idx).collect())
        } else if let Some(col) = frame.get_uint(block, keys::TYPE) {
            Some((0..n).map(|i| col[[i]]).collect())
        } else {
            frame
                .get_int(block, keys::TYPE)
                .map(|col| (0..n).map(|i| col[[i]] as Idx).collect())
        };
        let Some(type_ids) = type_ids else {
            return Err(format!(
                "frame[{block:?}] has {n} rows but neither 'type' nor 'type_id'; \
                 assign a 'type' or 'type_id' column first"
            ));
        };
        let max_id = type_ids.iter().copied().max().unwrap_or(0) as usize;
        let n_types = max_id.max(inventory.len()).max(min_types);
        let labels = if inventory.is_empty() {
            None
        } else {
            Some(inventory)
        };
        Ok(Some(BlockTypes {
            type_ids,
            labels,
            n_types,
        }))
    }
}

/// The Frame's type-id contract: per block, the type id of every row, the
/// ordered labels, and the number of types.
///
/// Covered blocks, each with its inventory meta key: `atoms`
/// ([`keys::ATOM_TYPE_LABELS`]), `bonds` ([`keys::BOND_TYPE_LABELS`]), `angles`
/// ([`keys::ANGLE_TYPE_LABELS`]), `dihedrals` ([`keys::DIHEDRAL_TYPE_LABELS`])
/// and `impropers` ([`keys::IMPROPER_TYPE_LABELS`]). No block is required.
///
/// # Id rules
///
/// - Ids are 1-based and dense; label `i` of [`BlockTypes::labels`] has id
///   `i + 1`.
/// - A string `type` column wins over `type_id` when both are present. Every
///   label must be non-empty.
/// - When every label is an integer (`"3"`, `"10"`) and the block has no
///   inventory, each label is its own id and no labels are kept; id `0` is an
///   error.
/// - Otherwise the labels get ids in sorted order: byte-wise, or numerically
///   when every label is an integer (`2` before `10`).
/// - The inventory meta key (`"1:C,2:H"`) declares types no row uses; they
///   are merged into the ordering, and a block with no rows is described by
///   its inventory alone. A malformed inventory (an entry without `:`, a
///   non-integer id, an empty label, a repeated id) is an error naming the key.
/// - A block with only a numeric `type_id` (or numeric `type`) column is taken
///   as is; its inventory, if any, supplies the labels.
/// - A block with rows but neither `type` nor `type_id` is an error.
/// - In `bonds`, `angles` and `dihedrals`, a label and its reverse name one
///   term, so they collapse to one id through [`TypeName::canonical`]
///   (qualified labels included). `impropers` labels never collapse:
///   reversing an improper moves its centre and names a different term.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TypeLabels {
    blocks: Vec<(&'static str, BlockTypes)>,
}

impl TypeLabels {
    /// Blocks covered, with their inventory key and whether reversed labels
    /// collapse.
    const BLOCKS: [(&'static str, &'static str, bool); 5] = [
        ("atoms", keys::ATOM_TYPE_LABELS, false),
        ("bonds", keys::BOND_TYPE_LABELS, true),
        ("angles", keys::ANGLE_TYPE_LABELS, true),
        ("dihedrals", keys::DIHEDRAL_TYPE_LABELS, true),
        ("impropers", keys::IMPROPER_TYPE_LABELS, false),
    ];

    /// Resolve every covered block of `frame` (see the type-level docs).
    pub fn from_frame(frame: &impl FrameAccess) -> Result<TypeLabels, String> {
        let mut blocks = Vec::new();
        for (block, meta_key, collapse) in Self::BLOCKS {
            if let Some(types) = BlockTypes::resolve(frame, block, meta_key, collapse)? {
                blocks.push((block, types));
            }
        }
        Ok(TypeLabels { blocks })
    }

    /// The resolved types of `block`, or `None` when the block has neither
    /// rows nor an inventory.
    pub fn block(&self, block: &str) -> Option<&BlockTypes> {
        self.blocks
            .iter()
            .find(|(name, _)| *name == block)
            .map(|(_, types)| types)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::store::block::Block;
    use crate::core::store::frame::Frame;
    use crate::core::store::keys;
    use crate::core::types::Idx;
    use ndarray::{ArrayD, IxDyn};

    // ------------------------------------------------------------------
    // Fixture builders
    // ------------------------------------------------------------------

    /// A block whose only column is the string `type` label per row.
    fn label_block(types: &[&str]) -> Block {
        let mut block = Block::new();
        block
            .insert(
                keys::TYPE,
                ArrayD::from_shape_vec(
                    IxDyn(&[types.len()]),
                    types.iter().map(|t| t.to_string()).collect::<Vec<String>>(),
                )
                .unwrap(),
            )
            .unwrap();
        block
    }

    /// A block whose only column is the numeric `type_id` per row.
    fn type_id_block(ids: &[Idx]) -> Block {
        let mut block = Block::new();
        block
            .insert(
                keys::TYPE_ID,
                ArrayD::from_shape_vec(IxDyn(&[ids.len()]), ids.to_vec()).unwrap(),
            )
            .unwrap();
        block
    }

    // ------------------------------------------------------------------
    // TypeName: endpoint grammar
    // ------------------------------------------------------------------

    #[test]
    fn join_uses_hyphen_between_plain_parts() {
        let name = TypeName::join(&["c3", "c3", "h1"]).unwrap();
        assert_eq!(name.as_str(), "c3-c3-h1");
    }

    #[test]
    fn join_uses_double_colon_when_a_part_contains_hyphen() {
        let name = TypeName::join(&["tip3p-O", "tip3p-H"]).unwrap();
        assert_eq!(name.as_str(), "tip3p-O::tip3p-H");
    }

    #[test]
    fn join_rejects_part_containing_at_and_names_it() {
        let Err(err) = TypeName::join(&["a@b", "c"]) else {
            panic!("a part containing '@' must be rejected");
        };
        assert!(err.contains("a@b"), "{err}");
    }

    #[test]
    fn pair_of_equal_types_is_the_type_itself() {
        assert_eq!(TypeName::pair("A", "A").unwrap().as_str(), "A");
    }

    #[test]
    fn pair_of_distinct_types_is_their_join() {
        assert_eq!(TypeName::pair("A", "B").unwrap().as_str(), "A-B");
        assert_eq!(
            TypeName::pair("tip3p-O", "tip3p-H").unwrap().as_str(),
            "tip3p-O::tip3p-H"
        );
    }

    #[test]
    fn pair_rejects_part_containing_at() {
        assert!(TypeName::pair("a@b", "c").is_err());
        assert!(TypeName::pair("a@b", "a@b").is_err());
    }

    #[test]
    fn endpoints_split_on_hyphen() {
        let name = TypeName::from(String::from("c3-c3-h1"));
        assert_eq!(name.endpoints(), vec!["c3", "c3", "h1"]);
    }

    #[test]
    fn endpoints_split_on_double_colon_first() {
        let name = TypeName::from(String::from("tip3p-O::tip3p-H"));
        assert_eq!(name.endpoints(), vec!["tip3p-O", "tip3p-H"]);
    }

    #[test]
    fn endpoints_of_single_type_is_itself() {
        let name = TypeName::from(String::from("A"));
        assert_eq!(name.endpoints(), vec!["A"]);
    }

    #[test]
    fn endpoints_keep_empty_wildcard_positions() {
        let name = TypeName::from(String::from("-CT-CT-"));
        assert_eq!(name.endpoints(), vec!["", "CT", "CT", ""]);
        assert_eq!(name.endpoints().len(), 4);
    }

    #[test]
    fn reversed_reverses_hyphen_endpoints() {
        let name = TypeName::from(String::from("h1-c3-c3"));
        assert_eq!(name.reversed().as_str(), "c3-c3-h1");
    }

    #[test]
    fn reversed_keeps_double_colon_separator() {
        let name = TypeName::from(String::from("tip3p-O::tip3p-H"));
        assert_eq!(name.reversed().as_str(), "tip3p-H::tip3p-O");
    }

    #[test]
    fn reversed_keeps_empty_wildcard_positions() {
        let name = TypeName::from(String::from("X-CT-CT-"));
        assert_eq!(name.reversed().as_str(), "-CT-CT-X");
        let wildcard = TypeName::from(String::from("-CT-CT-"));
        assert_eq!(wildcard.reversed().as_str(), "-CT-CT-");
        assert_eq!(wildcard.reversed().endpoints().len(), 4);
    }

    #[test]
    fn reversed_of_single_type_is_itself() {
        let name = TypeName::from(String::from("A"));
        assert_eq!(name.reversed().as_str(), "A");
    }

    #[test]
    fn canonical_picks_bytewise_smaller_orientation() {
        let backward = TypeName::from(String::from("h1-c3-c3"));
        let forward = TypeName::from(String::from("c3-c3-h1"));
        assert_eq!(backward.canonical().as_str(), "c3-c3-h1");
        assert_eq!(forward.canonical().as_str(), "c3-c3-h1");
    }

    #[test]
    fn canonical_honours_double_colon() {
        let name = TypeName::from(String::from("tip3p-O::tip3p-H"));
        assert_eq!(name.canonical().as_str(), "tip3p-H::tip3p-O");
    }

    #[test]
    fn canonical_keeps_four_wildcard_endpoints() {
        let name = TypeName::from(String::from("-CT-CT-"));
        let canonical = name.canonical();
        assert_eq!(canonical.as_str(), "-CT-CT-");
        assert_eq!(canonical.endpoints(), vec!["", "CT", "CT", ""]);
    }

    #[test]
    fn display_prints_the_whole_name() {
        let name = TypeName::from(String::from("C_3-C_R@1.5"));
        assert_eq!(format!("{name}"), "C_3-C_R@1.5");
    }

    // ------------------------------------------------------------------
    // TypeName: `@` qualifier
    // ------------------------------------------------------------------

    #[test]
    fn with_qualifier_appends_single_field() {
        let name = TypeName::join(&["C_3", "C_R"])
            .unwrap()
            .with_qualifier(&["1.5"])
            .unwrap();
        assert_eq!(name.as_str(), "C_3-C_R@1.5");
    }

    #[test]
    fn with_qualifier_joins_fields_with_underscore() {
        let name = TypeName::join(&["C_3", "C_R", "O_2"])
            .unwrap()
            .with_qualifier(&["1", "1.5", "2"])
            .unwrap();
        assert_eq!(name.as_str(), "C_3-C_R-O_2@1_1.5_2");
    }

    #[test]
    fn with_qualifier_rejects_already_qualified_name() {
        let name = TypeName::from(String::from("C_3-C_R@1.5"));
        assert!(name.with_qualifier(&["2"]).is_err());
    }

    #[test]
    fn with_qualifier_rejects_no_fields() {
        let name = TypeName::from(String::from("C_3-C_R"));
        assert!(name.with_qualifier(&[]).is_err());
    }

    #[test]
    fn with_qualifier_rejects_empty_field() {
        let name = TypeName::from(String::from("C_3-C_R"));
        assert!(name.with_qualifier(&[""]).is_err());
        assert!(name.with_qualifier(&["1", ""]).is_err());
    }

    #[test]
    fn with_qualifier_rejects_field_containing_underscore() {
        let name = TypeName::from(String::from("C_3-C_R"));
        assert!(name.with_qualifier(&["1_5"]).is_err());
    }

    #[test]
    fn with_qualifier_rejects_field_containing_at() {
        let name = TypeName::from(String::from("C_3-C_R"));
        assert!(name.with_qualifier(&["1@5"]).is_err());
    }

    #[test]
    fn qualifier_is_text_after_first_at() {
        let bond = TypeName::from(String::from("C_3-C_R@1.5"));
        assert_eq!(bond.qualifier(), Some("1.5"));
        let angle = TypeName::from(String::from("C_3-C_R-O_2@1_1.5_2"));
        assert_eq!(angle.qualifier(), Some("1_1.5_2"));
        let odd = TypeName::from(String::from("a-b@c@d"));
        assert_eq!(odd.qualifier(), Some("c@d"));
        assert_eq!(odd.endpoints(), vec!["a", "b"]);
    }

    #[test]
    fn qualifier_is_none_for_unqualified_name() {
        let name = TypeName::from(String::from("C_3-C_R"));
        assert_eq!(name.qualifier(), None);
    }

    #[test]
    fn endpoints_exclude_qualifier() {
        let name = TypeName::from(String::from("C_3-C_R@1.5"));
        assert_eq!(name.endpoints(), vec!["C_3", "C_R"]);
    }

    #[test]
    fn endpoints_exclude_qualifier_with_double_colon() {
        let name = TypeName::from(String::from("tip3p-O::tip3p-H@1"));
        assert_eq!(name.endpoints(), vec!["tip3p-O", "tip3p-H"]);
    }

    #[test]
    fn reversed_bond_keeps_qualifier_fields() {
        let name = TypeName::from(String::from("C_3-O_R@1.5"));
        assert_eq!(name.reversed().as_str(), "O_R-C_3@1.5");
    }

    #[test]
    fn reversed_qualified_double_colon_bond() {
        let name = TypeName::from(String::from("tip3p-O::tip3p-H@1"));
        assert_eq!(name.reversed().as_str(), "tip3p-H::tip3p-O@1");
    }

    #[test]
    fn reversed_angle_swaps_first_two_qualifier_fields() {
        let name = TypeName::from(String::from("C_3-C_R-O_2@1_1.5_2"));
        assert_eq!(name.reversed().as_str(), "O_2-C_R-C_3@1.5_1_2");
        assert_eq!(name.reversed().reversed().as_str(), "C_3-C_R-O_2@1_1.5_2");
    }

    #[test]
    fn reversed_angle_with_single_field_keeps_it() {
        let name = TypeName::from(String::from("a-b-c@7"));
        assert_eq!(name.reversed().as_str(), "c-b-a@7");
    }

    #[test]
    fn canonical_is_shared_by_both_angle_orientations() {
        let forward = TypeName::from(String::from("C_3-C_R-O_2@1_1.5_2"));
        let backward = TypeName::from(String::from("O_2-C_R-C_3@1.5_1_2"));
        assert_eq!(forward.canonical().as_str(), "C_3-C_R-O_2@1_1.5_2");
        assert_eq!(backward.canonical().as_str(), "C_3-C_R-O_2@1_1.5_2");
    }

    #[test]
    fn canonical_compares_qualifier_for_symmetric_endpoints() {
        // Endpoints read the same both ways; only the swapped qualifier differs.
        let forward = TypeName::from(String::from("a-b-a@1_2_0"));
        let backward = TypeName::from(String::from("a-b-a@2_1_0"));
        assert_eq!(forward.reversed().as_str(), "a-b-a@2_1_0");
        assert_eq!(forward.canonical().as_str(), "a-b-a@1_2_0");
        assert_eq!(backward.canonical().as_str(), "a-b-a@1_2_0");
    }

    #[test]
    fn canonical_of_qualified_bond_includes_qualifier() {
        let name = TypeName::from(String::from("O_R-C_3@1.5"));
        assert_eq!(name.canonical().as_str(), "C_3-O_R@1.5");
    }

    #[test]
    fn reversed_torsion_keeps_qualifier_fields() {
        let name = TypeName::from(String::from("H_-C_3-C_3-O_3@0.2_3_0"));
        assert_eq!(name.reversed().as_str(), "O_3-C_3-C_3-H_@0.2_3_0");
    }

    // ------------------------------------------------------------------
    // TypeLabels: the Frame's type-id contract
    // ------------------------------------------------------------------

    #[test]
    fn pure_integer_labels_map_to_themselves() {
        let mut frame = Frame::new();
        frame.insert("atoms", label_block(&["3", "1", "10"]));
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let atoms = labels.block("atoms").expect("atoms resolved");
        assert_eq!(atoms.type_ids(), &[3, 1, 10][..]);
        assert!(atoms.labels().is_none());
        assert_eq!(atoms.n_types(), 10);
    }

    #[test]
    fn integer_id_zero_is_rejected() {
        let mut frame = Frame::new();
        frame.insert("atoms", label_block(&["0", "1"]));
        assert!(TypeLabels::from_frame(&frame).is_err());
    }

    #[test]
    fn other_labels_get_dense_ids_in_sorted_order() {
        let mut frame = Frame::new();
        frame.insert("atoms", label_block(&["O", "C", "H", "C"]));
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let atoms = labels.block("atoms").expect("atoms resolved");
        assert_eq!(
            atoms.labels().map(|l| l.to_vec()),
            Some(vec!["C".to_string(), "H".to_string(), "O".to_string()])
        );
        assert_eq!(atoms.type_ids(), &[3, 1, 2, 1][..]);
        assert_eq!(atoms.n_types(), 3);
    }

    #[test]
    fn integer_labels_with_inventory_sort_numerically() {
        // Lexical order would put "10" before "2".
        let mut frame = Frame::new();
        frame.insert("atoms", label_block(&["10", "2"]));
        frame.meta.insert(keys::ATOM_TYPE_LABELS, "1:2,2:10");
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let atoms = labels.block("atoms").expect("atoms resolved");
        assert_eq!(
            atoms.labels().map(|l| l.to_vec()),
            Some(vec!["2".to_string(), "10".to_string()])
        );
        assert_eq!(atoms.type_ids(), &[2, 1][..]);
        assert_eq!(atoms.n_types(), 2);
    }

    #[test]
    fn inventory_meta_adds_unused_types() {
        let mut frame = Frame::new();
        frame.insert("atoms", label_block(&["H", "C"]));
        frame.meta.insert(keys::ATOM_TYPE_LABELS, "1:C,2:H,3:O");
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let atoms = labels.block("atoms").expect("atoms resolved");
        assert_eq!(
            atoms.labels().map(|l| l.to_vec()),
            Some(vec!["C".to_string(), "H".to_string(), "O".to_string()])
        );
        assert_eq!(atoms.type_ids(), &[2, 1][..]);
        assert_eq!(atoms.n_types(), 3);
    }

    #[test]
    fn inventory_only_block_has_labels_and_no_rows() {
        let mut frame = Frame::new();
        frame.insert("atoms", label_block(&["C"]));
        frame.meta.insert(keys::BOND_TYPE_LABELS, "1:c3-h1,2:c3-c3");
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let bonds = labels.block("bonds").expect("bonds from inventory");
        assert!(bonds.type_ids().is_empty());
        assert_eq!(
            bonds.labels().map(|l| l.to_vec()),
            Some(vec!["c3-h1".to_string(), "c3-c3".to_string()])
        );
        assert_eq!(bonds.n_types(), 2);
    }

    #[test]
    fn absent_block_without_inventory_is_none() {
        let mut frame = Frame::new();
        frame.insert("atoms", label_block(&["C"]));
        let labels = TypeLabels::from_frame(&frame).unwrap();
        assert!(labels.block("bonds").is_none());
        assert!(labels.block("impropers").is_none());
    }

    #[test]
    fn reversed_bond_labels_collapse_including_qualified() {
        let mut frame = Frame::new();
        frame.insert("bonds", label_block(&["C_3-O_R@1.5", "O_R-C_3@1.5"]));
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let bonds = labels.block("bonds").expect("bonds resolved");
        assert_eq!(bonds.type_ids(), &[1, 1][..]);
        assert_eq!(
            bonds.labels().map(|l| l.to_vec()),
            Some(vec!["C_3-O_R@1.5".to_string()])
        );
        assert_eq!(bonds.n_types(), 1);
    }

    /// UFF-style qualified labels in both orientations, bonds and angles in one
    /// frame: each block collapses to one id, keyed by the canonical label (the
    /// angle's reversal swaps its two bond-order fields).
    #[test]
    fn reversed_uff_qualified_bond_and_angle_labels_give_one_id_each() {
        let mut frame = Frame::new();
        frame.insert("bonds", label_block(&["C_3-O_R@1.5", "O_R-C_3@1.5"]));
        frame.insert(
            "angles",
            label_block(&["C_3-C_R-O_2@1_1.5_2", "O_2-C_R-C_3@1.5_1_2"]),
        );

        let labels = TypeLabels::from_frame(&frame).unwrap();

        let bonds = labels.block("bonds").expect("bonds resolved");
        assert_eq!(bonds.type_ids(), &[1, 1][..]);
        assert_eq!(bonds.n_types(), 1);
        assert_eq!(
            bonds.labels().map(|l| l.to_vec()),
            Some(vec!["C_3-O_R@1.5".to_string()])
        );
        let angles = labels.block("angles").expect("angles resolved");
        assert_eq!(angles.type_ids(), &[1, 1][..]);
        assert_eq!(angles.n_types(), 1);
        assert_eq!(
            angles.labels().map(|l| l.to_vec()),
            Some(vec!["C_3-C_R-O_2@1_1.5_2".to_string()])
        );
    }

    #[test]
    fn reversed_angle_labels_collapse() {
        let mut frame = Frame::new();
        frame.insert("angles", label_block(&["h1-c3-c3", "c3-c3-h1"]));
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let angles = labels.block("angles").expect("angles resolved");
        assert_eq!(angles.type_ids(), &[1, 1][..]);
        assert_eq!(
            angles.labels().map(|l| l.to_vec()),
            Some(vec!["c3-c3-h1".to_string()])
        );
        assert_eq!(angles.n_types(), 1);
    }

    #[test]
    fn reversed_dihedral_labels_collapse() {
        let mut frame = Frame::new();
        frame.insert("dihedrals", label_block(&["oh-c3-c3-hc", "hc-c3-c3-oh"]));
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let dihedrals = labels.block("dihedrals").expect("dihedrals resolved");
        assert_eq!(dihedrals.type_ids(), &[1, 1][..]);
        assert_eq!(
            dihedrals.labels().map(|l| l.to_vec()),
            Some(vec!["hc-c3-c3-oh".to_string()])
        );
        assert_eq!(dihedrals.n_types(), 1);
    }

    #[test]
    fn reversed_improper_labels_keep_separate_ids() {
        let mut frame = Frame::new();
        frame.insert(
            "impropers",
            label_block(&["C_R-P_3+3-C_R-C_R", "C_R-C_R-P_3+3-C_R"]),
        );
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let impropers = labels.block("impropers").expect("impropers resolved");
        assert_eq!(
            impropers.labels().map(|l| l.to_vec()),
            Some(vec![
                "C_R-C_R-P_3+3-C_R".to_string(),
                "C_R-P_3+3-C_R-C_R".to_string(),
            ])
        );
        assert_eq!(impropers.type_ids(), &[2, 1][..]);
        assert_eq!(impropers.n_types(), 2);
    }

    #[test]
    fn type_id_only_block_is_taken_as_is() {
        let mut frame = Frame::new();
        frame.insert("atoms", type_id_block(&[2, 1, 2]));
        let labels = TypeLabels::from_frame(&frame).unwrap();
        let atoms = labels.block("atoms").expect("atoms resolved");
        assert_eq!(atoms.type_ids(), &[2, 1, 2][..]);
        assert!(atoms.labels().is_none());
        assert_eq!(atoms.n_types(), 2);
    }

    #[test]
    fn empty_label_is_rejected() {
        let mut frame = Frame::new();
        frame.insert("atoms", label_block(&["C", ""]));
        assert!(TypeLabels::from_frame(&frame).is_err());
    }

    #[test]
    fn rows_without_type_or_type_id_are_rejected() {
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert(
                keys::X,
                ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0_f64, 1.0]).unwrap(),
            )
            .unwrap();
        frame.insert("atoms", atoms);
        assert!(TypeLabels::from_frame(&frame).is_err());
    }

    // ------------------------------------------------------------------
    // Inventory meta parser: malformed input is an error naming the key
    // ------------------------------------------------------------------

    fn inventory_error(raw: &str) -> String {
        let mut frame = Frame::new();
        frame.insert("atoms", label_block(&["C"]));
        frame.meta.insert(keys::ATOM_TYPE_LABELS, raw);
        let Err(err) = TypeLabels::from_frame(&frame) else {
            panic!("malformed inventory {raw:?} must be an error");
        };
        err
    }

    #[test]
    fn inventory_pair_without_colon_is_rejected() {
        let err = inventory_error("1C");
        assert!(err.contains(keys::ATOM_TYPE_LABELS), "{err}");
        assert!(err.contains("1C"), "{err}");
    }

    #[test]
    fn inventory_non_integer_id_is_rejected() {
        let err = inventory_error("x:C");
        assert!(err.contains(keys::ATOM_TYPE_LABELS), "{err}");
        assert!(err.contains('x'), "{err}");
    }

    #[test]
    fn inventory_empty_label_is_rejected() {
        let err = inventory_error("1:");
        assert!(err.contains(keys::ATOM_TYPE_LABELS), "{err}");
    }

    #[test]
    fn inventory_repeated_id_is_rejected() {
        let err = inventory_error("1:C,1:H");
        assert!(err.contains(keys::ATOM_TYPE_LABELS), "{err}");
    }
}
