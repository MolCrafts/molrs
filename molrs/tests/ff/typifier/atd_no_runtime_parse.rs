//! ac-003 — no runtime table parsing on the typify path.
//!
//! The engine used to re-parse its `ATOMTYPE_*.DEF` text inside `typify()`, i.e.
//! once per molecule. The tables are now generated `&'static` Rust data
//! (`ff::params::ATOMTYPE_*`), so there is nothing left to parse — and this file
//! keeps it that way from two directions, because either gate alone is weak:
//!
//! 1. **Source gate** — nothing under `molrs/src/ff/typifier/` parses a table or
//!    reads a file at runtime. Structural, and the one the acceptance contract
//!    words directly (`parse_str` -> 0 hits).
//! 2. **Cost gate** — typifying 500 molecules must cost the same regardless of
//!    how big the table is. A re-parse allocates at least once per rule, so a
//!    table-size-dependent cost is precisely the signature this test denies. A
//!    source gate alone cannot see a parser reached through a helper; a cost gate
//!    alone cannot see a parse that happens to allocate little.
//!
//! The cost gate counts allocations rather than wall-clock time: allocation
//! counts are deterministic for a given build, timings are not, and a timing
//! assertion on a shared CI box is a flake generator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use molrs::Atomistic;
use molrs::ff::params::{ATOMTYPE_ABCG2, ATOMTYPE_GAS};
use molrs::ff::typifier::Typifier;
use molrs::ff::typifier::atd::{AtdParameterSet, AtdTypifier};

use super::antechamber_oracle::CASES;
use super::oracle_mol::build_case;
use super::source_gate::lines_containing;

/// Molecules typified in the cost gate. "Typifying 500 molecules does not
/// re-parse anything" is the acceptance criterion's own wording.
const WORKLOAD: usize = 500;

// ── allocation counter ──────────────────────────────────────────────────────
//
// Thread-local, so tests running concurrently in this binary cannot pollute each
// other's count, and so the `System` passthrough stays the only cost for every
// other test in the target. `Cell<u64>` needs no destructor, hence registering
// the TLS key allocates nothing and the counter cannot recurse into itself.

thread_local! {
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

struct CountingAllocator;

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        // SAFETY: `layout` is forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        // SAFETY: `layout` is forwarded unchanged to the system allocator.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        // SAFETY: `ptr`/`layout`/`new_size` are forwarded unchanged; the caller
        // upholds `GlobalAlloc::realloc`'s contract.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr`/`layout` are forwarded unchanged; the caller upholds
        // `GlobalAlloc::dealloc`'s contract.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Allocations made by `f` on the calling thread.
fn allocations_of(f: impl FnOnce()) -> u64 {
    let before = ALLOCATIONS.with(Cell::get);
    f();
    ALLOCATIONS.with(Cell::get) - before
}

// ── 1. source gate ──────────────────────────────────────────────────────────

/// The typify path parses no table text at runtime.
#[test]
fn the_typifier_tree_never_parses_a_table() {
    let hits = lines_containing("parse_str");
    assert!(
        hits.is_empty(),
        "the typify path still parses a table at runtime; the ATD rules are \
         generated `&'static` data in `ff::params`, so nothing here should parse:\n{}",
        hits.join("\n")
    );
}

/// ...nor reads one off disk.
///
/// Separate from `parse_str` because it is a separate way to reintroduce the same
/// bug: a table loaded per `typify()` is a runtime table load whether it is parsed
/// with `parse_str` or read with `read_to_string`.
#[test]
fn the_typifier_tree_never_reads_a_file_at_runtime() {
    let hits: Vec<String> = ["read_to_string", "File::open", "fs::read"]
        .iter()
        .flat_map(|needle| lines_containing(needle))
        .collect();
    assert!(
        hits.is_empty(),
        "the typify path reads a file at runtime; parameter tables are compiled \
         in, not loaded:\n{}",
        hits.join("\n")
    );
}

// ── 2. cost gate ────────────────────────────────────────────────────────────

/// Typifying is as cheap with a 200-rule table as with a 46-rule one.
///
/// If `typify()` re-parsed its table, each call would allocate at least once per
/// rule, and ABCG2 (200 rules) would cost ~154 allocations per molecule more than
/// GAS (46). With `&'static` tables the two differ only by the handful of
/// allocations the *answers* cost, so the spread sits near zero.
#[test]
fn typify_cost_is_independent_of_table_size() {
    let molecules: Vec<Atomistic> = CASES
        .iter()
        .cycle()
        .take(WORKLOAD)
        .map(|case| build_case(case).0)
        .collect();

    let gas = AtdTypifier::new(AtdParameterSet::Gas);
    let abcg2 = AtdTypifier::new(AtdParameterSet::Abcg2);

    // Warm up outside the measurement: a lazily-initialized static would
    // otherwise be charged to whichever parameter set happened to run first.
    gas.typify(&molecules[0]).expect("GAS typify");
    abcg2.typify(&molecules[0]).expect("ABCG2 typify");

    let gas_allocations = allocations_of(|| {
        for mol in &molecules {
            gas.typify(mol).expect("GAS typify");
        }
    });
    let abcg2_allocations = allocations_of(|| {
        for mol in &molecules {
            abcg2.typify(mol).expect("ABCG2 typify");
        }
    });

    // Sanity: the counter is wired up. Without this a broken allocator hook
    // would make the assertion below pass by measuring nothing.
    assert!(
        gas_allocations > 0 && abcg2_allocations > 0,
        "the allocation counter recorded nothing ({gas_allocations} / \
         {abcg2_allocations}); the gate below would be vacuous"
    );

    let per_molecule = |total: u64| total as f64 / WORKLOAD as f64;
    let spread = (per_molecule(abcg2_allocations) - per_molecule(gas_allocations)).abs();
    let rule_gap = (ATOMTYPE_ABCG2.rules.len() - ATOMTYPE_GAS.rules.len()) as f64;

    assert!(
        spread < rule_gap,
        "typify cost tracks table size: {:.1} allocations/molecule with ABCG2 \
         ({} rules) vs {:.1} with GAS ({} rules) — a spread of {spread:.1}, at or \
         above the {rule_gap:.0}-rule difference between the tables. That is what \
         re-parsing the table on every typify() looks like.",
        per_molecule(abcg2_allocations),
        ATOMTYPE_ABCG2.rules.len(),
        per_molecule(gas_allocations),
        ATOMTYPE_GAS.rules.len(),
    );
}
