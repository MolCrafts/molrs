# Regression — special-bonds-01-molrs

Public API: `Topology::from_frame` → `exclusions(&BondDistanceWeights::from_exclusion_depth(3))`.

Hard-coded C12 goldens (hand-counted hop distance ≤ 3, Cassandra tail `[0,0,0,1]`):

```text
exclusions[0]  == [0, 1, 2, 3]
exclusions[5]  == [2, 3, 4, 5, 6, 7, 8]
exclusions[11] == [8, 9, 10, 11]
```

Runnable gate:

```text
cargo test -p molcrafts-molrs --lib --features full,filesystem topology::tests::
```

No third-party tool at run time. Goldens are graph distances, not an external packer.
