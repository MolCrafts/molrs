# Dielectric fs + two-pass + SimBox f64

- Analysis `dt` docs: **fs**
- Conductivity SI uses `FEMTOSECOND_S` (1e-15)
- `static_dielectric_constant_components` two-pass variance
- SimBox Zarr write/read f64 (legacy f32 promote on read)

Verified by:
- `cargo test … dielectric`
- `simbox_geometry_roundtrips_as_f64` in zarr record_io tests
