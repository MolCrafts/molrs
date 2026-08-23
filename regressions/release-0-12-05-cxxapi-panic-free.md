# cxxapi Result surface

Converted to `Result` (no panic on fallible paths):
- write_frame_xyz / write_frame_zarr / read_frame_zarr_first / xyz_read_first_frame
- frame_set_column_* / frame_set_box
- symbol_for_z / z_for_symbol / frame_with_elements / xyz_frame
- am1_bcc charge insert

Production body before `#[cfg(test)]` must not contain unwrap/expect/panic
except allowlisted dead helpers (none remaining).

Verified: `cargo test --manifest-path molrs-cxxapi/Cargo.toml`
