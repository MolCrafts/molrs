//! Integration tests for the C API, calling only `extern "C"` functions.
//!
//! Tests run serially to avoid global store contention.

use std::ffi::CString;
use std::sync::Mutex;

use molrs_capi::block::*;
use molrs_capi::forcefield::*;
use molrs_capi::frame::*;
use molrs_capi::simbox::*;
use molrs_capi::*;

/// Re-export the float type so tests compile under both f32 and f64 features.
type Float = molrs_capi::F;

/// Global lock to serialize tests (they share a global store).
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn assert_ok(status: MolrsStatus) {
    assert_eq!(
        status,
        MolrsStatus::Ok,
        "expected Ok, got {:?}: {:?}",
        status,
        unsafe { std::ffi::CStr::from_ptr(molrs_last_error()) }
    );
}

/// Helper: intern a key and return its id.
fn intern(name: &str) -> u32 {
    let cstr = CString::new(name).unwrap();
    let mut id: u32 = 0;
    assert_ok(unsafe { molrs_intern_key(cstr.as_ptr(), &mut id) });
    id
}

#[test]
fn test_init_shutdown() {
    let _g = TEST_LOCK.lock().unwrap();
    unsafe { molrs_init() };
    unsafe { molrs_shutdown() };
}

#[test]
fn test_intern_key_roundtrip() {
    let _g = TEST_LOCK.lock().unwrap();
    let key = CString::new("test_key_roundtrip").unwrap();
    let mut key_id: u32 = 0;
    assert_ok(unsafe { molrs_intern_key(key.as_ptr(), &mut key_id) });

    // Same key returns same id
    let mut key_id2: u32 = 0;
    assert_ok(unsafe { molrs_intern_key(key.as_ptr(), &mut key_id2) });
    assert_eq!(key_id, key_id2);

    // Can look up the name
    let name_ptr = unsafe { molrs_key_name(key_id) };
    assert!(!name_ptr.is_null());
    let name = unsafe { std::ffi::CStr::from_ptr(name_ptr) };
    assert_eq!(name.to_str().unwrap(), "test_key_roundtrip");
}

#[test]
fn test_frame_lifecycle() {
    let _g = TEST_LOCK.lock().unwrap();

    let mut frame = MolrsFrameHandle { idx: 0, version: 0 };
    assert_ok(unsafe { molrs_frame_new(&mut frame) });

    let mut frame2 = MolrsFrameHandle { idx: 0, version: 0 };
    assert_ok(unsafe { molrs_frame_clone(frame, &mut frame2) });

    assert_ok(unsafe { molrs_frame_drop(frame) });
    assert_ok(unsafe { molrs_frame_drop(frame2) });

    // Dropping again should fail
    let status = unsafe { molrs_frame_drop(frame) };
    assert_ne!(status, MolrsStatus::Ok);
}

#[test]
fn test_block_insert_and_read() {
    let _g = TEST_LOCK.lock().unwrap();

    let mut frame = MolrsFrameHandle { idx: 0, version: 0 };
    assert_ok(unsafe { molrs_frame_new(&mut frame) });

    let atoms_id = intern("atoms_bir");
    let pos_id = intern("positions_bir");

    assert_ok(unsafe { molrs_frame_set_block(frame, atoms_id, 0) });

    let mut block = MolrsBlockHandle {
        frame,
        key_id: 0,
        block_version: 0,
    };
    assert_ok(unsafe { molrs_frame_get_block(frame, atoms_id, &mut block) });

    // Insert float column: 3 atoms x 3 coords = shape [3, 3]
    let data: [Float; 9] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let shape: [usize; 2] = [3, 3];
    assert_ok(unsafe { molrs_block_set_F(&mut block, pos_id, data.as_ptr(), shape.as_ptr(), 2) });

    // Check nrows
    let mut nrows: usize = 0;
    assert_ok(unsafe { molrs_block_nrows(block, &mut nrows) });
    assert_eq!(nrows, 3);

    // Check ncols
    let mut ncols: usize = 0;
    assert_ok(unsafe { molrs_block_ncols(block, &mut ncols) });
    assert_eq!(ncols, 1);

    // Check dtype
    let mut dtype = MolrsDType::Int;
    assert_ok(unsafe { molrs_block_col_dtype(block, pos_id, &mut dtype) });
    assert_eq!(dtype, MolrsDType::Float);

    // Check shape
    let mut col_shape: [usize; 4] = [0; 4];
    let mut ndim: usize = 4;
    assert_ok(unsafe { molrs_block_col_shape(block, pos_id, col_shape.as_mut_ptr(), &mut ndim) });
    assert_eq!(ndim, 2);
    assert_eq!(col_shape[0], 3);
    assert_eq!(col_shape[1], 3);

    // Zero-copy read
    let mut ptr: *const Float = std::ptr::null();
    let mut len: usize = 0;
    assert_ok(unsafe { molrs_block_get_F(block, pos_id, &mut ptr, &mut len) });
    assert_eq!(len, 9);
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    assert_eq!(slice, &data);

    // Copy read
    let mut buf = [0.0 as Float; 9];
    assert_ok(unsafe { molrs_block_copy_F(block, pos_id, buf.as_mut_ptr(), 9) });
    assert_eq!(buf, data);

    unsafe { molrs_frame_drop(frame) };
}

#[test]
fn test_block_ptr_mut() {
    let _g = TEST_LOCK.lock().unwrap();

    let mut frame = MolrsFrameHandle { idx: 0, version: 0 };
    assert_ok(unsafe { molrs_frame_new(&mut frame) });

    let atoms_id = intern("atoms_pm");
    let x_id = intern("x_pm");

    assert_ok(unsafe { molrs_frame_set_block(frame, atoms_id, 0) });

    let mut block = MolrsBlockHandle {
        frame,
        key_id: 0,
        block_version: 0,
    };
    assert_ok(unsafe { molrs_frame_get_block(frame, atoms_id, &mut block) });

    let data: [Float; 3] = [1.0, 2.0, 3.0];
    let shape: [usize; 1] = [3];
    assert_ok(unsafe { molrs_block_set_F(&mut block, x_id, data.as_ptr(), shape.as_ptr(), 1) });

    // Get mutable pointer and modify
    let mut ptr: *mut Float = std::ptr::null_mut();
    let mut len: usize = 0;
    assert_ok(unsafe { molrs_block_get_F_mut(&mut block, x_id, &mut ptr, &mut len) });
    assert_eq!(len, 3);
    unsafe {
        *ptr = 10.0;
        *ptr.add(1) = 20.0;
        *ptr.add(2) = 30.0;
    }
    assert_ok(unsafe { molrs_block_col_commit(&mut block) });

    // Verify changes via copy
    let mut buf = [0.0 as Float; 3];
    assert_ok(unsafe { molrs_block_copy_F(block, x_id, buf.as_mut_ptr(), 3) });
    assert_eq!(buf, [10.0, 20.0, 30.0]);

    unsafe { molrs_frame_drop(frame) };
}

#[test]
fn test_simbox_lifecycle() {
    let _g = TEST_LOCK.lock().unwrap();

    let origin: [Float; 3] = [0.0, 0.0, 0.0];
    let pbc: [bool; 3] = [true, true, true];
    let mut sb = MolrsSimBoxHandle { idx: 0, version: 0 };
    assert_ok(unsafe { molrs_simbox_cube(10.0 as Float, origin.as_ptr(), pbc.as_ptr(), &mut sb) });

    let mut vol: Float = 0.0;
    assert_ok(unsafe { molrs_simbox_volume(sb, &mut vol) });
    assert!((vol - 1000.0).abs() < 1e-3);

    let mut lengths: [Float; 3] = [0.0; 3];
    assert_ok(unsafe { molrs_simbox_lengths(sb, lengths.as_mut_ptr()) });
    assert!((lengths[0] - 10.0).abs() < 1e-6);

    let mut h: [Float; 9] = [0.0; 9];
    assert_ok(unsafe { molrs_simbox_h(sb, h.as_mut_ptr()) });
    assert!((h[0] - 10.0).abs() < 1e-6);
    assert!((h[4] - 10.0).abs() < 1e-6);
    assert!((h[8] - 10.0).abs() < 1e-6);

    // Wrap coordinates
    let xyz_in: [Float; 6] = [11.0, -1.0, 21.0, 0.5, 0.5, 0.5];
    let mut xyz_out: [Float; 6] = [0.0; 6];
    assert_ok(unsafe { molrs_simbox_wrap(sb, xyz_in.as_ptr(), xyz_out.as_mut_ptr(), 2) });
    assert!((xyz_out[0] - 1.0).abs() < 1e-4);
    assert!((xyz_out[1] - 9.0).abs() < 1e-4);
    assert!((xyz_out[2] - 1.0).abs() < 1e-4);

    assert_ok(unsafe { molrs_simbox_drop(sb) });
    assert_ne!(unsafe { molrs_simbox_drop(sb) }, MolrsStatus::Ok);
}

#[test]
fn test_simbox_shortest_vector() {
    let _g = TEST_LOCK.lock().unwrap();

    let origin: [Float; 3] = [0.0, 0.0, 0.0];
    let pbc: [bool; 3] = [true, true, true];
    let mut sb = MolrsSimBoxHandle { idx: 0, version: 0 };
    assert_ok(unsafe { molrs_simbox_cube(10.0 as Float, origin.as_ptr(), pbc.as_ptr(), &mut sb) });

    let r1: [Float; 3] = [0.5, 0.0, 0.0];
    let r2: [Float; 3] = [9.5, 0.0, 0.0];
    let mut dr: [Float; 3] = [0.0; 3];
    assert_ok(unsafe {
        molrs_simbox_shortest_vector(sb, r1.as_ptr(), r2.as_ptr(), dr.as_mut_ptr(), 1)
    });
    assert!((dr[0] - (-1.0)).abs() < 1e-4);

    unsafe { molrs_simbox_drop(sb) };
}

#[test]
fn test_frame_simbox_association() {
    let _g = TEST_LOCK.lock().unwrap();

    let mut frame = MolrsFrameHandle { idx: 0, version: 0 };
    assert_ok(unsafe { molrs_frame_new(&mut frame) });

    let origin: [Float; 3] = [0.0, 0.0, 0.0];
    let pbc: [bool; 3] = [true, true, true];
    let mut sb = MolrsSimBoxHandle { idx: 0, version: 0 };
    assert_ok(unsafe { molrs_simbox_cube(5.0 as Float, origin.as_ptr(), pbc.as_ptr(), &mut sb) });

    assert_ok(unsafe { molrs_frame_set_simbox(frame, sb) });

    let mut sb2 = MolrsSimBoxHandle { idx: 0, version: 0 };
    assert_ok(unsafe { molrs_frame_get_simbox(frame, &mut sb2) });

    let mut vol: Float = 0.0;
    assert_ok(unsafe { molrs_simbox_volume(sb2, &mut vol) });
    assert!((vol - 125.0).abs() < 1e-3);

    assert_ok(unsafe { molrs_frame_clear_simbox(frame) });

    let mut sb3 = MolrsSimBoxHandle { idx: 0, version: 0 };
    assert_ne!(
        unsafe { molrs_frame_get_simbox(frame, &mut sb3) },
        MolrsStatus::Ok
    );

    unsafe {
        molrs_simbox_drop(sb);
        molrs_simbox_drop(sb2);
        molrs_frame_drop(frame);
    }
}

#[test]
fn test_frame_metadata() {
    let _g = TEST_LOCK.lock().unwrap();

    let mut frame = MolrsFrameHandle { idx: 0, version: 0 };
    assert_ok(unsafe { molrs_frame_new(&mut frame) });

    let key = CString::new("source").unwrap();
    let val = CString::new("test.pdb").unwrap();
    assert_ok(unsafe { molrs_frame_set_meta(frame, key.as_ptr(), val.as_ptr()) });

    let mut out: *mut std::ffi::c_char = std::ptr::null_mut();
    assert_ok(unsafe { molrs_frame_get_meta(frame, key.as_ptr(), &mut out) });
    assert!(!out.is_null());
    let result = unsafe { std::ffi::CStr::from_ptr(out) };
    assert_eq!(result.to_str().unwrap(), "test.pdb");
    unsafe { molrs_free_string(out) };

    unsafe { molrs_frame_drop(frame) };
}

#[test]
fn test_forcefield_lifecycle() {
    let _g = TEST_LOCK.lock().unwrap();

    let name = CString::new("test_ff").unwrap();
    let mut ff = MolrsForceFieldHandle { idx: 0, version: 0 };
    assert_ok(unsafe { molrs_ff_new(name.as_ptr(), &mut ff) });

    let harmonic = CString::new("harmonic").unwrap();
    assert_ok(unsafe { molrs_ff_def_bondstyle(ff, harmonic.as_ptr()) });
    assert_ok(unsafe { molrs_ff_def_anglestyle(ff, harmonic.as_ptr()) });

    let full = CString::new("full").unwrap();
    assert_ok(unsafe { molrs_ff_def_atomstyle(ff, full.as_ptr()) });

    let mut count: usize = 0;
    assert_ok(unsafe { molrs_ff_style_count(ff, &mut count) });
    assert_eq!(count, 3);

    // Define a bond type
    let bond_cat = CString::new("bond").unwrap();
    let type_name = CString::new("CT-OH").unwrap();
    let pk0 = CString::new("k0").unwrap();
    let pr0 = CString::new("r0").unwrap();
    let param_keys: [*const std::ffi::c_char; 2] = [pk0.as_ptr(), pr0.as_ptr()];
    let param_values: [f64; 2] = [300.0, 1.4];
    assert_ok(unsafe {
        molrs_ff_def_type(
            ff,
            bond_cat.as_ptr(),
            harmonic.as_ptr(),
            type_name.as_ptr(),
            param_keys.as_ptr(),
            param_values.as_ptr(),
            2,
        )
    });

    let mut out_cat: *mut std::ffi::c_char = std::ptr::null_mut();
    let mut out_name: *mut std::ffi::c_char = std::ptr::null_mut();
    assert_ok(unsafe { molrs_ff_get_style_name(ff, 0, &mut out_cat, &mut out_name) });
    assert!(!out_cat.is_null());
    assert!(!out_name.is_null());
    unsafe {
        molrs_free_string(out_cat);
        molrs_free_string(out_name);
    }

    assert_ok(unsafe { molrs_ff_drop(ff) });
    assert_ne!(unsafe { molrs_ff_drop(ff) }, MolrsStatus::Ok);
}

#[test]
fn test_forcefield_json_roundtrip() {
    let _g = TEST_LOCK.lock().unwrap();

    let name = CString::new("lj_system").unwrap();
    let mut ff = MolrsForceFieldHandle { idx: 0, version: 0 };
    assert_ok(unsafe { molrs_ff_new(name.as_ptr(), &mut ff) });

    let lj = CString::new("lj/cut").unwrap();
    let cutoff_key = CString::new("cutoff").unwrap();
    let param_keys: [*const std::ffi::c_char; 1] = [cutoff_key.as_ptr()];
    let param_values: [f64; 1] = [10.0];
    assert_ok(unsafe {
        molrs_ff_def_pairstyle(
            ff,
            lj.as_ptr(),
            param_keys.as_ptr(),
            param_values.as_ptr(),
            1,
        )
    });

    let pair_cat = CString::new("pair").unwrap();
    let type_name = CString::new("Ar").unwrap();
    let eps_key = CString::new("epsilon").unwrap();
    let sig_key = CString::new("sigma").unwrap();
    let type_keys: [*const std::ffi::c_char; 2] = [eps_key.as_ptr(), sig_key.as_ptr()];
    let type_values: [f64; 2] = [1.0, 3.4];
    assert_ok(unsafe {
        molrs_ff_def_type(
            ff,
            pair_cat.as_ptr(),
            lj.as_ptr(),
            type_name.as_ptr(),
            type_keys.as_ptr(),
            type_values.as_ptr(),
            2,
        )
    });

    let mut json_ptr: *mut std::ffi::c_char = std::ptr::null_mut();
    let mut json_len: usize = 0;
    assert_ok(unsafe { molrs_ff_to_json(ff, &mut json_ptr, &mut json_len) });
    assert!(!json_ptr.is_null());
    assert!(json_len > 0);

    let mut ff2 = MolrsForceFieldHandle { idx: 0, version: 0 };
    assert_ok(unsafe { molrs_ff_from_json(json_ptr, &mut ff2) });

    let mut count: usize = 0;
    assert_ok(unsafe { molrs_ff_style_count(ff2, &mut count) });
    assert_eq!(count, 1);

    unsafe {
        molrs_free_string(json_ptr);
        molrs_ff_drop(ff);
        molrs_ff_drop(ff2);
    }
}

#[test]
fn test_full_simulation_loop() {
    let _g = TEST_LOCK.lock().unwrap();

    let atoms_id = intern("sim_atoms");
    let pos_id = intern("sim_positions");

    let mut frame = MolrsFrameHandle { idx: 0, version: 0 };
    assert_ok(unsafe { molrs_frame_new(&mut frame) });

    // SimBox
    let origin: [Float; 3] = [0.0, 0.0, 0.0];
    let pbc: [bool; 3] = [true, true, true];
    let mut sb = MolrsSimBoxHandle { idx: 0, version: 0 };
    assert_ok(unsafe { molrs_simbox_cube(10.0 as Float, origin.as_ptr(), pbc.as_ptr(), &mut sb) });
    assert_ok(unsafe { molrs_frame_set_simbox(frame, sb) });

    // Block + positions
    assert_ok(unsafe { molrs_frame_set_block(frame, atoms_id, 0) });
    let mut block = MolrsBlockHandle {
        frame,
        key_id: 0,
        block_version: 0,
    };
    assert_ok(unsafe { molrs_frame_get_block(frame, atoms_id, &mut block) });

    let positions: [Float; 9] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let shape: [usize; 2] = [3, 3];
    assert_ok(unsafe {
        molrs_block_set_F(&mut block, pos_id, positions.as_ptr(), shape.as_ptr(), 2)
    });

    // READ: zero-copy
    let mut ptr: *const Float = std::ptr::null();
    let mut len: usize = 0;
    assert_ok(unsafe { molrs_block_get_F(block, pos_id, &mut ptr, &mut len) });
    assert_eq!(len, 9);
    let read_data = unsafe { std::slice::from_raw_parts(ptr, len) };
    assert_eq!(read_data[0], 1.0);

    // WRITE: mutable pointer
    let mut mut_ptr: *mut Float = std::ptr::null_mut();
    let mut mut_len: usize = 0;
    assert_ok(unsafe { molrs_block_get_F_mut(&mut block, pos_id, &mut mut_ptr, &mut mut_len) });

    let write_slice = unsafe { std::slice::from_raw_parts_mut(mut_ptr, mut_len) };
    for v in write_slice.iter_mut() {
        *v *= 2.0;
    }
    assert_ok(unsafe { molrs_block_col_commit(&mut block) });

    // Verify
    let mut buf = [0.0 as Float; 9];
    assert_ok(unsafe { molrs_block_copy_F(block, pos_id, buf.as_mut_ptr(), 9) });
    assert_eq!(buf, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]);

    // Query SimBox from frame
    let mut frame_sb = MolrsSimBoxHandle { idx: 0, version: 0 };
    assert_ok(unsafe { molrs_frame_get_simbox(frame, &mut frame_sb) });
    let mut h: [Float; 9] = [0.0; 9];
    assert_ok(unsafe { molrs_simbox_h(frame_sb, h.as_mut_ptr()) });
    assert!((h[0] - 10.0).abs() < 1e-6);

    unsafe {
        molrs_simbox_drop(sb);
        molrs_simbox_drop(frame_sb);
        molrs_frame_drop(frame);
    }
}
