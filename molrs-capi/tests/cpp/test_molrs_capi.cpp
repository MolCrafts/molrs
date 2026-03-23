// GoogleTest suite for molrs-capi C API.
//
// Build:
//   cargo build -p molcrafts-molrs-capi          # build Rust library first
//   cmake -B build tests/cpp && cmake --build build && ctest --test-dir build

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>

extern "C" {
#include "molrs.h"
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#define ASSERT_MOLRS_OK(expr)                                                  \
    do {                                                                        \
        MolrsStatus s_ = (expr);                                                \
        ASSERT_EQ(s_, MOLRS_STATUS_OK)                                          \
            << "molrs error: " << molrs_last_error();                           \
    } while (0)

#define EXPECT_MOLRS_OK(expr)                                                  \
    do {                                                                        \
        MolrsStatus s_ = (expr);                                                \
        EXPECT_EQ(s_, MOLRS_STATUS_OK)                                          \
            << "molrs error: " << molrs_last_error();                           \
    } while (0)

static uint32_t intern(const char* key) {
    uint32_t id = 0;
    MolrsStatus s = molrs_intern_key(key, &id);
    if (s != MOLRS_STATUS_OK) {
        ADD_FAILURE() << "intern(\"" << key << "\") failed: " << molrs_last_error();
    }
    return id;
}

// ---------------------------------------------------------------------------
// Fixture: manages init/shutdown per test suite
// ---------------------------------------------------------------------------

class MolrsTest : public ::testing::Test {
protected:
    void SetUp() override { molrs_init(); }
};

// ===== Lifecycle & Utilities ==============================================

TEST_F(MolrsTest, InitShutdown) {
    // init already called in SetUp
    molrs_shutdown();
    // re-init so TearDown doesn't break
    molrs_init();
}

TEST_F(MolrsTest, InternKeyRoundtrip) {
    uint32_t id1 = intern("gtest_key_a");
    uint32_t id2 = intern("gtest_key_a");
    EXPECT_EQ(id1, id2) << "same string must yield same id";

    uint32_t id3 = intern("gtest_key_b");
    EXPECT_NE(id1, id3) << "different strings must yield different ids";

    const char* name = molrs_key_name(id1);
    ASSERT_NE(name, nullptr);
    EXPECT_STREQ(name, "gtest_key_a");
}

TEST_F(MolrsTest, LastErrorAfterBadDrop) {
    MolrsFrameHandle bogus{999, 999};
    MolrsStatus s = molrs_frame_drop(bogus);
    EXPECT_NE(s, MOLRS_STATUS_OK);
    const char* msg = molrs_last_error();
    ASSERT_NE(msg, nullptr);
    EXPECT_GT(std::strlen(msg), 0u);
}

// ===== Frame ==============================================================

TEST_F(MolrsTest, FrameLifecycle) {
    MolrsFrameHandle frame{};
    ASSERT_MOLRS_OK(molrs_frame_new(&frame));

    MolrsFrameHandle clone{};
    ASSERT_MOLRS_OK(molrs_frame_clone(frame, &clone));

    ASSERT_MOLRS_OK(molrs_frame_drop(frame));
    ASSERT_MOLRS_OK(molrs_frame_drop(clone));

    // double drop must fail
    EXPECT_NE(molrs_frame_drop(frame), MOLRS_STATUS_OK);
}

TEST_F(MolrsTest, FrameMetadata) {
    MolrsFrameHandle frame{};
    ASSERT_MOLRS_OK(molrs_frame_new(&frame));

    ASSERT_MOLRS_OK(molrs_frame_set_meta(frame, "author", "gtest"));

    char* val = nullptr;
    ASSERT_MOLRS_OK(molrs_frame_get_meta(frame, "author", &val));
    ASSERT_NE(val, nullptr);
    EXPECT_STREQ(val, "gtest");
    molrs_free_string(val);

    // missing key
    char* missing = nullptr;
    EXPECT_NE(molrs_frame_get_meta(frame, "nope", &missing), MOLRS_STATUS_OK);

    ASSERT_MOLRS_OK(molrs_frame_drop(frame));
}

// ===== Block Insert & Read ================================================

TEST_F(MolrsTest, BlockInsertAndRead) {
    MolrsFrameHandle frame{};
    ASSERT_MOLRS_OK(molrs_frame_new(&frame));

    uint32_t atoms_id = intern("gt_atoms");
    uint32_t pos_id   = intern("gt_positions");

    ASSERT_MOLRS_OK(molrs_frame_set_block(frame, atoms_id, 0));

    MolrsBlockHandle block{};
    ASSERT_MOLRS_OK(molrs_frame_get_block(frame, atoms_id, &block));

    // Insert 3x3 F column (F = float by default)
    F data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    size_t shape[2] = {3, 3};
    ASSERT_MOLRS_OK(molrs_block_set_F(&block, pos_id, data, shape, 2));

    // nrows
    size_t nrows = 0;
    ASSERT_MOLRS_OK(molrs_block_nrows(block, &nrows));
    EXPECT_EQ(nrows, 3u);

    // ncols
    size_t ncols = 0;
    ASSERT_MOLRS_OK(molrs_block_ncols(block, &ncols));
    EXPECT_EQ(ncols, 1u);

    // dtype
    MolrsDType dtype{};
    ASSERT_MOLRS_OK(molrs_block_col_dtype(block, pos_id, &dtype));
    EXPECT_EQ(dtype, MOLRS_D_TYPE_FLOAT);

    // shape query
    size_t col_shape[4] = {};
    size_t ndim = 4;
    ASSERT_MOLRS_OK(molrs_block_col_shape(block, pos_id, col_shape, &ndim));
    EXPECT_EQ(ndim, 2u);
    EXPECT_EQ(col_shape[0], 3u);
    EXPECT_EQ(col_shape[1], 3u);

    // zero-copy read
    const F* ptr = nullptr;
    size_t len = 0;
    ASSERT_MOLRS_OK(molrs_block_get_F(block, pos_id, &ptr, &len));
    ASSERT_EQ(len, 9u);
    for (size_t i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(ptr[i], data[i]);
    }

    // copy read
    F buf[9] = {};
    ASSERT_MOLRS_OK(molrs_block_copy_F(block, pos_id, buf, 9));
    for (size_t i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(buf[i], data[i]);
    }

    ASSERT_MOLRS_OK(molrs_frame_drop(frame));
}

TEST_F(MolrsTest, BlockInsertMultipleTypes) {
    MolrsFrameHandle frame{};
    ASSERT_MOLRS_OK(molrs_frame_new(&frame));

    uint32_t blk_id  = intern("gt_multi");
    uint32_t f_id    = intern("gt_float");
    uint32_t i_id    = intern("gt_int");
    uint32_t u_id    = intern("gt_uint");

    ASSERT_MOLRS_OK(molrs_frame_set_block(frame, blk_id, 0));

    MolrsBlockHandle block{};
    ASSERT_MOLRS_OK(molrs_frame_get_block(frame, blk_id, &block));

    // F column (float by default)
    F f_data[3] = {-1.5f, 2.7f, 3.14f};
    size_t shape1[1] = {3};
    ASSERT_MOLRS_OK(molrs_block_set_F(&block, f_id, f_data, shape1, 1));

    // I column (int32_t by default)
    int32_t i_data[3] = {100, 200, 300};
    ASSERT_MOLRS_OK(molrs_block_set_I(&block, i_id, i_data, shape1, 1));

    // U column (uint32_t by default)
    uint32_t u_data[3] = {1, 2, 3};
    ASSERT_MOLRS_OK(molrs_block_set_U(&block, u_id, u_data, shape1, 1));

    // verify ncols = 3
    size_t ncols = 0;
    ASSERT_MOLRS_OK(molrs_block_ncols(block, &ncols));
    EXPECT_EQ(ncols, 3u);

    // verify dtypes
    MolrsDType dt{};
    ASSERT_MOLRS_OK(molrs_block_col_dtype(block, f_id, &dt));
    EXPECT_EQ(dt, MOLRS_D_TYPE_FLOAT);

    ASSERT_MOLRS_OK(molrs_block_col_dtype(block, i_id, &dt));
    EXPECT_EQ(dt, MOLRS_D_TYPE_INT);

    ASSERT_MOLRS_OK(molrs_block_col_dtype(block, u_id, &dt));
    EXPECT_EQ(dt, MOLRS_D_TYPE_U_INT);

    // zero-copy read F
    const F* f_ptr = nullptr;
    size_t f_len = 0;
    ASSERT_MOLRS_OK(molrs_block_get_F(block, f_id, &f_ptr, &f_len));
    ASSERT_EQ(f_len, 3u);
    EXPECT_FLOAT_EQ(f_ptr[2], 3.14f);

    // zero-copy read I
    const int32_t* i_ptr = nullptr;
    size_t i_len = 0;
    ASSERT_MOLRS_OK(molrs_block_get_I(block, i_id, &i_ptr, &i_len));
    ASSERT_EQ(i_len, 3u);
    EXPECT_EQ(i_ptr[1], 200);

    ASSERT_MOLRS_OK(molrs_frame_drop(frame));
}

// ===== Block Mutable Pointer ==============================================

TEST_F(MolrsTest, BlockMutablePointer) {
    MolrsFrameHandle frame{};
    ASSERT_MOLRS_OK(molrs_frame_new(&frame));

    uint32_t blk_id = intern("gt_mut_blk");
    uint32_t x_id   = intern("gt_x");

    ASSERT_MOLRS_OK(molrs_frame_set_block(frame, blk_id, 0));

    MolrsBlockHandle block{};
    ASSERT_MOLRS_OK(molrs_frame_get_block(frame, blk_id, &block));

    F data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    size_t shape[1] = {4};
    ASSERT_MOLRS_OK(molrs_block_set_F(&block, x_id, data, shape, 1));

    // get mutable pointer
    F* ptr = nullptr;
    size_t len = 0;
    ASSERT_MOLRS_OK(molrs_block_get_F_mut(&block, x_id, &ptr, &len));
    ASSERT_EQ(len, 4u);

    // modify in-place
    for (size_t i = 0; i < len; ++i) {
        ptr[i] *= 10.0f;
    }
    ASSERT_MOLRS_OK(molrs_block_col_commit(&block));

    // verify via copy
    F buf[4] = {};
    ASSERT_MOLRS_OK(molrs_block_copy_F(block, x_id, buf, 4));
    EXPECT_FLOAT_EQ(buf[0], 10.0f);
    EXPECT_FLOAT_EQ(buf[1], 20.0f);
    EXPECT_FLOAT_EQ(buf[2], 30.0f);
    EXPECT_FLOAT_EQ(buf[3], 40.0f);

    ASSERT_MOLRS_OK(molrs_frame_drop(frame));
}

// ===== SimBox =============================================================

TEST_F(MolrsTest, SimBoxCube) {
    F origin[3] = {0, 0, 0};
    bool pbc[3] = {true, true, true};
    MolrsSimBoxHandle sb{};
    ASSERT_MOLRS_OK(molrs_simbox_cube(static_cast<F>(10.0), origin, pbc, &sb));

    F vol = 0;
    ASSERT_MOLRS_OK(molrs_simbox_volume(sb, &vol));
    EXPECT_NEAR(vol, 1000.0, 1e-3);

    F lengths[3] = {};
    ASSERT_MOLRS_OK(molrs_simbox_lengths(sb, lengths));
    EXPECT_NEAR(lengths[0], 10.0, 1e-6);
    EXPECT_NEAR(lengths[1], 10.0, 1e-6);
    EXPECT_NEAR(lengths[2], 10.0, 1e-6);

    F h[9] = {};
    ASSERT_MOLRS_OK(molrs_simbox_h(sb, h));
    EXPECT_NEAR(h[0], 10.0, 1e-6);  // h[0][0]
    EXPECT_NEAR(h[4], 10.0, 1e-6);  // h[1][1]
    EXPECT_NEAR(h[8], 10.0, 1e-6);  // h[2][2]
    EXPECT_NEAR(h[1], 0.0, 1e-10);  // off-diagonal

    F tilts[3] = {};
    ASSERT_MOLRS_OK(molrs_simbox_tilts(sb, tilts));
    EXPECT_NEAR(tilts[0], 0.0, 1e-10);
    EXPECT_NEAR(tilts[1], 0.0, 1e-10);
    EXPECT_NEAR(tilts[2], 0.0, 1e-10);

    bool pbc_out[3] = {};
    ASSERT_MOLRS_OK(molrs_simbox_pbc(sb, pbc_out));
    EXPECT_TRUE(pbc_out[0]);
    EXPECT_TRUE(pbc_out[1]);
    EXPECT_TRUE(pbc_out[2]);

    ASSERT_MOLRS_OK(molrs_simbox_drop(sb));
    EXPECT_NE(molrs_simbox_drop(sb), MOLRS_STATUS_OK);  // double drop
}

TEST_F(MolrsTest, SimBoxOrtho) {
    F lens[3] = {2, 3, 4};
    F origin[3] = {0, 0, 0};
    bool pbc[3] = {true, true, true};
    MolrsSimBoxHandle sb{};
    ASSERT_MOLRS_OK(molrs_simbox_ortho(lens, origin, pbc, &sb));

    F vol = 0;
    ASSERT_MOLRS_OK(molrs_simbox_volume(sb, &vol));
    EXPECT_NEAR(vol, 24.0, 1e-3);

    ASSERT_MOLRS_OK(molrs_simbox_drop(sb));
}

TEST_F(MolrsTest, SimBoxWrap) {
    F origin[3] = {0, 0, 0};
    bool pbc[3] = {true, true, true};
    MolrsSimBoxHandle sb{};
    ASSERT_MOLRS_OK(molrs_simbox_cube(static_cast<F>(10.0), origin, pbc, &sb));

    // (11, -1, 21) -> should wrap to (1, 9, 1)
    F xyz_in[6]  = {11, -1, 21, 0.5, 0.5, 0.5};
    F xyz_out[6] = {};
    ASSERT_MOLRS_OK(molrs_simbox_wrap(sb, xyz_in, xyz_out, 2));

    EXPECT_NEAR(xyz_out[0], 1.0, 1e-4);
    EXPECT_NEAR(xyz_out[1], 9.0, 1e-4);
    EXPECT_NEAR(xyz_out[2], 1.0, 1e-4);

    ASSERT_MOLRS_OK(molrs_simbox_drop(sb));
}

TEST_F(MolrsTest, SimBoxShortestVector) {
    F origin[3] = {0, 0, 0};
    bool pbc[3] = {true, true, true};
    MolrsSimBoxHandle sb{};
    ASSERT_MOLRS_OK(molrs_simbox_cube(static_cast<F>(10.0), origin, pbc, &sb));

    F r1[3] = {0.5, 0, 0};
    F r2[3] = {9.5, 0, 0};
    F dr[3] = {};
    ASSERT_MOLRS_OK(molrs_simbox_shortest_vector(sb, r1, r2, dr, 1));

    // shortest path across periodic boundary: -1.0 in x
    EXPECT_NEAR(dr[0], -1.0, 1e-4);
    EXPECT_NEAR(dr[1],  0.0, 1e-4);
    EXPECT_NEAR(dr[2],  0.0, 1e-4);

    ASSERT_MOLRS_OK(molrs_simbox_drop(sb));
}

TEST_F(MolrsTest, SimBoxTriclinic) {
    // upper-triangular cell matrix
    F h9[9] = {
        2, 1, 2,
        0, 4, 3,
        0, 0, 5,
    };
    F origin[3] = {0, 0, 0};
    bool pbc[3] = {true, true, true};
    MolrsSimBoxHandle sb{};
    ASSERT_MOLRS_OK(molrs_simbox_new(h9, origin, pbc, &sb));

    F tilts[3] = {};
    ASSERT_MOLRS_OK(molrs_simbox_tilts(sb, tilts));
    EXPECT_NEAR(tilts[0], 1.0, 1e-6);  // xy
    EXPECT_NEAR(tilts[1], 2.0, 1e-6);  // xz
    EXPECT_NEAR(tilts[2], 3.0, 1e-6);  // yz

    ASSERT_MOLRS_OK(molrs_simbox_drop(sb));
}

// ===== Frame <-> SimBox =====================================================

TEST_F(MolrsTest, FrameSimBoxAssociation) {
    MolrsFrameHandle frame{};
    ASSERT_MOLRS_OK(molrs_frame_new(&frame));

    F origin[3] = {0, 0, 0};
    bool pbc[3] = {true, true, true};
    MolrsSimBoxHandle sb{};
    ASSERT_MOLRS_OK(molrs_simbox_cube(static_cast<F>(5.0), origin, pbc, &sb));

    ASSERT_MOLRS_OK(molrs_frame_set_simbox(frame, sb));

    MolrsSimBoxHandle sb2{};
    ASSERT_MOLRS_OK(molrs_frame_get_simbox(frame, &sb2));

    F vol = 0;
    ASSERT_MOLRS_OK(molrs_simbox_volume(sb2, &vol));
    EXPECT_NEAR(vol, 125.0, 1e-3);

    // clear and verify absence
    ASSERT_MOLRS_OK(molrs_frame_clear_simbox(frame));
    MolrsSimBoxHandle sb3{};
    EXPECT_NE(molrs_frame_get_simbox(frame, &sb3), MOLRS_STATUS_OK);

    molrs_simbox_drop(sb);
    molrs_simbox_drop(sb2);
    ASSERT_MOLRS_OK(molrs_frame_drop(frame));
}

// ===== ForceField =========================================================

TEST_F(MolrsTest, ForceFieldLifecycle) {
    MolrsForceFieldHandle ff{};
    ASSERT_MOLRS_OK(molrs_ff_new("gtest_ff", &ff));

    ASSERT_MOLRS_OK(molrs_ff_def_bondstyle(ff, "harmonic"));
    ASSERT_MOLRS_OK(molrs_ff_def_anglestyle(ff, "harmonic"));
    ASSERT_MOLRS_OK(molrs_ff_def_atomstyle(ff, "full"));

    size_t count = 0;
    ASSERT_MOLRS_OK(molrs_ff_style_count(ff, &count));
    EXPECT_EQ(count, 3u);

    // define a bond type via unified API
    const char* pk[] = {"k0", "r0"};
    double pv[] = {300.0, 1.4};
    ASSERT_MOLRS_OK(molrs_ff_def_type(ff, "bond", "harmonic", "CT-OH", pk, pv, 2));

    // query style name
    char* cat = nullptr;
    char* name = nullptr;
    ASSERT_MOLRS_OK(molrs_ff_get_style_name(ff, 0, &cat, &name));
    ASSERT_NE(cat, nullptr);
    ASSERT_NE(name, nullptr);
    molrs_free_string(cat);
    molrs_free_string(name);

    ASSERT_MOLRS_OK(molrs_ff_drop(ff));
    EXPECT_NE(molrs_ff_drop(ff), MOLRS_STATUS_OK);  // double drop
}

TEST_F(MolrsTest, ForceFieldPairStyle) {
    MolrsForceFieldHandle ff{};
    ASSERT_MOLRS_OK(molrs_ff_new("gtest_pair", &ff));

    const char* style_pk[] = {"cutoff"};
    double style_pv[] = {10.0};
    ASSERT_MOLRS_OK(molrs_ff_def_pairstyle(ff, "lj/cut", style_pk, style_pv, 1));

    const char* type_pk[] = {"epsilon", "sigma"};
    double type_pv[] = {0.5, 3.4};
    ASSERT_MOLRS_OK(molrs_ff_def_type(ff, "pair", "lj/cut", "Ar", type_pk, type_pv, 2));
    ASSERT_MOLRS_OK(molrs_ff_def_type(ff, "pair", "lj/cut", "Ar-Kr", type_pk, type_pv, 2));

    size_t count = 0;
    ASSERT_MOLRS_OK(molrs_ff_style_count(ff, &count));
    EXPECT_EQ(count, 1u);

    ASSERT_MOLRS_OK(molrs_ff_drop(ff));
}

TEST_F(MolrsTest, ForceFieldJsonRoundtrip) {
    MolrsForceFieldHandle ff{};
    ASSERT_MOLRS_OK(molrs_ff_new("gtest_json", &ff));

    const char* spk[] = {"cutoff"};
    double spv[] = {12.0};
    ASSERT_MOLRS_OK(molrs_ff_def_pairstyle(ff, "lj/cut", spk, spv, 1));

    const char* tpk[] = {"epsilon", "sigma"};
    double tpv[] = {1.0, 3.4};
    ASSERT_MOLRS_OK(molrs_ff_def_type(ff, "pair", "lj/cut", "Ar", tpk, tpv, 2));

    // serialize
    char* json = nullptr;
    size_t json_len = 0;
    ASSERT_MOLRS_OK(molrs_ff_to_json(ff, &json, &json_len));
    ASSERT_NE(json, nullptr);
    EXPECT_GT(json_len, 0u);

    // deserialize
    MolrsForceFieldHandle ff2{};
    ASSERT_MOLRS_OK(molrs_ff_from_json(json, &ff2));

    size_t count = 0;
    ASSERT_MOLRS_OK(molrs_ff_style_count(ff2, &count));
    EXPECT_EQ(count, 1u);

    molrs_free_string(json);
    ASSERT_MOLRS_OK(molrs_ff_drop(ff));
    ASSERT_MOLRS_OK(molrs_ff_drop(ff2));
}

// ===== Full Simulation Loop ===============================================

TEST_F(MolrsTest, FullSimulationLoop) {
    // Mimics the typical C/CUDA engine workflow:
    //   1. Create frame + SimBox
    //   2. Insert position data
    //   3. Read via zero-copy pointer
    //   4. Modify via mutable pointer (simulate GPU write-back)
    //   5. Verify results

    uint32_t atoms_id = intern("gt_sim_atoms");
    uint32_t pos_id   = intern("gt_sim_pos");

    // frame
    MolrsFrameHandle frame{};
    ASSERT_MOLRS_OK(molrs_frame_new(&frame));

    // simbox
    F origin[3] = {0, 0, 0};
    bool pbc[3] = {true, true, true};
    MolrsSimBoxHandle sb{};
    ASSERT_MOLRS_OK(molrs_simbox_cube(static_cast<F>(10.0), origin, pbc, &sb));
    ASSERT_MOLRS_OK(molrs_frame_set_simbox(frame, sb));

    // block + positions
    ASSERT_MOLRS_OK(molrs_frame_set_block(frame, atoms_id, 0));
    MolrsBlockHandle block{};
    ASSERT_MOLRS_OK(molrs_frame_get_block(frame, atoms_id, &block));

    F positions[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    size_t shape[2] = {3, 3};
    ASSERT_MOLRS_OK(molrs_block_set_F(&block, pos_id, positions, shape, 2));

    // zero-copy READ
    const F* read_ptr = nullptr;
    size_t read_len = 0;
    ASSERT_MOLRS_OK(molrs_block_get_F(block, pos_id, &read_ptr, &read_len));
    ASSERT_EQ(read_len, 9u);
    EXPECT_FLOAT_EQ(read_ptr[0], 1.0f);
    EXPECT_FLOAT_EQ(read_ptr[8], 9.0f);

    // zero-copy WRITE (simulate cudaMemcpy device->host)
    F* write_ptr = nullptr;
    size_t write_len = 0;
    ASSERT_MOLRS_OK(molrs_block_get_F_mut(&block, pos_id, &write_ptr, &write_len));
    ASSERT_EQ(write_len, 9u);
    for (size_t i = 0; i < write_len; ++i) {
        write_ptr[i] *= 2.0f;
    }
    ASSERT_MOLRS_OK(molrs_block_col_commit(&block));

    // verify
    F buf[9] = {};
    ASSERT_MOLRS_OK(molrs_block_copy_F(block, pos_id, buf, 9));
    for (size_t i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(buf[i], positions[i] * 2.0f);
    }

    // query SimBox through frame
    MolrsSimBoxHandle frame_sb{};
    ASSERT_MOLRS_OK(molrs_frame_get_simbox(frame, &frame_sb));
    F h[9] = {};
    ASSERT_MOLRS_OK(molrs_simbox_h(frame_sb, h));
    EXPECT_NEAR(h[0], 10.0, 1e-6);

    // cleanup
    molrs_simbox_drop(sb);
    molrs_simbox_drop(frame_sb);
    ASSERT_MOLRS_OK(molrs_frame_drop(frame));
}

// ===== Error Handling =====================================================

TEST_F(MolrsTest, NullPointerErrors) {
    EXPECT_EQ(molrs_frame_new(nullptr), MOLRS_STATUS_NULL_POINTER);
    EXPECT_EQ(molrs_intern_key(nullptr, nullptr), MOLRS_STATUS_NULL_POINTER);
    EXPECT_EQ(molrs_simbox_cube(1.0, nullptr, nullptr, nullptr), MOLRS_STATUS_NULL_POINTER);
}

TEST_F(MolrsTest, InvalidHandleErrors) {
    MolrsFrameHandle bad_frame{999, 999};
    EXPECT_NE(molrs_frame_drop(bad_frame), MOLRS_STATUS_OK);

    MolrsSimBoxHandle bad_sb{999, 999};
    EXPECT_NE(molrs_simbox_drop(bad_sb), MOLRS_STATUS_OK);

    MolrsForceFieldHandle bad_ff{999, 999};
    EXPECT_NE(molrs_ff_drop(bad_ff), MOLRS_STATUS_OK);
}

TEST_F(MolrsTest, BlockCopyBufferTooSmall) {
    MolrsFrameHandle frame{};
    ASSERT_MOLRS_OK(molrs_frame_new(&frame));

    uint32_t blk_id = intern("gt_small_buf_blk");
    uint32_t col_id = intern("gt_small_buf_col");

    ASSERT_MOLRS_OK(molrs_frame_set_block(frame, blk_id, 0));
    MolrsBlockHandle block{};
    ASSERT_MOLRS_OK(molrs_frame_get_block(frame, blk_id, &block));

    F data[5] = {1, 2, 3, 4, 5};
    size_t shape[1] = {5};
    ASSERT_MOLRS_OK(molrs_block_set_F(&block, col_id, data, shape, 1));

    // buffer too small
    F small_buf[2] = {};
    MolrsStatus s = molrs_block_copy_F(block, col_id, small_buf, 2);
    EXPECT_NE(s, MOLRS_STATUS_OK);

    ASSERT_MOLRS_OK(molrs_frame_drop(frame));
}
