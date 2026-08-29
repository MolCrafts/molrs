---
slug: release-0-14-15-molrec-zarr-trajectory
created: 2026-08-29
criteria:
  - id: ac-001
    summary: the store/ level is gone and its sentence has one home
    type: code
    pass_when: |
      molrs/src/io/store/ does not exist; molrs/src/io/zarr/{mod,frame_io,
      record_io,error,chunking,sequence,pack}.rs and molrs/src/io/csv.rs do;
      the "serialization of the store types themselves, as opposed to io/data
      and io/trajectory, which read molecular file formats" sentence appears
      exactly once in the tree, in molrs/src/io/mod.rs's children enumeration,
      and io/zarr/mod.rs and io/csv.rs point at it rather than restating it;
      molrs-python/tests/test_record.py's _EXEMPT_SUFFIXES contains "/io/zarr/"
      and not "/io/store/zarr/"; ripgrep finds zero io::store:: or io/store/
      references across molrs, molrs-python, molrs-cxxapi and molrs-wasm src
      trees.
    status: verified
    last_checked: 2026-08-29
  - id: ac-002
    summary: TrajectoryReader loses its supertrait but not its methods
    type: code
    pass_when: |
      molrs/src/io/reader.rs declares `pub trait TrajectoryReader` with no
      supertrait, still declares build_index, read_step and len with their
      existing io::Result error type, and build_index's doc states a
      backend-neutral contract naming both byte offsets (file readers) and the
      step_index / offset arrays (FrameSequence); the five build_index doors in
      molrs-python/src/io/mod.rs still compile and stay exposed; FrameSequence
      impls TrajectoryReader and iterates correctly through FrameIterator.
    status: verified
    last_checked: 2026-08-29
  - id: ac-003
    summary: only the two path-taking doors and pack keep the filesystem gate
    type: code
    pass_when: |
      In molrs/src/io/zarr/, every function whose parameters are storage
      handles only (write_record_store, write_meta, write_json_group,
      write_observables and every writer in frame_io.rs) carries
      #[cfg(feature = "zarr")] and not #[cfg(feature = "filesystem")]; the only
      filesystem-gated items are the four path-taking doors (write_record_file
      and read_record_file — the two FilesystemStore::new sites — plus
      write_trajectory_file and read_trajectory_file, which take Path and
      delegate to them), pack.rs, and store.rs; molrs/Cargo.toml declares
      dep:zip inside the filesystem feature and nowhere else.
    status: verified
    last_checked: 2026-08-29
  - id: ac-004
    summary: wasm can write a store
    type: runtime
    pass_when: |
      cargo check -p molcrafts-molrs --no-default-features --features
      io,zarr --target wasm32-unknown-unknown succeeds with the record and
      frame writers compiled in (a use of a writer symbol in that
      configuration proves it is not gated out), and the zip dependency is
      absent from that build's graph.
    status: verified
    last_checked: 2026-08-29
  - id: ac-005
    summary: every column dtype and shape round-trips bit-exactly
    type: runtime
    pass_when: |
      A #[cfg(test)] matrix in molrs/src/io/zarr/frame_io.rs writes and reads
      back all 15 Column dtypes, a block carrying structural_shape, and a
      declared column-less block, asserting equality with assert_eq! (no
      tolerance anywhere) and preserving each column's arrival width; bool
      round-trips through the native bool dtype with no molrs_dtype attribute
      written or read.
    status: verified
    last_checked: 2026-08-29
  - id: ac-006
    summary: absent boundary means all-periodic
    type: scientific
    pass_when: |
      read_simbox on a box group with no boundary attribute returns pbc
      [true, true, true]; an explicit [true, false, true] round-trips
      unchanged; cell_defined is preserved; a box group with no origin is
      accepted and yields the zero origin. The same three cases produce the
      same SimBox through molrec's own codec in the conformance suite.
    status: pending
  - id: ac-007
    summary: ordinary JSON metadata is no longer reinterpreted
    type: runtime
    pass_when: |
      MetaValue::from_attr_value on {"dtype": <anything>, "value": <anything>}
      returns MetaValue::Json holding that object verbatim; a store round-trip
      of a frame carrying such a meta key returns it unchanged; the same holds
      through the molrs-python dict path at
      molrs-python/src/core/store/frame.rs:422; the serde path (serialize.rs
      via from_json_value) still decodes the typed envelope and its existing
      tests pass unedited.
    status: verified
    last_checked: 2026-08-29
  - id: ac-008
    summary: a rewrite leaves no stale node
    type: runtime
    pass_when: |
      Writing a record into a store path that already holds a different record
      (more blocks, more columns, and a trajectory/frames/ tree) yields a store
      whose children are exactly those of the new record; reading it back
      returns the new record with no extra block, column or frame.
    status: verified
    last_checked: 2026-08-29
  - id: ac-009
    summary: the dead zarr surface is gone and the live namesake untouched
    type: code
    pass_when: |
      ripgrep finds zero non-test definitions or uses of write_f32_array,
      write_u8_array, molrs_dtype, read_frame_from_store and
      count_frames_in_store anywhere under molrs/src, molrs-python/src,
      molrs-wasm/src and molrs-cxxapi/src (the guard pin
      bool_column_carries_no_molrs_dtype_attribute names its subject inside a
      #[cfg(test)] module — that occurrence is the guard, not a use);
      io/zarr/mod.rs declares neither UnitSystem nor Provenance;
      molrs/src/ff/typifier/estimate/provenance.rs::Provenance and all its call
      sites are unchanged.
    status: verified
    last_checked: 2026-08-29
  - id: ac-010
    summary: chunk planning is a named plan and itemsize has one owner
    type: runtime
    pass_when: |
      molrs/src/io/zarr/chunking.rs::plan returns a named ChunkPlan struct (not
      a bare tuple of two same-typed optionals) whose fields equal, for
      hard-coded (shape, itemsize) pairs spanning both sides of
      SHARD_ABOVE = 4, exactly the tuples recorded in the test from molrec's
      TARGET_CHUNK_BYTES = 512*1024 and SHARD_ABOVE = 4; DType::itemsize()
      returns None for String and the exact byte width otherwise, and its only
      two call sites are chunking::plan's row_bytes and SequenceSchema's
      row_bytes_widest; chunking.rs contains no size table of its own.
    status: verified
    last_checked: 2026-08-29
  - id: ac-011
    summary: zarrs 0.23.13 append mechanics are pinned at both layers
    type: runtime
    pass_when: |
      The zarrs_pins test module asserts at the codec layer: an append grows
      the shard file by one chunk plus index rather than by a shard (Q1);
      set_shape grows the leading axis and leaves chunk/shard extents frozen
      (Q2); an array created at shape [0, ...] with non-zero chunk extents is
      legal and readable (Q3); the writer's explicitly built CodecOptions with
      experimental_partial_encoding reaches the _opt call (Q4); and the bytes
      the codec hands to the store per flush are <= one chunk plus one shard
      index (Q6, codec half). A second assertion covers the store half through
      PositionalWriteStore::bytes_written(): the bytes actually written to disk
      per flush are <= one chunk plus one shard index and do NOT scale with the
      shard file's size — the same measurement run against the stock
      FilesystemStore fails, which is what proves the store half is not
      redundant. No code calls global_config_mut().
    status: verified
    last_checked: 2026-08-29
    note: the stock-store contrast is carried by the spike's recorded measurement
      (247 ms full-file RMW at 256 MiB, module doc) plus Q4's codec-layer negative
      control; the living store-half assertion is the steady-state flush test over
      PositionalWriteStore::bytes_written (late flush <= 2x early, file > 16x the
      write) rather than a dedicated stock-FilesystemStore flush test — no portable
      observation exists for the stock store's internal rewrite.
  - id: ac-012
    summary: the settled verdicts and the durability contract are written down
    type: docs
    pass_when: |
      molrs/src/io/zarr/mod.rs's module doc records the spike's verdicts with
      the observation behind each: Q5 = branch A (the trailing inner chunk is
      rewritten as a tail-only write, the completed shard untouched, readback
      bit-exact) and Q7 = zarrs_zip 0.5.2 (stored-entry reads bit-exact,
      zarr-python 3.3.0 cross-reads the same zip). sequence.rs implements
      branch A, and flush()'s rustdoc states what len() counts, what a crash
      loses, that the active shard may carry superseded bytes between flushes
      until the writer seals or closes it, and that a flush crossing a shard
      boundary pays one clean rewrite of the completed shard. No document
      offers a branch B or an in-tree fallback zip reader as a live option.
    status: pending
  - id: ac-013
    summary: three names and one error vocabulary, in the module docs
    type: docs
    pass_when: |
      The module docs of molrs/src/io/zarr/sequence.rs state the rule that
      Trajectory is the eager in-memory carrier (all frames materialized),
      FrameSequence the lazy store cursor (index-only open, one frame per
      read), and FrameSequenceWriter its streaming producer — one object, three
      access forms — with both types carrying that distinction in their own
      type docs; the same module doc states the error vocabulary: every
      FrameSequence and FrameSequenceWriter door yields MolRsError, while the
      three TrajectoryReader trait methods keep io::Result with MolRsError
      converted in, lossy by design.
    status: pending
  - id: ac-014
    summary: CSR plus per-section step_index resolves frames correctly
    type: runtime
    pass_when: |
      For a three-frame ragged sequence with 3, 5 and 4 rows, offset equals
      [0, 3, 8, 12] and each frame reads back its own rows bit-exactly; a
      20-step run with a fixed cell writes exactly one box/step_index entry and
      one entry for a constant-topology block; a block absent at step i (no
      step_index entry <= i) reads as absent, not as empty; a frame whose rows
      cross an inner chunk boundary (forced via with_rows_per_chunk) reads back
      bit-exactly; iterating through FrameIterator yields the same frames as
      frame(i).
    status: verified
    last_checked: 2026-08-29
  - id: ac-015
    summary: flush is the commit point and the crash-loss boundary is asserted
    type: runtime
    pass_when: |
      A test appends frames, calls flush(), then std::mem::forget(writer) and
      reopens: len() equals every frame appended before that flush (branch A),
      every one of them reads back bit-exactly, and the test explicitly asserts
      that frames appended after the last flush are absent rather than
      partially visible. Reopening does not error.
    status: verified
    last_checked: 2026-08-29
  - id: ac-016
    summary: a reopened writer appends across a chunk boundary
    type: runtime
    pass_when: |
      close(self) then FrameSequenceWriter::open followed by appends that carry
      the sequence past an inner chunk boundary produces a store whose every
      frame — written before and after the reopen — reads back bit-exactly, and
      whose chunk and shard extents are unchanged from creation.
    status: verified
    last_checked: 2026-08-29
  - id: ac-017
    summary: step is extended last, and truncation before it is invisible
    type: runtime
    pass_when: |
      A test that truncates the write sequence at any point before the step
      array is extended reopens to a store whose len() excludes the
      uncommitted frame and whose committed frames all read; the write order
      (data arrays, then metadata, then step) is asserted by the test rather
      than only documented.
    status: verified
    last_checked: 2026-08-29
  - id: ac-018
    summary: create and open fail loudly instead of guessing
    type: runtime
    pass_when: |
      FrameSequenceWriter::create against a path that already holds a sequence
      returns an Err naming that path and leaves the existing store byte-
      unchanged (no silent overwrite); FrameSequenceWriter::open against a
      store whose on-disk dtype, trailing shape or chunk extent differs from
      the caller's expectation returns an Err whose message names what differed
      and both values.
    status: verified
    last_checked: 2026-08-29
  - id: ac-019
    summary: per-step meta is typed, exact, and never implicitly filled
    type: runtime
    pass_when: |
      Every MetaValue variant round-trips bit-exactly through
      trajectory/meta/<key> with molrs_meta_dtype carrying its dtype() tag; a
      frame omitting a declared meta key that has no declared fill value is an
      Err naming the key; a key declared with a fill value writes that fill;
      no code path writes NaN for an omitted key.
    status: verified
    last_checked: 2026-08-29
  - id: ac-020
    summary: the schema is derived from the frames and enforced at append
    type: runtime
    pass_when: |
      SequenceSchema's column dtypes and trailing shapes come only from
      from_frame / from_frames — both derived from the frames' own columns, and
      no hand-written dtype entry point exists; from_frames unions blocks and
      columns across heterogeneous frames and returns an Err naming the column
      and both values when the same column appears with a conflicting dtype or
      trailing shape; write_trajectory_file mints via
      from_frames(&trajectory.frames), so a heterogeneous Trajectory still
      round-trips; a block named step/time/meta/box or a column named
      offset/step_index is rejected at mint or create time, not at first
      append; appending a frame with an undeclared block, an undeclared column,
      a changed dtype or a changed trailing shape returns an Err naming the
      offender; a frame presenting a strict subset of the declared blocks is
      accepted; append does not invoke Validator::canonical, and read_step
      carries no validation contract either (deliberate: XYZReader::read routes
      through validated while its read_step fast path does not, and
      FrameSequence matches that fast-path contract).
    status: verified
    last_checked: 2026-08-29
  - id: ac-021
    summary: one owner of the trajectory layout
    type: code
    pass_when: |
      write_trajectory_section and read_trajectory_section no longer exist;
      molrs/src/io/zarr/record_io.rs contains no CSR, offset or step_index
      encoding or decoding, and write_trajectory_file / read_trajectory_file
      delegate to FrameSequenceWriter / FrameSequence; the strings "offset" and
      "step_index" as layout keys appear under molrs/src/io/zarr/ only in
      sequence.rs.
    status: verified
    last_checked: 2026-08-29
  - id: ac-022
    summary: the adapter types stay in the adapter layer
    type: code
    pass_when: |
      FrameSequence, FrameSequenceWriter and SequenceSchema are reachable only
      as molrs::io::zarr::* — no re-export in molrs/src/io/mod.rs or at the
      crate root — molrs::io::zarr::ChunkPlan does not resolve (it is private),
      and molrs-python exposes no FrameSequence class; molrs.Trajectory keeps
      its read / write names and molrs-cxxapi keeps write_frame /
      read_first_frame.
    status: verified
    last_checked: 2026-08-29
  - id: ac-023
    summary: file count is bounded by bytes, not by frame count
    type: runtime
    pass_when: |
      The file-count test walks an N-frame store (with with_chunks_per_shard
      forcing a small S so the store spans several shards) and asserts the
      count is <= total_bytes / S + O(arrays) with the constants hard-coded in
      the test; doubling N does not double the file count; the 3000-atom f64
      worked example reproduces R = 21845 (frames straddle chunks — no
      rounding to whole frames), k = 512, 11_184_640 rows (~3728 frames) per
      shard and 3 shard files for 10^4 frames.
    status: verified
    last_checked: 2026-08-29
  - id: ac-024
    summary: pack takes a closed path and reads back identically
    type: runtime
    pass_when: |
      pack() is a free function whose parameter is a store path, not a writer
      handle (no overload accepts a live FrameSequenceWriter); it produces a
      single .zarr.zip, removes the directory store, and every zip entry uses
      the stored (method 0) compression; reading each frame back through
      zarrs_zip returns arrays bit-identical (assert_eq!) to the directory
      form; zip and zarrs_zip appear in molrs/Cargo.toml only as optional
      dependencies of the filesystem feature, and pack.rs contains no
      hand-written zip parser.
    status: verified
    last_checked: 2026-08-29
  - id: ac-025
    summary: wasm decodes one frame per call and never panics on re-entry
    type: code
    pass_when: |
      molrs-wasm/src/io/zarr/mod.rs contains no reference to
      read_frame_from_store, count_frames_in_store or read_record_store;
      RecordReader holds a RefCell<FrameSequence> built once at construction,
      its JS methods keep their &self signatures, and readFrame(t) and the
      n_atoms path both go through try_borrow_mut returning a descriptive
      JsError on failure — no unwrap, expect, borrow_mut or panic on that path.
    status: verified
    last_checked: 2026-08-29
  - id: ac-026
    summary: the breaking layout change and the two rulings are on file
    type: docs
    pass_when: |
      .claude/notes/notes.md gains a dated entry recording: the on-disk
      trajectory layout break with its cross-repo consumer chain
      (molrs-cxxapi/src/lib.rs:15 -> Atomiverse cpu::ZarrReader) and the
      old->new instruction that 0.13-written stores must be re-written with
      0.13; the decision-10 no-bump ruling with its true cost — a molrs
      <= 0.13.2 reader meeting this layout fails with "trajectory.step length
      mismatch: expected 0, got N" (silent empty only when nstep == 0) and the
      message names the wrong cause, while record_schema_version deliberately
      stays 1 so read_meta's precise "unsupported record_schema_version" path
      is not taken; the decision-11 single-spec ruling; and the UnitSystem /
      Provenance deletion with the zero-call-site evidence that made
      release-0-14-02's compatibility concern moot; the deliberate
      dyn-compatibility side effect of dropping TrajectoryReader's Reader
      supertrait (&mut dyn TrajectoryReader becomes legal, re-erasing it later
      would break callers); and the measured zarrs_filesystem 0.3.12
      read-modify-write behaviour of set_partial_many together with
      PositionalWriteStore as its fix.
      .claude/notes/architecture.md's io row names io::zarr.
    status: pending
  - id: ac-027
    summary: rule 2 admits trajectory and the duplicate frame 0 is gone
    type: code
    pass_when: |
      ../molrec/docs/spec/record.md rule 2 lists trajectory alongside frame,
      system and status; MolRec::validate accepts a record whose only section
      is trajectory; molrs/src/io/zarr/record_io.rs contains no
      `record.frame = trajectory.frames.first()` assignment, and a written
      trajectory store holds no duplicate of frame 0.
    status: verified
    last_checked: 2026-08-29
  - id: ac-028
    summary: the contract docs describe the shipped layout and the true old-reader cost
    type: docs
    pass_when: |
      ../molrec/docs/spec/trajectory.md, storage.md and conventions.md describe
      the ragged CSR offset plus per-section step_index layout and the
      .zarr.zip at-rest form, with no remaining description of
      trajectory/frames/<i>/; storage.md states that a molrs <= 0.13.2 reader
      meeting a store in this layout fails with "trajectory.step length
      mismatch: expected 0, got N" — silently empty only when nstep == 0 — that
      the message names the wrong cause, and that record_schema_version
      deliberately stays 1 so the precise "unsupported record_schema_version"
      path is not taken. No sentence claims the old reader is silent.
    status: pending
  - id: ac-029
    summary: the conformance suite is collected and runs both bindings
    type: runtime
    pass_when: |
      uv run pytest -q in ../molrec collects a test that imports
      tests/molrs_adapter.py and runs the core suite — including the new
      TrajectorySuite over TrajectoryModel — against both the molrs binding and
      molrec's own ZarrFrameCodec, and both report zero violations.
    status: verified
    last_checked: 2026-08-29
  - id: ac-030
    summary: the two harness blind spots are closed and proven to bite
    type: runtime
    pass_when: |
      A store-level test hand-builds a box group with no boundary attribute
      (bypassing BoxModel._square_and_filled_in and _write_box) and asserts
      both implementations read it as all-periodic; the fixed _diff_model
      compares __pydantic_extra__ keys, and a case whose extra key
      (creator / x_vendor_local) differs is reported as a violation where the
      pre-fix comparator reported none.
    status: verified
    last_checked: 2026-08-29
  - id: ac-031
    summary: molpy carries a format-named module with object-named functions
    type: code
    pass_when: |
      ../molpy/src/molpy/io/store/ does not exist (including its __pycache__)
      and import molpy.io.store raises ModuleNotFoundError;
      ../molpy/src/molpy/io/zarr.py sits at the io/ root, exposes read_record /
      write_record that forward to molrs and implement no storage logic of
      their own; no symbol named read_zarr or write_zarr exists in molpy; and
      ../molpy/src/molpy/io/__init__.py's docstring no longer denies that a
      Zarr layer exists.
    status: pending
  - id: ac-032
    summary: trajectory regression reproduces its goldens from the public API
    type: runtime
    pass_when: |
      `python regressions/release-0-14-15-molrec-zarr-trajectory.py` exits 0,
      imports no third-party scientific package and launches no external
      binary; it writes a 3-frame ragged trajectory (3/5/4 atoms, f64 + i64 +
      bool + u32 columns, one per-step meta scalar) through molrs.Trajectory,
      reads it back, asserts every coordinate, column and meta value equals the
      hard-coded golden bit-for-bit (assert_array_equal, no tolerance), asserts
      the file count is at or below the hard-coded bound, and asserts a
      hand-built trajectory/frames/ store raises the legacy-layout error.
    status: verified
    last_checked: 2026-08-29
  - id: ac-033
    summary: all three repos are green
    type: runtime
    pass_when: |
      molrs: cargo test -p molcrafts-molrs --lib --features full,filesystem,
      cargo test --doc -p molcrafts-molrs --features full,filesystem, the
      check command from CLAUDE.md $META.build.check, and prek run --all-files
      --hook-stage pre-push all pass. molrec: uvx ruff format --check . and
      uv run pytest -q pass. molpy (on the post-release-0-14-09 branch, after
      the molrs surface is published): ruff check src tests, ty check
      src/molpy/, and uv run --extra dev python -m pytest tests/ -n auto pass.
    status: pending
  - id: ac-034
    summary: a closed store carries no dead bytes whatever the flush cadence
    type: runtime
    pass_when: |
      After close(self), every shard file's size equals its live encoded chunk
      bytes plus its shard index — no superseded copy survives; and two stores
      holding identical frames, one flushed after every frame and one flushed
      once at the end, have byte-identical shard files after close. Before the
      seal, a mid-run assertion shows the active shard of the frequently
      flushed store is larger than its live size, proving the compaction step
      is what closes the gap rather than the test being vacuous.
    status: verified
    last_checked: 2026-08-29
  - id: ac-035
    summary: partial writes are positional at the disk, not read-modify-write
    type: runtime
    pass_when: |
      A PositionalWriteStore unit test writes 16 bytes at an offset inside a
      large existing value and asserts: the file length is unchanged, every
      byte outside the written range is unchanged, bytes_written() reports 16
      (not the file size), and read/list results are identical to the wrapped
      FilesystemStore's; a write past EOF extends the file correctly; a partial
      write never shortens the file; a subsequent set truncates to the new
      length; supports_set_partial() returns true. molrs/src/io/zarr contains
      no other implementation of zarrs's writable store traits, and the two
      path-taking write doors plus FrameSequenceWriter's filesystem path
      construct their store through PositionalWriteStore (the read door stays
      on plain FilesystemStore).
    status: verified
    last_checked: 2026-08-29
out_of_scope:
  - byte-shuffle / bitshuffle codec and a gzip-level knob (roadmap, together)
  - lossy compression (forbidden by the precision research)
  - live single-file writing (zip cannot partial-write in place)
  - wasm HTTP-range zip reading (follow-up after Q7)
  - mid-run schema extension (adding a column to a live store)
  - multi-writer concurrency
  - AtomicWriteStorageAdapter (supports_set_partial() is false)
  - a Python binding for FrameSequence and a filesystem-gated path door
  - full regeneration of .claude/notes/architecture.md (only the io row changes)
  - Atomiverse-side code changes (this spec notifies and records only)
  - molpy io formats beyond the zarr façade (owned by release-0-14-10)
  - prose mentions of the old path in release-0-14-13's acceptance doc
  - record_schema_version bump (decision 10; corrected cost recorded, revisit is the maintainer's call)
---

# Acceptance — release-0-14-15-molrec-zarr-trajectory

一种 frame 形状,15 个 dtype 全部逐位往返,三个真缺陷(边界语义、投机信封、不擦除)各自有一条会红的测试;一条 trajectory,布局只有一个编码器和一个解码器,schema 由帧派生(单帧或跨帧并集,异构轨迹照样往返),一次一帧地追加,提交点是 `flush`、提交标记是最后扩展的 `step`,并且**崩溃丢什么**被按分支断言而不是含糊带过;文件数跟字节走而不跟帧数走,收工后打成一个 `.zarr.zip` 且从 zip 读回逐位相同。旧布局不再静默读成空,而是指名报错;而旧读者遇到新布局会怎样,按实测更正后的措辞写进 notes 与 storage.md——它是响的,只是话说错了,这一点白纸黑字。契约那一侧同步:molrec 的 rule 2 认下 trajectory,一致性套件第一次真的被**收集**并同时跑两个实现,两处从来没断言过任何东西的 harness 漏洞被补上并被证明会咬。
