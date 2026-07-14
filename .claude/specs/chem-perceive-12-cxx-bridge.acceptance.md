---
slug: chem-perceive-12-cxx-bridge
created: 2026-07-12
criteria:
  - id: ac-001
    summary: The bridge returns Result and never aborts on user chemistry
    type: code
    pass_when: |
      The cxx bridge fn is declared `-> Result<Vec<f64>>`. A molecule whose chemistry has no
              BCC parameter (e.g. containing boron) raises a catchable `rust::Error` on the C++ side;
              the process does NOT abort. A test demonstrates the catch.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      The bridge fn is declared `-> Result<Vec<f64>>` in `build.rs`'s CXX_BRIDGE_SCHEMA (the source of
      truth; `src/bridge.rs` is generated from it). Mechanical proof in the generated `bridge.rs.h`:
      `am1_bcc_assign_frame_from_base(...)` no longer carries `noexcept`, while its neighbours
      (`write_frame_xyz(...) noexcept`) still do — cxx emits `noexcept` for every non-Result extern fn,
      so before this change a C++ `catch` was unreachable BY CONSTRUCTION, and `rust::Error` was not
      even emitted into the header. `molrs-cxxapi/tests/cxx/bridge_probe.cc` now calls the bridge inside
      `try { ... } catch (const rust::Error&)`; the BF4- case (`missing BCC correction for bond B|71|1`)
      is caught and the child process exits 0. It exited on `signal: 6 (SIGABRT)` before. The
      `_result`/`.expect()` two-function split is gone, so there is no longer a site where a Result can
      be discarded.
  - id: ac-002
    summary: ABCG2 is reachable from C++
    type: code
    pass_when: |
      The bridge takes a parameter-set selector; passing ABCG2 produces ABCG2 charges.
              `grep -rn 'abcg2\|parameter_set' molrs-cxxapi/src` returns hits (currently 0).
    status: verified
    last_checked: 2026-07-14
    evidence: |
      `parameter_set: &str` ("bcc" | "abcg2", case-insensitive) crosses the bridge; `parse_bcc_parameter_set`
      refuses an unknown name with an Err rather than silently defaulting to BCC. Tests anchor the expected
      numbers on the existing antechamber oracle's water case (bcc [-0.785, 0.392, 0.392] vs abcg2
      [-0.863, 0.431, 0.431]), and a separate test asserts the two columns actually disagree so an ABCG2
      assertion cannot silently degrade into a BCC one.
  - id: ac-003
    summary: The fake backend and the normalize argument are gone
    type: code
    pass_when: |
      `grep -rnE 'ProvidedAM1ChargeBackend|normalize_total_charge' molrs-cxxapi/src` returns
              0 hits.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      `grep -rnE 'ProvidedAM1ChargeBackend|normalize_total_charge' molrs-cxxapi/src` -> 0 hits, enforced by
      `molrs-cxxapi/tests/source_gate.rs` (the gate scans build.rs too, since src/bridge.rs is generated and
      a gate reading only src/ would be reading a mirror). ProvidedAM1ChargeBackend was already gone (spec 07);
      the gate stays as a regression guard. `total_charge` was ALSO deleted — it was bound as `_total_charge`
      and never read, and there is no correct future reader: AM1-BCC does not renormalize to a target
      (antechamber's am1bcc.c ends at the increment loop). An argument crossing an FFI boundary into an
      underscore is a promise to Atomiverse that molrs does not keep.
  - id: ac-004
    summary: The Atomiverse companion change is declared
    type: manual
    pass_when: |
      The spec names the exact Atomiverse edit required (am1_bcc_charge_assigner.cpp +
              AM1BCCChargeConfig) and it is tracked as a cross-repo dependency, not silently assumed.
    status: verified
    last_checked: 2026-07-14
    evidence: |
      DECLARED, not assumed — and it is a real cross-repo ABI break: Atomiverse will NOT COMPILE against the new
      bridge until updated. Generated C++ declaration went from
      `am1_bcc_assign_frame_from_base(FrameRef&, Slice<const double>, double total_charge, bool normalize_total_charge) noexcept`
      to `am1_bcc_assign_frame_from_base(FrameRef&, Slice<const double>, rust::Str parameter_set)`.
      Three edits Atomiverse owes, in `src/cpu/semiempirical/am1_bcc_charge_assigner.cpp` + `AM1BCCChargeConfig`:
      (1) drop both trailing args at the call site and the matching config fields — normalize_total_charge is no
      longer expressible; (2) add a parameter-set field ("bcc" default, "abcg2" newly reachable) passed as
      rust::Str — this is the whole of ABCG2's reachability from C++; (3) wrap the call in
      `try { ... } catch (const rust::Error& e)` and surface it as the assigner already surfaces its own
      std::runtime_error — `#include "rust/cxx.h"` is required. Letting it escape is now a LIVE failure mode:
      previously the process aborted before any exception existed, so no catch was ever exercised.
      NOT DONE HERE — the Atomiverse repo was not touched.
---
