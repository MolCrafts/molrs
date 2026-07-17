// A C++ caller of the AM1-BCC bridge, for the Rust integration tests.
//
// This is the C++ half of `tests/am1bcc_bridge.rs`, and it exists because the
// criterion it serves is a claim ABOUT C++: chem-perceive-12 ac-001 says a
// molecule whose chemistry has no BCC parameter must raise "a catchable
// `rust::Error` on the C++ side; the process does NOT abort". No Rust-only test
// can witness that. `am1_bcc_assign_frame_from_base_result` in lib.rs already
// returns `Err` today and always has — what aborts is the `.expect()` the bridge
// wraps it in, and the abort happens *in the FFI shim*, on the far side of a
// declaration cxx currently marks `noexcept`:
//
//     ::rust::Vec<double> am1_bcc_assign_frame_from_base(         // TODAY
//         ::molrs::FrameRef&, ::rust::Slice<double const>,
//         double total_charge, bool normalize_total_charge) noexcept;
//
// `noexcept` is not decoration: cxx emits it for every `extern "Rust"` fn that
// does not return `Result`, so a `catch` here is unreachable BY CONSTRUCTION and
// a Rust panic can only abort the process. Declaring the fn `-> Result<Vec<f64>>`
// is what makes cxx drop the `noexcept` and throw `rust::Error` instead. That is
// the whole of ac-001, and this file is what can see it.
//
// It is compiled by build.rs against the GENERATED header (`src/bridge.rs.h`,
// which build.rs rewrites from `CXX_BRIDGE_SCHEMA` on every build) — never
// against a hand-written prototype, which would only prove that this file agrees
// with itself.
//
//
// ── The signature this file is written against ───────────────────────────────
//
// The tests pin the post-spec-12 bridge to:
//
//     fn am1_bcc_assign_frame_from_base(
//         fref: &mut FrameRef,
//         am1_charges: &[f64],
//         parameter_set: &str,          // "bcc" | "abcg2"
//     ) -> Result<Vec<f64>>;
//
// `&str` is the selector, because `&str` is what this bridge already uses for
// every other name it takes (`block`, `col`, `path`), and an unknown value gives
// the Result something to refuse.
//
// One degree of freedom is left open on purpose: the spec hands the implementer
// the choice of whether the (already-unused, bound as `_total_charge`) argument
// dies with `normalize_total_charge` or stays. So `call_assign` accepts EITHER
// arity via `if constexpr`, and neither choice can turn this test red.
//
// It also still accepts today's pre-spec signature — the one with
// `normalize_total_charge` and no selector — and reports it as `kNoSelector`.
// That is what keeps the RED honest: the probe COMPILES today, so the tests fail
// on what the bridge DOES (aborts; cannot reach ABCG2) rather than on a C++
// compile error, which would prove nothing about behaviour.

#include "bridge.rs.h"

// The cxx runtime header, in full.
//
// `bridge.rs.h` embeds only the pieces of it the bridge actually uses, and today
// `rust::Error` is not among them — cxx emits that class when some `extern "Rust"`
// fn returns `Result`, and none does. (Try deleting this line once the bridge is
// fixed: it will still be needed, but the error it currently produces —
// "no type named 'Error' in namespace 'rust'" — is itself a fingerprint of the
// bug, printed by the code generator.)
#include "rust/cxx.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <type_traits>

static_assert(sizeof(::molrs::Element) == sizeof(std::uint8_t));
static_assert(static_cast<std::uint8_t>(::molrs::Element::H) == 1);
static_assert(static_cast<std::uint8_t>(::molrs::Element::O) == 8);
static_assert(static_cast<std::uint8_t>(::molrs::Element::Cu) == 29);
static_assert(static_cast<std::uint8_t>(::molrs::Element::Og) == 118);
static_assert(
    std::is_same_v<decltype(::molrs::cxx_api_version()), std::uint32_t>);
static_assert(
    std::is_same_v<decltype(::molrs::cxx_api_capabilities()), std::uint64_t>);

namespace {

// Status codes shared with the Rust side (`tests/am1bcc_bridge.rs`).
constexpr int kOk = 0;
constexpr int kCaughtRustError = 1;
constexpr int kCaughtStdException = 2;
constexpr int kCaughtUnknown = 3;
constexpr int kNoSelector = 4;

template <typename>
constexpr bool always_false = false;

void copy_message(const char *what, char *err, std::size_t err_cap) {
  if (err == nullptr || err_cap == 0) {
    return;
  }
  const std::size_t n = std::strlen(what);
  const std::size_t fit = n < err_cap - 1 ? n : err_cap - 1;
  std::memcpy(err, what, fit);
  err[fit] = '\0';
}

// Call the bridge whatever arity it was declared with.
//
// `has_selector` comes back false only for the pre-spec bridge, whose parameter
// set is hardcoded to `BccParameterSet::Bcc` in lib.rs and therefore cannot be
// chosen from C++ at all.
template <typename F>
::rust::Vec<double> call_assign(F fn, ::molrs::FrameRef &fref,
                                ::rust::Slice<const double> am1,
                                ::rust::Str parameter_set, bool &has_selector) {
  has_selector = true;

  if constexpr (std::is_invocable_v<F, ::molrs::FrameRef &,
                                    ::rust::Slice<const double>, ::rust::Str>) {
    // Contract shape, `total_charge` dropped.
    return fn(fref, am1, parameter_set);
  } else if constexpr (std::is_invocable_v<F, ::molrs::FrameRef &,
                                           ::rust::Slice<const double>, double,
                                           ::rust::Str>) {
    // Contract shape, `total_charge` kept (it is unused either way).
    return fn(fref, am1, 0.0, parameter_set);
  } else if constexpr (std::is_invocable_v<F, ::molrs::FrameRef &,
                                           ::rust::Slice<const double>, double,
                                           bool>) {
    // Pre-spec-12 shape: `(…, total_charge, normalize_total_charge)`, `noexcept`,
    // BCC hardcoded. `false` is the only value of `normalize_total_charge` the
    // impl does not reject outright.
    has_selector = false;
    return fn(fref, am1, 0.0, false);
  } else {
    static_assert(always_false<F>,
                  "am1_bcc_assign_frame_from_base has a shape this probe does "
                  "not know; see the pinned signature at the top of this file");
  }
}

} // namespace

extern "C" {

// Call the bridge from C++ inside a try/catch and report what came back.
//
// @param frame_ref   `*mut molrs_cxxapi::FrameRef` — the frame, built on the
//                    Rust side and handed over as an opaque pointer, exactly as
//                    cxx hands `rust::Box<FrameRef>` to Atomiverse.
// @param am1         AM1 base charges (one per atom).
// @param parameter_set  NUL-terminated "bcc" / "abcg2" (or garbage, to test the
//                    selector's refusal).
// @param out         corrected charges, on kOk.
// @param err         the exception's `what()`, on kCaught*.
// @return kOk / kCaughtRustError / kCaughtStdException / kCaughtUnknown, or
//         kNoSelector when the bridge takes no parameter set (pre-spec-12).
int molrs_cxxapi_test_am1_bcc_assign(void *frame_ref, const double *am1,
                                     std::size_t n_am1,
                                     const char *parameter_set, double *out,
                                     std::size_t out_cap, std::size_t *out_len,
                                     char *err, std::size_t err_cap) {
  *out_len = 0;
  if (err != nullptr && err_cap > 0) {
    err[0] = '\0';
  }

  auto &fref = *static_cast<::molrs::FrameRef *>(frame_ref);
  bool has_selector = false;

  try {
    ::rust::Vec<double> charges = call_assign(
        &::molrs::am1_bcc_assign_frame_from_base, fref,
        ::rust::Slice<const double>(am1, n_am1), ::rust::Str(parameter_set),
        has_selector);

    if (!has_selector) {
      return kNoSelector;
    }

    const std::size_t n = charges.size() < out_cap ? charges.size() : out_cap;
    for (std::size_t i = 0; i < n; ++i) {
      out[i] = charges[i];
    }
    *out_len = n;
    return kOk;
  } catch (const ::rust::Error &e) {
    // The one branch ac-001 is about: molrs refused the molecule's chemistry and
    // said so in an exception, and this process is still alive to read it.
    copy_message(e.what(), err, err_cap);
    return kCaughtRustError;
  } catch (const std::exception &e) {
    copy_message(e.what(), err, err_cap);
    return kCaughtStdException;
  } catch (...) {
    copy_message("unknown C++ exception", err, err_cap);
    return kCaughtUnknown;
  }
}

} // extern "C"
