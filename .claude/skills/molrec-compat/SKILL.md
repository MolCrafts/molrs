---
name: molrec-compat
description: Evaluate whether a file format's data flows smoothly into molrec. Checks molrs internal conventions, structural fit, cross-format consistency, and user ergonomics.
argument-hint: "<format-name>"
user-invocable: true
---

Evaluate MolRec compatibility for format: **$ARGUMENTS**

## Trigger

`/molrec-compat <format-name>`

Examples:

- `/molrec-compat cube` — Gaussian Cube integration
- `/molrec-compat chgcar` — VASP CHGCAR integration
- `/molrec-compat lammps-dump` — LAMMPS dump integration

## Key principles

- **molrec is a universal container** — it defines structure (Frame, Grid, Collection, Observable), NOT field names. The question is "does the data fit naturally?", not "does it cover every molrec feature?"
- **molrs has internal naming conventions** — readers MUST map format-specific names to molrs standard names (e.g., always `element`, never `symbol`). Check the reader against molrs conventions, not molrec spec names.
- **Smooth flow, not perfect coverage** — a cube file doesn't need to fill `method` or `observables`. It just needs to land in `frame/atoms` + `frame/grids` without friction.

## Workflow

### Step 1 — Gather context (parallel)

Launch up to 3 Explore agents in parallel:

1. **Format reader/writer** — Read `molrs-io/src/<format>.rs`. Extract: column names used in Block inserts, grid key, grid array keys, metadata keys, SimBox handling.

2. **Cross-format comparison** — Read at least one other reader handling similar data (e.g., for volumetric formats compare cube vs chgcar). Same field inventory. Flag divergences.

3. **Existing molrec tests** — Read `molrec/src/molrec/tests/test_<format>.py`. Check: what's tested, what passes, what's missing. Also check Zarr roundtrip tests.

### Step 2 — Product Manager evaluation

Using the **product-manager** agent persona (`.claude/agents/product-manager.md`), evaluate:

1. **molrs convention compliance** — Does the reader use molrs internal names?

   | molrs standard | Common violations |
   |----------------|-------------------|
   | `element` | `symbol`, `type`, `atom_type` |
   | `x`, `y`, `z` | `pos_x`, `coord_x` |
   | `atomic_number` | `z`, `Z`, `atnum` |
   | `charge` (partial) | nuclear charge stored as `charge` |

2. **Structural fit** — Does the Frame map to molrec's tree?
   - `frame/atoms` ← Block with per-atom columns
   - `frame/grids/<name>` ← Grid with dim/origin/cell/pbc + arrays
   - `frame/box` ← SimBox (if appropriate)
   - `frame/meta` ← key-value metadata

3. **User ergonomics** — Can a scientist discover and use the data without reading source code?

4. **Cross-format consistency** — Behavioral surprises compared to similar readers?

5. **Roundtrip fidelity** — Zarr roundtrip and format roundtrip status.

### Step 3 — Generate report

Output format:

```markdown
# MolRec Compatibility Report: <Format>

## Verdict
<One sentence: YES / YES WITH ISSUES / NEEDS WORK>

## molrs Convention Check
| Field used | Expected | Status |
|------------|----------|--------|

## Structural Fit
<How the data maps to frame tree — what fits, what doesn't>

## Friction Points
### [P1] <title>
- **Severity**: HIGH / MEDIUM / LOW
- **What**: ...
- **Impact**: ...
- **Fix**: <file path + change>

## Cross-Format Consistency
<Table vs similar format>

## Roundtrip Status
- Zarr: PASS/FAIL
- Format: PASS/FAIL
```

### Step 4 — Action items

For each friction point, propose a concrete fix:

- **Reader fix** → file path (`molrs-io/src/<format>.rs`) + what to change
- **molrec spec gap** → what's missing in the structural model
- **Convention issue** → which molrs convention is violated and how to fix

Present to user for prioritization. Do NOT auto-implement.
