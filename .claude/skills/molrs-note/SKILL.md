---
name: molrs-note
description: Capture an evolving decision into .claude/NOTES.md, detect conflicts with CLAUDE.md, promote stable notes up, and sweep stale ones. Writes NOTES.md and (on promotion) CLAUDE.md.
argument-hint: "<decision-in-one-sentence> | sweep | promote <slug>"
user-invocable: true
---

# molrs-note

Read `CLAUDE.md` for molrs conventions before recording a decision.

This skill is the **memory phase** for `/molrs-impl`. Decisions start here;
once they've survived two implementation cycles without amendment, they get
promoted into `CLAUDE.md` and removed from `NOTES.md` so the index stays
short.

## Procedure

### Mode A — capture (default)

1. Parse the argument as the one-line decision.
2. Grep `CLAUDE.md` for the topic. If there is an existing rule, **diff** it
   against the new one and classify:
   - **compatible** — refines or narrows the CLAUDE.md rule → append as a
     sub-bullet under it in `NOTES.md`, status `provisional`.
   - **conflicting** — contradicts CLAUDE.md → STOP. Ask the user whether
     to override the CLAUDE.md rule or drop the note.
3. Grep `.claude/NOTES.md` for the same topic. If present, update the
   entry in place (bump date). Do not duplicate.
4. Append a new entry at the top of `NOTES.md` using the format in its
   header. Fill in `**Why:**` — never leave it blank; if the motivation is
   not obvious from the conversation, ask one clarifying question.

### Mode B — `sweep`

1. For every entry in `NOTES.md`:
   - If `Status: provisional` and date > 90 days old → tag `STALE`.
   - If `Status: hardening` and date > 30 days old → propose promotion.
   - If the topic now has a contradicting CLAUDE.md rule → tag `CONFLICT`.
2. Report a table grouped by tag. Do not modify files; the user runs
   `/molrs-note promote <slug>` or deletes manually.

### Mode C — `promote <slug>`

1. Find the matching entry in `NOTES.md`.
2. Choose the CLAUDE.md section it belongs in (Core Data Model, Critical
   Conventions, Feature Flags, etc.). If no section fits, ask.
3. Append the rule to that CLAUDE.md section in the same style as
   surrounding bullets.
4. Remove the entry from `NOTES.md` and record `promoted (→ CLAUDE.md
   §<section>)` in the commit message.

## Rules

- Never delete a CLAUDE.md rule as a side effect of capture. Conflicts
  require a separate, explicit user decision.
- Never promote a note younger than two implementation cycles (check
  `git log` for references to the slug).
- `NOTES.md` is an index of **decisions**, not a log of work done. Do not
  capture "finished task X"; capture "we settled on approach X because Y".

## Output

One line: `captured: <slug>` | `swept: N stale, M conflicts, K promotable` |
`promoted: <slug> → CLAUDE.md §<section>`.
