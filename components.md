# 🏗️ Atom Combat — High-Level Components (Core Only)

> **Scope:** This document lists only the **common, league-agnostic components** that every division uses during sanctioned fights.  
> Components for **building/training/coaching** fighters are valuable ecosystem add-ons, but **not** part of the core runtime.

---

## 1) System at a glance (core runtime)


**Principle:** The **Arena** is the source of truth. The **Protocol** is the shared contract.  
Fighters bring their own “minds,” but inside the ring they **all** interact through the same loop.

---

Fighter A (mind) Fighter B (mind)
│ │
▼ ▼
┌─────── Match Orchestrator (tick-by-tick) ───────┐
│ share snapshot → get actions → apply → log │
└───────────────→ Arena (rules & physics) ←────────
│
▼
Telemetry & Replays
│
├─ Evaluator (scoring)
├─ Governance (fairness & compliance)
└─ Replay Renderer (video) + UI

---

## 2) Core components & responsibilities

### A) **Arena (Rules & Physics)**
- **Role:** The stage and the referee.
- **Responsible for:** movement, collisions, damage, timers, win conditions, round flow.
- **Promises:** deterministic outcomes from seed + move history; identical behavior for all fighters.

### B) **Combat Protocol (The Contract)**
- **Role:** The shared language between fighters and the Arena.
- **Responsible for:** what fighters can *sense* each moment, what *actions* are valid, decision timing/budgets.
- **Promises:** fairness and compatibility regardless of how a fighter was built or trained.

### C) **Match Orchestrator (The Director)**
- **Role:** The heartbeat of the fight.
- **Responsible for:** per-tick scheduling, delivering snapshots to fighters, enforcing legality and time limits, applying actions, emitting events.
- **Output:** the complete replay timeline (snapshots, actions, outcomes).

### D) **Fighter Runtime (The Mind in the Ring)**
- **Role:** Make one decision per tick.
- **Responsible for:** reading the snapshot, deciding on an action within the time budget, returning it on time.
- **Note:** No learning occurs here; this is match-time decision only.

### E) **Telemetry & Replay Store (The Record Keeper)**
- **Role:** Preserve truth and history.
- **Responsible for:** per-tick logs, events, summaries, immutable replays; retrieval for scoring and rendering.

### F) **Evaluator (The Judge)**
- **Role:** Score results and style (division-specific metrics).
- **Responsible for:** win/loss, ties, basic style/quality metrics, rule-violation checks.
- **Output:** official score report and rankings input.

### G) **Governance & Sandbox (The Commission)**
- **Role:** Keep the sport fair and reproducible.
- **Responsible for:** runtime/time budgets, protocol compliance, anti-cheat, provenance, versioning of rules and seasons.
- **Actions:** certify fighters, flag violations, apply or announce season patches.

### H) **Replay Renderer (The Film Crew)**
- **Role:** Turn replays into watchable rounds.
- **Responsible for:** camera beats, overlays, effects; export to MP4/Web.
- **Constraint:** visuals never affect outcomes; they re-tell the recorded truth.

### I) **User Surface (CLI / Web)**
- **Role:** The front door during sanctioned play.
- **Responsible for:** queue fights, select arenas/divisions, view results, browse replays, read reports.

### J) **Registry (Fighters, Arenas, Seasons)**
- **Role:** Canonical catalog of what’s eligible.
- **Responsible for:** versions, metadata, eligibility per division, deprecations, audit trail.

---

## 3) Key artifacts (core)

- **Fighter Spec:** what a fighter *is* in the ring (body, stats, sensors, constraints).
- **Fighter Artifact:** the packaged "mind" (spec + runtime logic) used at match time.
- **Match Spec:** arena, ruleset, participants, seed.
- **Replay:** authoritative timeline of snapshots, actions, and events.
- **Score Report:** evaluator's official outcome + metrics.
- **Season Patch:** division-wide adjustments between seasons (never mid-match).

---

## 4) Lifecycles (core loops only)

### a) **Competition loop**
Create fight → Run (tick-by-tick) → Record → Judge → Publish results & replay.

### b) **Spectatorship loop**
Replay → Render → Share/Analyze (no impact on recorded truth).

> Any model improvement or coaching happens **outside** these loops.

---

## 5) What’s explicitly **outside** the core (but part of the ecosystem)

> These enable casual and non-pro participation, but pros may replace them with their own pipelines.

- **Training & Tuning (“The Gym”)** — improving fighters between matches.
- **Coach (Language → Plan)** — translating human intent into safe, bounded tweaks.

They are **extensions**, not requirements, and are not involved during sanctioned fights.

---

## 6) Versioning & longevity (core stance)

- **Protocol versions** define what can be sensed, what actions are valid, and timing budgets in a division.
- **Replays** remain reproducible forever within their protocol version.
- **Patches** are applied between seasons, with clear notes and migration paths.

---

**One-liner recap (core):**  
*The Arena is the truth, the Protocol is the contract, the Orchestrator runs the loop, Fighters decide within the rules, and everything is logged, judged, and replayable.*
