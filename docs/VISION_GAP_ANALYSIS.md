# Atom Combat - Vision Gap Analysis

**Date:** 2025-11-03
**Status:** Post-POC Component Build

This document compares what we've built against the original vision documents in `mds/`.

---

## ✅ What We Built Successfully

### Core Components (All Working)

| Component | Status | Implementation |
|-----------|--------|----------------|
| **Arena** | ✅ Complete | `src/arena/` - Physics engine with WorldConfig |
| **Combat Protocol** | ✅ Complete | `src/protocol/` - Snapshot, Action, ProtocolValidator |
| **Match Orchestrator** | ✅ Complete | `src/orchestrator/` - Tick loop coordination |
| **Fighter Runtime** | ✅ Complete | `src/ai/` - Tactical AI decision functions |
| **Telemetry & Replay Store** | ✅ Complete | `src/telemetry/` - Save/load compressed replays |
| **Evaluator** | ✅ Complete | `src/evaluator/` - Spectacle scoring (7 metrics) |
| **Replay Renderer** | ✅ Complete | `src/renderer/` - ASCII + HTML5 animations |

**Achievement:** Core runtime loop works perfectly!
- Fighters decide → Orchestrator executes → Arena simulates → Telemetry records → Evaluator scores → Renderer visualizes

---

## ❌ Critical Gaps (High Priority)

### 1. **User Surface / CLI** ✅ FIXED!

**Vision:**
```bash
# Simple commands to run fights
atom fight rusher.py tank.py --render html
```

**Current Reality:**
```bash
# NOW WORKS! Simple one-command fights:
python atom_fight.py fighters/rusher.py fighters/tank.py --html replay.html

# Full featured:
python atom_fight.py fighters/rusher.py fighters/tank.py \
    --mass-a 65 --mass-b 80 \
    --watch \
    --html replay.html \
    --save telemetry.json.gz
```

**Status:** ✅ **COMPLETE!**

**What We Built:**
- ✅ CLI fight runner (`atom_fight.py`)
- ✅ Fighter file loading system (Python files with `decide` function)
- ✅ Default configurations (sensible defaults for all params)
- ✅ Quick replay generation (--html flag)
- ✅ Example fighters (`fighters/rusher.py`, `fighters/tank.py`, `fighters/balanced.py`)

**Remaining (lower priority):**
- Tournament runner
- Create-fighter wizard
- Division management

---

### 2. **Fighter Spec & Artifact System** 🔴 MISSING IDENTITY

**Vision:**
```json
{
  "id": "rusher_v1",
  "name": "Rusher",
  "version": "1.0",
  "creator": "alice",
  "tags": ["aggressive", "lightweight"],
  "mass": 50.0,
  "stats": {"max_hp": 80, "max_stamina": 12.0},
  "sensors": {
    "distance_precision": 0.5,
    "velocity_precision": 0.5,
    "stance_detection": "fuzzy"
  },
  "runtime": {
    "type": "python_function",
    "entrypoint": "decide",
    "code_hash": "sha256:abc123..."
  }
}
```

**Current Reality:**
```python
# Just bare Python functions
def tactical_aggressive(snapshot):
    # ... no metadata, no identity
```

**Impact:**
- No fighter identity/provenance
- No versioning
- No certification
- Can't track lineage
- No sensor system!

**What's Needed:**
- Fighter spec dataclass
- Fighter artifact packaging
- Metadata storage
- Code hash verification

---

### 3. **Sensor System** 🟡 AFFECTS FAIRNESS

**Vision:**
Fighters have varying sensor quality as a strategic tradeoff:
```python
"sensors": {
  "distance_precision": 0.5,     # See opponent at ±0.5m accuracy
  "velocity_precision": 0.3,     # ±0.3 m/s accuracy
  "stance_detection": "fuzzy"    # Only see attacking/defending/neutral
}
```

**Current Reality:**
All fighters see exact values! No precision limits, no sensor tradeoffs.

**Impact:**
- Missing strategic dimension
- No tradeoff between precision vs other stats
- All fighters have "perfect vision"

**What's Needed:**
- Sensor precision in fighter spec
- Snapshot bucketing/rounding based on precision
- Fuzzy stance detection
- Test different sensor configs

---

### 4. **Registry** 🟡 NO CATALOG

**Vision:**
Official catalog of certified fighters, world specs, and seasons.

```python
registry.register_fighter(
    fighter_artifact,
    division="poc",
    creator="alice"
)

fighters = registry.list_fighters(division="poc", certified=True)
```

**Current Reality:**
No registry at all. Fighters exist as loose Python files.

**Impact:**
- No fighter catalog
- No certification system
- No tournament organization
- No versioning/deprecation

**What's Needed:**
- Fighter registry database
- Certification workflow
- Division management
- Version tracking

---

### 5. **Governance & Sandbox** 🟡 NO COMPLIANCE

**Vision:**
- 50ms time limit enforcement
- Anti-cheat verification
- Protocol compliance checks
- Code sandboxing

**Current Reality:**
- No time limits
- No sandboxing
- No compliance checks
- Fighters run with full Python access

**Impact:**
- Fighters could cheat (access global state, run forever)
- No reproducibility guarantees
- Security risks

**What's Needed:**
- Timeout enforcement
- Sandboxed execution
- Compliance validator
- Resource limits

---

## 🔶 Design Divergences (Intentional Changes)

### 1. **World-Calculated Stats** ✅ IMPROVEMENT

**Vision:** Fighter spec defines `max_hp` and `max_stamina` directly.

**Current:** World calculates HP/stamina from mass using formulas:
```python
hp = hp_min + (mass - min_mass) * (hp_max - hp_min) / (max_mass - min_mass)
stamina = stamina_max - (mass - min_mass) * (stamina_max - stamina_min) / (...)
```

**Why Different:** Prevents "perfect fighter" with max mass + max HP + max stamina.

**Assessment:** ✅ Better for balance! Keep this change.

---

## 🟢 Minor Gaps (Lower Priority)

### 6. **Determinism/Seeds** 🟢

**Vision:** Match seed for reproducibility, passed to fighters.

**Current:** Seed used in Arena but not exposed to fighters.

**Impact:** Mostly works, but fighters can't use controlled randomness.

---

### 7. **Match Spec Artifact** 🟢

**Vision:** Formal match specification with arena, ruleset, participants, seed.

**Current:** Match params passed loosely to orchestrator.

**Impact:** Works but not formalized as an artifact.

---

### 8. **Score Report Artifact** 🟢

**Vision:** Formal score report from evaluator.

**Current:** Returns SpectacleScore dataclass (good enough).

**Impact:** Mostly fine, just needs to be formalized.

---

## 📊 Priority Ranking

### Must Fix (Blocking Core Use Cases)

1. 🔴 **User Surface/CLI** - Can't easily run fights!
2. 🔴 **Fighter Spec/Artifact** - No identity or metadata
3. 🟡 **Sensor System** - Missing strategic dimension

### Should Fix (Important for Vision)

4. 🟡 **Registry** - Needed for tournaments
5. 🟡 **Governance/Sandbox** - Needed for fairness

### Nice to Have (Can Wait)

6. 🟢 **Determinism improvements**
7. 🟢 **Formal artifact types**
8. 🟢 **Time limit enforcement**

---

## 🎯 Recommended Next Steps

### Phase 1: Make It Usable ✅ DONE!

**Goal:** Simple CLI to run fights

```bash
# NOW WORKS!
python atom_fight.py fighters/rusher.py fighters/tank.py --html replay.html
```

**Tasks:**
- [x] Create simple fight runner CLI
- [x] Fighter file loading system
- [x] Default config presets
- [x] One-command replay generation

**Files Created:**
- `atom_fight.py` - Main CLI entry point ✅
- `fighters/rusher.py` - Aggressive fighter example ✅
- `fighters/tank.py` - Defensive fighter example ✅
- `fighters/balanced.py` - Adaptive fighter example ✅
- `README.md` - Complete usage guide ✅

---

### Phase 2: Fighter Identity

**Goal:** Proper fighter specs with metadata

**Tasks:**
- [ ] FighterSpec dataclass
- [ ] FighterArtifact packaging
- [ ] Load/save fighter specs
- [ ] Code hash generation

**Files to Create:**
- `src/fighter/spec.py` - FighterSpec, FighterArtifact
- `src/fighter/loader.py` - Load from disk
- `fighters/rusher.json` - Example spec file

---

### Phase 3: Sensors

**Goal:** Implement sensor precision system

**Tasks:**
- [ ] Add sensors to FighterSpec
- [ ] Bucket/round snapshots by precision
- [ ] Fuzzy stance detection
- [ ] Test sensor tradeoffs

**Files to Update:**
- `src/protocol/combat_protocol.py` - Add sensor filtering
- `src/fighter/spec.py` - Add sensor config

---

### Phase 4: Registry

**Goal:** Fighter catalog and certification

**Tasks:**
- [ ] Registry database (SQLite)
- [ ] Register/list/certify commands
- [ ] Version tracking
- [ ] Division management

**Files to Create:**
- `src/registry/registry.py` - Main registry
- `src/registry/db.py` - Database layer

---

## 📝 Summary

**What Works Well:**
- ✅ Core simulation loop is excellent
- ✅ Component architecture is clean
- ✅ Config system works great
- ✅ Replay system is powerful
- ✅ HTML renderer is beautiful
- ✅ Spectacle evaluation is sophisticated

**Critical Issues:**
- ✅ ~~No simple way to run fights~~ **FIXED!** Simple CLI works!
- 🔴 No fighter identity/metadata system
- 🟡 No sensor precision (missing strategic depth)
- 🟡 No registry/certification
- 🟡 No governance/sandboxing

**Bottom Line:**
We built a **fantastic simulation engine** with a **simple CLI** that works! ✅

**What's Left:**
- Fighter identity/metadata system (specs, artifacts, versioning)
- Sensor precision (strategic depth)
- Registry & certification (tournament infrastructure)

The core is solid and now it's **easy to use!** 🎉
