
# 🎛️ Atom Combat — Match Orchestrator (Component)

## Purpose
The **Match Orchestrator** is the director of every fight.
It coordinates the flow between the Arena and both fighters, advancing time tick by tick, enforcing fairness and pacing.

> The Orchestrator is the heartbeat of the match.

---

## Responsibilities (Full Vision)
- Initialize the match (load fighters, Arena, World Spec, Protocol)
- Manage the tick loop at configured rate
- Generate and send snapshots to fighters
- Receive and validate actions
- Coordinate simultaneous application of actions to Arena
- Record all actions and events into the Telemetry system
- Detect end conditions and declare results

---

## Core Loop (Overview)
1. Generate snapshot for each fighter
2. Wait for each to respond within time budget
3. Apply actions to the Arena simultaneously
4. Log results and events
5. Repeat until win condition or time expires

---

## POC: Detailed Tick Sequence (1D)

### Initialization
```python
def initialize_match(match_spec):
    # Load world spec
    world = load_world_spec(match_spec.world_spec)

    # Load fighter specs
    fighter_a = load_fighter(match_spec.fighter_a_id)
    fighter_b = load_fighter(match_spec.fighter_b_id)

    # Initialize arena
    arena = Arena1D(
        width=world.arena.width,
        friction=world.physics.friction_coefficient,
        max_accel=world.physics.max_acceleration,
        max_velocity=world.physics.max_velocity,
        dt=world.physics.dt
    )

    # Set initial states
    arena.set_fighter_state(
        fighter_a,
        position=match_spec.fighter_a_start_pos,
        velocity=0.0,
        hp=fighter_a.stats.max_hp,
        stamina=fighter_a.stats.max_stamina,
        stance="neutral"
    )

    arena.set_fighter_state(
        fighter_b,
        position=match_spec.fighter_b_start_pos,
        velocity=0.0,
        hp=fighter_b.stats.max_hp,
        stamina=fighter_b.stats.max_stamina,
        stance="neutral"
    )

    # Initialize telemetry
    telemetry = TelemetryStore(match_spec.match_id)

    # Seed RNG (for any stochastic elements)
    random.seed(match_spec.seed)

    return arena, fighter_a, fighter_b, telemetry, world
```

### Main Tick Loop
```python
def run_match(arena, fighter_a, fighter_b, telemetry, world):
    tick = 0
    max_ticks = world.timing.round_duration_ticks

    while tick < max_ticks:
        # 1. Generate snapshots for both fighters
        snapshot_a = generate_snapshot(arena, fighter_a, "a", "b", world)
        snapshot_b = generate_snapshot(arena, fighter_b, "b", "a", world)

        # 2. Request actions from fighters (with timeout)
        action_a = request_action(fighter_a, snapshot_a, world.timing.decision_time_budget_ms)
        action_b = request_action(fighter_b, snapshot_b, world.timing.decision_time_budget_ms)

        # 3. Validate and sanitize actions
        action_a = validate_action(action_a, fighter_a, arena, world)
        action_b = validate_action(action_b, fighter_b, arena, world)

        # 4. Apply physics (SIMULTANEOUS)
        events = arena.step(action_a, action_b, world)

        # 5. Log to telemetry
        telemetry.record_tick(tick, snapshot_a, snapshot_b, action_a, action_b, events, arena.get_state())

        # 6. Check win conditions
        result = check_win_conditions(arena, tick, max_ticks)
        if result:
            telemetry.record_result(result)
            return result

        tick += 1

    # Timeout - judge by remaining HP
    return determine_winner_by_hp(arena)
```

### Snapshot Generation (with Sensor Filtering)
```python
def generate_snapshot(arena, my_fighter, my_id, opp_id, world):
    my_state = arena.get_fighter_state(my_id)
    opp_state = arena.get_fighter_state(opp_id)

    # Calculate opponent distance (absolute)
    distance = abs(opp_state.position - my_state.position)

    # Apply sensor precision (bucketing)
    distance_precision = my_fighter.sensors.distance_precision
    distance_bucketed = round(distance / distance_precision) * distance_precision

    velocity_precision = my_fighter.sensors.velocity_precision
    opp_velocity_bucketed = round(opp_state.velocity / velocity_precision) * velocity_precision

    # Apply stance detection filtering
    stance_hint = apply_stance_filter(
        opp_state.stance,
        my_fighter.sensors.stance_detection
    )

    return {
        "protocol_version": "proto_1d_poc_v1",
        "tick": arena.current_tick,
        "time_remaining_ticks": world.timing.round_duration_ticks - arena.current_tick,
        "you": {
            "position": my_state.position,
            "velocity": my_state.velocity,
            "hp": my_state.hp,
            "stamina": my_state.stamina,
            "stance": my_state.stance
        },
        "opponent": {
            "distance": distance_bucketed,
            "velocity": opp_velocity_bucketed,
            "stance_hint": stance_hint,
            "hp_visible": False  # POC: opponent HP hidden
        },
        "arena": {
            "width": world.arena.width,
            "your_distance_to_left_wall": my_state.position,
            "your_distance_to_right_wall": world.arena.width - my_state.position
        }
    }
```

### Action Request (with Timeout)
```python
def request_action(fighter, snapshot, timeout_ms):
    try:
        # Call fighter runtime with timeout
        action = call_with_timeout(
            fighter.runtime.decide,
            args=(snapshot,),
            timeout_ms=timeout_ms
        )
        return action
    except TimeoutError:
        # Late response - return safe default
        return {"acceleration": 0.0, "stance": "neutral"}
    except Exception as e:
        # Fighter crashed - return safe default
        log_error(f"Fighter {fighter.id} error: {e}")
        return {"acceleration": 0.0, "stance": "neutral"}
```

### Action Validation
```python
def validate_action(action, fighter, arena, world):
    # Clamp acceleration to legal range
    max_accel = world.physics.max_acceleration
    action.acceleration = clamp(action.acceleration, -max_accel, max_accel)

    # Validate stance
    valid_stances = ["neutral", "extended", "retracted", "defending"]
    if action.stance not in valid_stances:
        action.stance = "neutral"

    # Check stamina - if zero, force safe action
    fighter_state = arena.get_fighter_state(fighter.id)
    if fighter_state.stamina <= 0:
        action.acceleration = 0.0
        action.stance = "neutral"

    return action
```

---

## Determinism & Reproducibility
- Each match runs from a seed and config file (Match Spec)
- The Orchestrator's sequence defines the official truth of the fight
- **Simultaneous action application** ensures no turn-order bias
- Replays can re-run the exact same tick sequence with no variation

---

## Evolution Path: POC → Full Vision

| Aspect | POC (1D) | Full Vision (2D/3D) |
|--------|----------|---------------------|
| **Snapshot Gen** | Simple distance/velocity bucketing | Visual ray-casting, occlusion, sensor simulation |
| **Action Type** | acceleration + stance | Forces/positions for body parts |
| **Validation** | Range checks | Joint limits, force limits, balance checks |
| **Physics Step** | 1D position/velocity update | Full rigid body simulation |
| **Simultaneity** | Both actions applied atomically | Same principle, more complex state |

**What stays constant:**
- Tick-based loop
- Snapshot → Action → Physics → Log sequence
- Time-bounded decision making
- Deterministic execution

---

**One-liner summary:**
*The Match Orchestrator runs the tick-by-tick heartbeat of Atom Combat, ensuring fairness, simultaneity, and perfect reproducibility — starting with simple 1D coordination in the POC, evolving toward complex multi-body orchestration.*
