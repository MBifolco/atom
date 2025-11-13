# Movement Decoupling Architecture
## Separating Fighter Intent from Arena Physics

---

## Problem Statement

Currently, the fighter AI and arena physics are tightly coupled through the concept of "acceleration". Fighters must output acceleration values, and the arena directly applies these accelerations to update positions. This creates several limitations:

1. **Fighter Limitation**: All fighters must think in terms of acceleration
2. **Physics Lock-in**: Changing physics requires retraining all fighters
3. **Coaching Complexity**: Coaching must work at the acceleration level
4. **Innovation Barrier**: Novel movement strategies are difficult to implement

---

## Proposed Solution: Position-Based Interface

Decouple fighter decision-making from arena physics by having fighters express **desired positions** rather than accelerations. The arena then validates and enforces physics constraints.

### Current Architecture (Tightly Coupled)
```
Fighter → Acceleration → Arena applies physics → New Position
```

### New Architecture (Decoupled)
```
Fighter → Desired Position → Arena validates → Actual Position
```

---

## Technical Design

### 1. Core Interfaces

```python
from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

class MovementMethod(Enum):
    """How the fighter calculated their desired position."""
    ACCELERATION = "acceleration"      # Traditional physics-based
    TELEPORT = "teleport"             # Instant movement attempts
    PATHFINDING = "pathfinding"       # Calculated path
    PATTERN = "pattern"                # Rhythmic/pattern-based
    NEURAL = "neural"                  # Direct neural network output
    COACHED = "coached"                # Coaching override

@dataclass
class MovementRequest:
    """What the fighter wants to do next tick."""
    desired_position: float            # Where fighter wants to be
    movement_method: MovementMethod    # How they calculated it
    priority: float = 1.0              # For resolving conflicts
    metadata: Dict[str, Any] = None   # Additional info (e.g., raw acceleration)

@dataclass
class MovementResponse:
    """What actually happens after physics validation."""
    actual_position: float             # Where fighter ends up
    position_delta: float              # How far they actually moved
    stamina_cost: float                # Energy consumed
    was_capped: bool                   # Was movement limited by physics?
    cap_reason: Optional[str] = None  # Why movement was limited
    physics_state: Dict[str, Any] = None  # Additional physics info
```

### 2. Fighter Interface Evolution

```python
# Current Fighter Interface (acceleration-based)
class LegacyFighter:
    def decide(self, snapshot: Dict) -> Dict:
        """Returns acceleration and stance."""
        return {
            "acceleration": 2.5,
            "stance": "extended"
        }

# New Fighter Interface (position-based)
class ModernFighter:
    def decide(self, snapshot: Dict) -> Dict:
        """Returns desired position and stance."""
        return {
            "desired_position": snapshot["position"] + 0.5,
            "stance": "extended",
            "movement_method": MovementMethod.NEURAL
        }

# Adapter for backward compatibility
class LegacyFighterAdapter:
    """Wraps old fighters to work with new system."""

    def __init__(self, legacy_fighter):
        self.fighter = legacy_fighter
        self.dt = 0.1  # Physics timestep

    def decide(self, snapshot: Dict) -> Dict:
        # Get legacy decision
        legacy = self.fighter.decide(snapshot)

        # Convert acceleration to desired position
        desired_pos = snapshot["position"] + legacy["acceleration"] * self.dt

        return {
            "desired_position": desired_pos,
            "stance": legacy["stance"],
            "movement_method": MovementMethod.ACCELERATION,
            "metadata": {"raw_acceleration": legacy["acceleration"]}
        }
```

### 3. Arena Physics Validator

```python
class PhysicsValidator:
    """Enforces physics constraints on movement requests."""

    def __init__(self, physics_config):
        self.max_acceleration = physics_config["max_acceleration"]  # 4.5 m/s²
        self.dt = physics_config["timestep"]  # 0.1 seconds
        self.stamina_model = StaminaModel(physics_config)

    def validate_movement(self,
                         request: MovementRequest,
                         fighter_state: FighterState) -> MovementResponse:
        """
        Validate and potentially modify movement request based on physics.
        """

        # Calculate requested movement
        current_pos = fighter_state.position
        requested_delta = request.desired_position - current_pos
        requested_speed = abs(requested_delta) / self.dt

        # Check physics constraints
        max_speed = self._calculate_max_speed(fighter_state)
        max_delta = max_speed * self.dt

        # Determine actual movement
        if abs(requested_delta) <= max_delta:
            # Movement is within physics limits
            actual_position = request.desired_position
            actual_delta = requested_delta
            was_capped = False
            cap_reason = None
        else:
            # Movement exceeds physics limits - cap it
            direction = np.sign(requested_delta)
            actual_delta = direction * max_delta
            actual_position = current_pos + actual_delta
            was_capped = True
            cap_reason = self._determine_cap_reason(fighter_state, requested_speed)

        # Calculate stamina cost
        stamina_cost = self.stamina_model.calculate_cost(
            abs(actual_delta),
            fighter_state.stamina,
            fighter_state.stance
        )

        return MovementResponse(
            actual_position=actual_position,
            position_delta=actual_delta,
            stamina_cost=stamina_cost,
            was_capped=was_capped,
            cap_reason=cap_reason,
            physics_state={
                "velocity": actual_delta / self.dt,
                "acceleration": (actual_delta / self.dt - fighter_state.velocity) / self.dt
            }
        )

    def _calculate_max_speed(self, fighter_state: FighterState) -> float:
        """Calculate maximum allowed speed based on stamina and stance."""
        base_max_speed = self.max_acceleration * self.dt

        # Stamina affects max speed
        stamina_factor = fighter_state.stamina / 100.0
        stamina_factor = max(0.5, stamina_factor)  # Minimum 50% speed

        # Stance affects max speed
        stance_multiplier = {
            "defending": 0.7,
            "neutral": 1.0,
            "extended": 1.1,
            "retracted": 0.8
        }.get(fighter_state.stance, 1.0)

        return base_max_speed * stamina_factor * stance_multiplier

    def _determine_cap_reason(self, fighter_state: FighterState,
                              requested_speed: float) -> str:
        """Determine why movement was capped."""
        if fighter_state.stamina < 10:
            return "exhausted"
        elif requested_speed > self.max_acceleration * 2:
            return "impossible_acceleration"
        elif fighter_state.stance == "defending":
            return "defensive_stance_limit"
        else:
            return "physics_limit"
```

### 4. Integration with Match Orchestrator

```python
class MatchOrchestrator:
    """Orchestrates matches with new movement system."""

    def __init__(self, arena, fighter_a, fighter_b):
        self.arena = arena
        self.physics_validator = PhysicsValidator(arena.physics_config)

        # Wrap fighters if they're legacy
        self.fighter_a = self._wrap_if_legacy(fighter_a)
        self.fighter_b = self._wrap_if_legacy(fighter_b)

    def _wrap_if_legacy(self, fighter):
        """Wrap legacy fighters with adapter."""
        if hasattr(fighter.decide.__code__, 'co_varnames'):
            # Check if fighter returns acceleration (legacy)
            if 'acceleration' in str(fighter.decide.__code__):
                return LegacyFighterAdapter(fighter)
        return fighter

    def step(self):
        """Execute one tick of the match."""

        # Get movement requests from fighters
        request_a = self._get_movement_request(self.fighter_a, self.snapshot_a)
        request_b = self._get_movement_request(self.fighter_b, self.snapshot_b)

        # Validate movements through physics
        response_a = self.physics_validator.validate_movement(
            request_a, self.fighter_a_state
        )
        response_b = self.physics_validator.validate_movement(
            request_b, self.fighter_b_state
        )

        # Apply movements
        self.fighter_a_state.position = response_a.actual_position
        self.fighter_a_state.stamina -= response_a.stamina_cost

        self.fighter_b_state.position = response_b.actual_position
        self.fighter_b_state.stamina -= response_b.stamina_cost

        # Rest of combat logic (damage, etc.)
        self._apply_combat_resolution()
```

---

## Benefits of Decoupling

### 1. Fighter Innovation

Developers can create fighters with novel movement strategies:

```python
class TeleportFighter:
    """Fighter that attempts to teleport."""
    def decide(self, snapshot):
        if random.random() < 0.1:  # 10% chance
            # Attempt to teleport behind opponent
            return {
                "desired_position": snapshot["opponent_position"] - 2.0,
                "movement_method": MovementMethod.TELEPORT
            }
        return {"desired_position": snapshot["position"]}

class PatternFighter:
    """Fighter that moves in mathematical patterns."""
    def decide(self, snapshot):
        # Sinusoidal movement pattern
        tick = snapshot["tick"]
        amplitude = 2.0
        frequency = 0.1

        desired_pos = snapshot["center"] + amplitude * sin(tick * frequency)
        return {
            "desired_position": desired_pos,
            "movement_method": MovementMethod.PATTERN
        }

class PredictiveFighter:
    """Fighter that predicts where to be."""
    def decide(self, snapshot):
        # Predict opponent's next position
        predicted_opp_pos = self.predict_opponent(snapshot)

        # Calculate optimal intercept position
        optimal_pos = self.calculate_intercept(predicted_opp_pos)

        return {
            "desired_position": optimal_pos,
            "movement_method": MovementMethod.PATHFINDING
        }
```

### 2. Coaching Evolution

Coaching can work at higher abstraction levels:

```python
class StrategicCoach:
    """Coach that thinks in terms of positions, not accelerations."""

    def modify_decision(self, base_decision, game_state):
        if self.strategy == "maintain_distance":
            optimal_distance = 3.0
            current_distance = game_state["distance"]

            if current_distance < optimal_distance:
                # Too close - retreat to optimal distance
                base_decision["desired_position"] = (
                    game_state["opponent_position"] -
                    optimal_distance * sign(game_state["position"] -
                                           game_state["opponent_position"])
                )

        elif self.strategy == "corner_pressure":
            # Push opponent toward arena edge
            edge = 10.0 if game_state["opponent_position"] > 0 else -10.0
            base_decision["desired_position"] = (
                game_state["opponent_position"] +
                sign(edge - game_state["opponent_position"]) * 2.0
            )

        return base_decision
```

### 3. Physics Evolution

Different arenas can have different physics without breaking fighters:

```python
class LowGravityArena(PhysicsValidator):
    """Arena with reduced gravity - allows bigger jumps."""
    def _calculate_max_speed(self, fighter_state):
        return super()._calculate_max_speed(fighter_state) * 1.5

class HighFrictionArena(PhysicsValidator):
    """Arena with high friction - reduces movement."""
    def _calculate_max_speed(self, fighter_state):
        return super()._calculate_max_speed(fighter_state) * 0.7

class PortalArena(PhysicsValidator):
    """Arena with portals - allows instant travel between points."""
    def validate_movement(self, request, fighter_state):
        # Check if movement goes through a portal
        if self._crosses_portal(fighter_state.position, request.desired_position):
            # Allow instant travel through portal
            return MovementResponse(
                actual_position=self._portal_exit(request.desired_position),
                was_capped=False,
                stamina_cost=0  # Portals are free!
            )
        return super().validate_movement(request, fighter_state)
```

---

## Migration Plan

### Phase 1: Create Interfaces (Week 1)
- [ ] Define MovementRequest/Response classes
- [ ] Create PhysicsValidator base class
- [ ] Build LegacyFighterAdapter

### Phase 2: Update Arena (Week 2)
- [ ] Modify Arena to use PhysicsValidator
- [ ] Update MatchOrchestrator for new interface
- [ ] Ensure backward compatibility

### Phase 3: Test with Legacy Fighters (Week 3)
- [ ] Verify all existing fighters work through adapter
- [ ] Performance testing
- [ ] Bug fixes

### Phase 4: Create New Fighter Examples (Week 4)
- [ ] Build position-based fighter
- [ ] Create pattern-based fighter
- [ ] Demonstrate novel movement strategies

### Phase 5: Update Coaching System (Week 5)
- [ ] Modify coaching to work with positions
- [ ] Create position-based coaching strategies
- [ ] Test coaching with new movement system

---

## Long-term Vision

This decoupling enables:

1. **Multi-dimensional combat**: Same fighters work in 1D, 2D, 3D
2. **Custom physics**: Community-created arenas with unique rules
3. **Movement innovation**: Novel movement strategies we haven't imagined
4. **Clean coaching**: Coaching at intent level, not implementation level
5. **Platform extensibility**: Third-party developers can innovate freely

The movement interface becomes a contract that enables unlimited innovation while maintaining compatibility.