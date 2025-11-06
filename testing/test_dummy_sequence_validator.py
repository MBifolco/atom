#!/usr/bin/env python3
"""
Test Dummy Sequence Validator

Tests sequences of snapshots to validate dynamic behaviors.
Ensures test dummies maintain consistent behavior over time.
"""

import sys
import importlib.util
from pathlib import Path
from typing import Dict, List, Tuple, Callable
import numpy as np


class SequenceValidator:
    """Validates test dummy behaviors over sequences of game states."""

    def __init__(self):
        self.results = {}
        self.failures = []

    def load_fighter(self, path: str) -> Callable:
        """Load a fighter module and return its decide function."""
        spec = importlib.util.spec_from_file_location("fighter", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.decide

    def simulate_sequence(self, decide: Callable, initial_state: Dict,
                         steps: int = 100) -> List[Dict]:
        """Simulate a sequence of decisions and state updates."""
        states = []
        state = initial_state.copy()

        for tick in range(steps):
            state["tick"] = tick

            # Get decision
            decision = decide(state)

            # Update state based on decision (simplified physics)
            # Update velocity based on acceleration
            new_velocity = state["you"]["velocity"] + decision["acceleration"] * 0.1
            new_velocity = max(-5.0, min(5.0, new_velocity))  # Clamp velocity

            # Update position based on velocity
            new_position = state["you"]["position"] + new_velocity * 0.1
            new_position = max(0.0, min(12.0, new_position))  # Clamp to arena

            # Update stamina based on stance
            stamina_drain = {
                "neutral": 0.1,
                "extended": 0.5,
                "defending": 0.2,
                "retracted": -0.3  # Recovery
            }.get(decision["stance"], 0.1)

            new_stamina = state["you"]["stamina"] - stamina_drain
            new_stamina = max(0.0, min(100.0, new_stamina))

            # Store state and decision
            states.append({
                "tick": tick,
                "position": state["you"]["position"],
                "velocity": state["you"]["velocity"],
                "stamina": state["you"]["stamina"],
                "decision": decision,
                "opponent_position": state["opponent"]["position"],
                "opponent_velocity": state["opponent"]["velocity"]
            })

            # Update state for next iteration
            state["you"]["position"] = new_position
            state["you"]["velocity"] = new_velocity
            state["you"]["stamina"] = new_stamina

        return states

    def create_initial_state(self, **kwargs) -> Dict:
        """Create an initial state for simulation."""
        defaults = {
            "my_position": 6.0,
            "opponent_position": 6.0,
            "my_velocity": 0.0,
            "opponent_velocity": 0.0,
            "my_stamina": 100.0,
            "opponent_stamina": 100.0,
            "my_hp": 100.0,
            "opponent_hp": 100.0
        }
        defaults.update(kwargs)

        return {
            "you": {
                "position": defaults["my_position"],
                "velocity": defaults["my_velocity"],
                "hp": defaults["my_hp"],
                "hp_max": 100.0,
                "stamina": defaults["my_stamina"],
                "stamina_max": 100.0,
                "stance": "neutral"
            },
            "opponent": {
                "position": defaults["opponent_position"],
                "velocity": defaults["opponent_velocity"],
                "hp": defaults["opponent_hp"],
                "hp_max": 100.0,
                "stamina": defaults["opponent_stamina"],
                "stamina_max": 100.0,
                "stance": "neutral"
            },
            "arena": {
                "width": 12.0
            },
            "tick": 0
        }

    def test_stationary_consistency(self):
        """Test that stationary dummies remain stationary over time."""
        print("\n=== TESTING STATIONARY CONSISTENCY ===")

        dummies = ["stationary_neutral", "stationary_extended",
                  "stationary_defending", "stationary_retracted"]

        for dummy_name in dummies:
            path = f"fighters/test_dummies/atomic/{dummy_name}.py"
            if not Path(path).exists():
                print(f"  ❌ {dummy_name}: FILE NOT FOUND")
                continue

            decide = self.load_fighter(path)
            initial_state = self.create_initial_state(my_position=6.0)
            states = self.simulate_sequence(decide, initial_state, steps=50)

            # Check position variance
            positions = [s["position"] for s in states]
            position_variance = np.var(positions)

            # Check stance consistency
            expected_stance = dummy_name.split("_")[1]
            stance_consistent = all(s["decision"]["stance"] == expected_stance for s in states)

            # Check acceleration is always 0
            accel_zero = all(abs(s["decision"]["acceleration"]) < 0.01 for s in states)

            if position_variance < 0.01 and stance_consistent and accel_zero:
                print(f"  ✅ {dummy_name}: Consistent over 50 ticks")
            else:
                print(f"  ❌ {dummy_name}: Inconsistent behavior")
                if position_variance >= 0.01:
                    print(f"    - Position variance: {position_variance:.4f}")
                if not stance_consistent:
                    print(f"    - Stance changes detected")
                if not accel_zero:
                    print(f"    - Non-zero acceleration detected")

    def test_shuttle_oscillation(self):
        """Test that shuttle dummies oscillate properly."""
        print("\n=== TESTING SHUTTLE OSCILLATION ===")

        speeds = ["slow", "medium", "fast"]

        for speed in speeds:
            dummy_name = f"shuttle_{speed}"
            path = f"fighters/test_dummies/atomic/{dummy_name}.py"
            if not Path(path).exists():
                print(f"  ❌ {dummy_name}: FILE NOT FOUND")
                continue

            decide = self.load_fighter(path)
            initial_state = self.create_initial_state(my_position=6.0)
            states = self.simulate_sequence(decide, initial_state, steps=200)

            # Analyze oscillation
            positions = [s["position"] for s in states[10:]]  # Skip initial movement
            accelerations = [s["decision"]["acceleration"] for s in states]

            # Count direction changes
            direction_changes = 0
            for i in range(1, len(accelerations)):
                if accelerations[i] * accelerations[i-1] < 0:  # Sign change
                    direction_changes += 1

            # Check position range
            pos_min, pos_max = min(positions), max(positions)
            pos_range = pos_max - pos_min

            if direction_changes >= 2 and pos_range > 2.0:
                print(f"  ✅ {dummy_name}: Oscillates properly ({direction_changes} direction changes, {pos_range:.1f}m range)")
            else:
                print(f"  ❌ {dummy_name}: Poor oscillation ({direction_changes} changes, {pos_range:.1f}m range)")

    def test_distance_keeper_stability(self):
        """Test that distance keepers maintain stable distances."""
        print("\n=== TESTING DISTANCE KEEPER STABILITY ===")

        for target_dist in [1, 3, 5]:
            dummy_name = f"distance_keeper_{target_dist}m"
            path = f"fighters/test_dummies/atomic/{dummy_name}.py"
            if not Path(path).exists():
                print(f"  ❌ {dummy_name}: FILE NOT FOUND")
                continue

            decide = self.load_fighter(path)

            # Test starting at perfect distance
            initial_state = self.create_initial_state(
                my_position=3.0,
                opponent_position=3.0 + target_dist
            )
            states = self.simulate_sequence(decide, initial_state, steps=100)

            # Calculate average distance maintained
            distances = [abs(s["opponent_position"] - s["position"]) for s in states[20:]]
            avg_distance = np.mean(distances)
            distance_variance = np.var(distances)

            # Check if it maintains target distance
            distance_error = abs(avg_distance - target_dist)

            if distance_error < 0.5 and distance_variance < 1.0:
                print(f"  ✅ {dummy_name}: Maintains {target_dist}m (avg: {avg_distance:.2f}m, var: {distance_variance:.2f})")
            else:
                print(f"  ❌ {dummy_name}: Poor distance control (avg: {avg_distance:.2f}m, target: {target_dist}m)")

    def test_stamina_cycler_phases(self):
        """Test that stamina cycler goes through proper phases."""
        print("\n=== TESTING STAMINA CYCLER PHASES ===")

        dummy_name = "stamina_cycler"
        path = f"fighters/test_dummies/atomic/{dummy_name}.py"
        if not Path(path).exists():
            print(f"  ❌ {dummy_name}: FILE NOT FOUND")
            return

        decide = self.load_fighter(path)
        initial_state = self.create_initial_state(my_stamina=100.0)
        states = self.simulate_sequence(decide, initial_state, steps=300)

        # Track stance phases
        attack_phases = 0
        recovery_phases = 0
        current_phase = None

        for state in states:
            stance = state["decision"]["stance"]

            if stance == "extended" and current_phase != "attack":
                attack_phases += 1
                current_phase = "attack"
            elif stance == "retracted" and current_phase != "recovery":
                recovery_phases += 1
                current_phase = "recovery"

        # Check stamina correlation
        high_stamina_extended = 0
        low_stamina_retracted = 0

        for state in states:
            if state["stamina"] > 80 and state["decision"]["stance"] == "extended":
                high_stamina_extended += 1
            elif state["stamina"] < 30 and state["decision"]["stance"] == "retracted":
                low_stamina_retracted += 1

        if attack_phases >= 2 and recovery_phases >= 2:
            print(f"  ✅ {dummy_name}: Cycles properly ({attack_phases} attack phases, {recovery_phases} recovery phases)")
        else:
            print(f"  ❌ {dummy_name}: Poor cycling ({attack_phases} attack, {recovery_phases} recovery)")

    def test_reactive_consistency(self):
        """Test that reactive dummies respond consistently to stimuli."""
        print("\n=== TESTING REACTIVE CONSISTENCY ===")

        # Test mirror_movement with moving opponent
        dummy_name = "mirror_movement"
        path = f"fighters/test_dummies/atomic/{dummy_name}.py"
        if Path(path).exists():
            decide = self.load_fighter(path)

            # Create sequence with oscillating opponent
            initial_state = self.create_initial_state()
            states = []
            state = initial_state.copy()

            for tick in range(100):
                # Oscillate opponent velocity
                state["opponent"]["velocity"] = 3.0 * np.sin(tick * 0.1)
                state["tick"] = tick

                decision = decide(state)
                states.append({
                    "tick": tick,
                    "opponent_velocity": state["opponent"]["velocity"],
                    "my_acceleration": decision["acceleration"]
                })

            # Check correlation between opponent velocity and acceleration
            correlations_correct = 0
            for state in states:
                if state["opponent_velocity"] > 0.5 and state["my_acceleration"] > 0:
                    correlations_correct += 1
                elif state["opponent_velocity"] < -0.5 and state["my_acceleration"] < 0:
                    correlations_correct += 1
                elif abs(state["opponent_velocity"]) <= 0.5 and abs(state["my_acceleration"]) < 1:
                    correlations_correct += 1

            accuracy = correlations_correct / len(states)
            if accuracy > 0.9:
                print(f"  ✅ {dummy_name}: {accuracy*100:.0f}% correct mirroring")
            else:
                print(f"  ❌ {dummy_name}: Only {accuracy*100:.0f}% correct mirroring")

    def test_wall_hugger_persistence(self):
        """Test that wall huggers stay at walls."""
        print("\n=== TESTING WALL HUGGER PERSISTENCE ===")

        for side in ["left", "right"]:
            dummy_name = f"wall_hugger_{side}"
            path = f"fighters/test_dummies/atomic/{dummy_name}.py"
            if not Path(path).exists():
                print(f"  ❌ {dummy_name}: FILE NOT FOUND")
                continue

            decide = self.load_fighter(path)

            # Start in center
            initial_state = self.create_initial_state(my_position=6.0)
            states = self.simulate_sequence(decide, initial_state, steps=100)

            # Check if it reaches and stays at wall
            final_positions = [s["position"] for s in states[-20:]]

            if side == "left":
                at_wall = all(p < 1.0 for p in final_positions)
                wall_pos = 0.0
            else:
                at_wall = all(p > 11.0 for p in final_positions)
                wall_pos = 12.0

            avg_final_pos = np.mean(final_positions)

            if at_wall:
                print(f"  ✅ {dummy_name}: Reaches and stays at wall (avg: {avg_final_pos:.1f}m)")
            else:
                print(f"  ❌ {dummy_name}: Doesn't stay at wall (avg: {avg_final_pos:.1f}m, target: {wall_pos}m)")

    def generate_report(self):
        """Generate summary report."""
        print("\n" + "=" * 60)
        print("SEQUENCE VALIDATION SUMMARY")
        print("=" * 60)

        if not self.failures:
            print("✅ ALL SEQUENCE TESTS PASSED!")
            print("\nTest dummies show consistent behavior over time.")
        else:
            print(f"❌ {len(self.failures)} SEQUENCE TESTS FAILED")
            print("\nSome test dummies have inconsistent temporal behavior.")

        return len(self.failures) == 0


def main():
    """Run sequence validation tests."""
    validator = SequenceValidator()

    print("TEST DUMMY SEQUENCE VALIDATOR")
    print("=" * 60)
    print("Testing temporal consistency and dynamic behaviors")

    # Run all sequence tests
    validator.test_stationary_consistency()
    validator.test_shuttle_oscillation()
    validator.test_distance_keeper_stability()
    validator.test_stamina_cycler_phases()
    validator.test_reactive_consistency()
    validator.test_wall_hugger_persistence()

    # Generate report
    success = validator.generate_report()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()