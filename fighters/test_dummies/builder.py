"""
Test Dummy Builder Framework

Provides factory classes and utilities for generating test fighters programmatically.
"""

from typing import Dict, Any, Callable, Optional
from pathlib import Path
import textwrap


class TestDummyBuilder:
    """Factory class for creating test dummy fighters."""

    def __init__(self, name: str = "Test Dummy"):
        self.name = name
        self.description = "Auto-generated test dummy"
        self.stance = "neutral"
        self.movement_type = "stationary"
        self.movement_params = {}
        self.conditions = []
        self.imports = set()

    def with_name(self, name: str) -> 'TestDummyBuilder':
        """Set the fighter name."""
        self.name = name
        return self

    def with_description(self, description: str) -> 'TestDummyBuilder':
        """Set the fighter description."""
        self.description = description
        return self

    def with_stance(self, stance: str) -> 'TestDummyBuilder':
        """Set the default stance (neutral, extended, retracted, defending)."""
        if stance not in ["neutral", "extended", "retracted", "defending"]:
            raise ValueError(f"Invalid stance: {stance}")
        self.stance = stance
        return self

    def stationary(self) -> 'TestDummyBuilder':
        """Make the fighter stationary."""
        self.movement_type = "stationary"
        return self

    def shuttle(self, left_bound: float = 3.0, right_bound: float = 9.0,
                speed: float = 2.0) -> 'TestDummyBuilder':
        """Make the fighter shuttle back and forth."""
        self.movement_type = "shuttle"
        self.movement_params = {
            "left_bound": left_bound,
            "right_bound": right_bound,
            "speed": speed
        }
        return self

    def approach_opponent(self, speed: float = 2.0) -> 'TestDummyBuilder':
        """Make the fighter always approach the opponent."""
        self.movement_type = "approach"
        self.movement_params = {"speed": speed}
        return self

    def flee_from_opponent(self, speed: float = 3.0) -> 'TestDummyBuilder':
        """Make the fighter always flee from the opponent."""
        self.movement_type = "flee"
        self.movement_params = {"speed": speed}
        return self

    def maintain_distance(self, target_distance: float = 3.0,
                         tolerance: float = 0.5) -> 'TestDummyBuilder':
        """Make the fighter maintain a specific distance from opponent."""
        self.movement_type = "maintain_distance"
        self.movement_params = {
            "target": target_distance,
            "tolerance": tolerance
        }
        return self

    def circle(self, direction: str = "left", speed: float = 2.0) -> 'TestDummyBuilder':
        """Make the fighter circle in one direction."""
        self.movement_type = "circle"
        self.movement_params = {
            "direction": direction,
            "speed": speed
        }
        return self

    def with_condition(self, condition: str, action: Dict[str, Any]) -> 'TestDummyBuilder':
        """Add a conditional behavior."""
        self.conditions.append({
            "condition": condition,
            "action": action
        })
        return self

    def build(self) -> str:
        """Generate the Python code for the fighter."""
        code = self._generate_header()
        code += self._generate_decide_function()
        return code

    def save(self, filepath: str) -> None:
        """Save the generated fighter to a file."""
        code = self.build()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code)

    def _generate_header(self) -> str:
        """Generate the file header with docstring."""
        return f'''"""
{self.name}

{self.description}

Auto-generated test dummy fighter.
Movement: {self.movement_type}
Default Stance: {self.stance}
"""


'''

    def _generate_decide_function(self) -> str:
        """Generate the decide function based on configuration."""
        body = self._generate_function_body()

        return f'''def decide(snapshot):
    """
    Test dummy fighter decision function.

    Movement type: {self.movement_type}
    Default stance: {self.stance}
    """
{body}
'''

    def _generate_function_body(self) -> str:
        """Generate the function body based on movement type."""

        # Extract common variables
        setup = '''    # Extract data from snapshot
    my_position = snapshot["you"]["position"]
    my_velocity = snapshot["you"]["velocity"]
    opp_position = snapshot["opponent"]["position"]
    distance = snapshot["opponent"]["distance"]
    arena_width = snapshot["arena"]["width"]
'''

        # Generate movement logic
        if self.movement_type == "stationary":
            movement_logic = f'''
    # Stationary behavior
    acceleration = 0.0
    stance = "{self.stance}"'''

        elif self.movement_type == "shuttle":
            left = self.movement_params["left_bound"]
            right = self.movement_params["right_bound"]
            speed = self.movement_params["speed"]
            movement_logic = f'''
    # Shuttle behavior
    left_bound = {left}
    right_bound = {right}

    if my_position <= left_bound:
        acceleration = {speed}  # Move right
    elif my_position >= right_bound:
        acceleration = -{speed}  # Move left
    elif my_velocity > 0:  # Moving right
        if my_position >= right_bound - 0.5:
            acceleration = -{speed}  # Start turning
        else:
            acceleration = {speed}  # Continue right
    else:  # Moving left
        if my_position <= left_bound + 0.5:
            acceleration = {speed}  # Start turning
        else:
            acceleration = -{speed}  # Continue left

    stance = "{self.stance}"'''

        elif self.movement_type == "approach":
            speed = self.movement_params["speed"]
            movement_logic = f'''
    # Approach opponent behavior
    if my_position < opp_position:
        acceleration = {speed}  # Move right toward opponent
    elif my_position > opp_position:
        acceleration = -{speed}  # Move left toward opponent
    else:
        acceleration = 0.0  # At opponent position

    stance = "{self.stance}"'''

        elif self.movement_type == "flee":
            speed = self.movement_params["speed"]
            movement_logic = f'''
    # Flee from opponent behavior
    if my_position < opp_position:
        acceleration = -{speed}  # Move left away from opponent
    elif my_position > opp_position:
        acceleration = {speed}  # Move right away from opponent
    else:
        acceleration = {speed}  # Emergency escape

    # Avoid walls
    if my_position < 2.0:
        acceleration = {speed}  # Don't flee into left wall
    elif my_position > arena_width - 2.0:
        acceleration = -{speed}  # Don't flee into right wall

    stance = "{self.stance}"'''

        elif self.movement_type == "maintain_distance":
            target = self.movement_params["target"]
            tolerance = self.movement_params["tolerance"]
            movement_logic = f'''
    # Maintain distance behavior
    target_distance = {target}
    tolerance = {tolerance}

    if distance > target_distance + tolerance:
        # Too far - approach
        if my_position < opp_position:
            acceleration = 2.0
        else:
            acceleration = -2.0
    elif distance < target_distance - tolerance:
        # Too close - retreat
        if my_position < opp_position:
            acceleration = -2.0
        else:
            acceleration = 2.0
    else:
        # Perfect distance - maintain
        acceleration = 0.0

    stance = "{self.stance}"'''

        elif self.movement_type == "circle":
            direction = self.movement_params["direction"]
            speed = self.movement_params["speed"]
            accel = speed if direction == "right" else -speed
            movement_logic = f'''
    # Circle {direction} behavior
    acceleration = {accel}

    # Basic wall avoidance
    if my_position < 1.0 and acceleration < 0:
        acceleration = {abs(speed)}  # Bounce off left wall
    elif my_position > arena_width - 1.0 and acceleration > 0:
        acceleration = -{abs(speed)}  # Bounce off right wall

    stance = "{self.stance}"'''

        else:
            movement_logic = f'''
    # Default behavior
    acceleration = 0.0
    stance = "{self.stance}"'''

        # Add conditions if any
        if self.conditions:
            condition_logic = "\n    # Conditional overrides\n"
            for cond in self.conditions:
                condition_logic += f'''    if {cond["condition"]}:
        acceleration = {cond["action"].get("acceleration", "acceleration")}
        stance = "{cond["action"].get("stance", self.stance)}"
'''
            movement_logic += condition_logic

        # Generate return statement
        return_stmt = '''
    return {"acceleration": acceleration, "stance": stance}'''

        return setup + movement_logic + return_stmt


class StationaryTemplate:
    """Quick template for stationary test dummies."""

    @staticmethod
    def create(name: str, stance: str = "neutral") -> TestDummyBuilder:
        return (TestDummyBuilder(name)
                .with_description(f"Stationary test dummy in {stance} stance")
                .with_stance(stance)
                .stationary())


class ShuttleTemplate:
    """Quick template for shuttling test dummies."""

    @staticmethod
    def create(name: str, speed: float = 2.0, stance: str = "neutral") -> TestDummyBuilder:
        return (TestDummyBuilder(name)
                .with_description(f"Shuttling test dummy at speed {speed}")
                .with_stance(stance)
                .shuttle(speed=speed))


class PursuitTemplate:
    """Quick template for pursuit/flee test dummies."""

    @staticmethod
    def create_pursuer(name: str, speed: float = 2.0, stance: str = "neutral") -> TestDummyBuilder:
        return (TestDummyBuilder(name)
                .with_description(f"Always approaches opponent at speed {speed}")
                .with_stance(stance)
                .approach_opponent(speed))

    @staticmethod
    def create_fleer(name: str, speed: float = 3.0, stance: str = "neutral") -> TestDummyBuilder:
        return (TestDummyBuilder(name)
                .with_description(f"Always flees from opponent at speed {speed}")
                .with_stance(stance)
                .flee_from_opponent(speed))


class DistanceTemplate:
    """Quick template for distance-maintaining test dummies."""

    @staticmethod
    def create(name: str, target_distance: float = 3.0, stance: str = "neutral") -> TestDummyBuilder:
        return (TestDummyBuilder(name)
                .with_description(f"Maintains {target_distance}m from opponent")
                .with_stance(stance)
                .maintain_distance(target_distance))