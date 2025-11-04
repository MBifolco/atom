"""
Atom Combat - World Constants

All physics constants and stance definitions.
Optimized for spectacle (150-sample parameter search).
"""

# ============================================================================
# PHYSICS CONSTANTS
# ============================================================================

ARENA_WIDTH = 12.4760  # meters (optimized for spectacle)
FRICTION = 0.3225
MAX_ACCELERATION = 4.3751  # m/s²
MAX_VELOCITY = 2.6696  # m/s
DT = 0.0842  # seconds per tick (~12 Hz)

# ============================================================================
# STAMINA ECONOMY
# ============================================================================

STAMINA_ACCEL_COST = 0.2192  # Cost per m/s² acceleration
STAMINA_BASE_REGEN = 0.0451  # Base regeneration per tick
STAMINA_NEUTRAL_BONUS = 2.0651  # Multiplier when in neutral stance

# ============================================================================
# DAMAGE SCALING
# ============================================================================

BASE_COLLISION_DAMAGE = 3.1096
VELOCITY_DAMAGE_SCALE = 0.3507
MASS_DAMAGE_SCALE = 0.3530

# ============================================================================
# FIGHTER STAT FORMULAS
# ============================================================================

MIN_MASS = 40.1071
MAX_MASS = 90.7961

HP_MIN = 47.9535
HP_MAX = 125.4919
STAMINA_MAX = 12.3595
STAMINA_MIN = 5.7635

# ============================================================================
# STANCES
# ============================================================================

STANCES = {
    "neutral": {
        "reach": 0.2768,
        "width": 0.4428,
        "drain": 0.0001,
        "defense": 1.0612
    },
    "extended": {
        "reach": 0.8189,
        "width": 0.1681,
        "drain": 0.0324,
        "defense": 0.8872
    },
    "retracted": {
        "reach": 0.1005,
        "width": 0.1185,
        "drain": 0.0139,
        "defense": 1.1542
    },
    "defending": {
        "reach": 0.3811,
        "width": 0.5421,
        "drain": 0.0611,
        "defense": 1.6290
    }
}
