"""
JAX-compatible opponent decision functions for vmap training.

These are JAX/JIT-compatible versions of the test dummy opponents,
allowing different opponents to be used across vmapped environments.
"""

import jax
import jax.numpy as jnp
from jax import lax


def stationary_neutral_jax(state, config):
    """Stationary neutral stance - just stands still."""
    return jnp.array([0.0, 0])  # [acceleration, stance_int]


def stationary_extended_jax(state, config):
    """Stationary extended stance - stands still in extended."""
    return jnp.array([0.0, 1])  # [acceleration, stance_int]


def stationary_defending_jax(state, config):
    """Stationary defending stance - stands still in defending."""
    return jnp.array([0.0, 2])  # [acceleration, stance_int=2 for defending]


def approach_slow_jax(state, config):
    """Slow approach - moves toward opponent at speed 1.5."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    # At overlap (same position), default to moving left (matching Python else branch)
    direction = jnp.where(opp_pos > my_pos, 1.0, -1.0)

    accel = 1.5 * direction
    return jnp.array([accel, 0])  # neutral stance


def flee_always_jax(state, config):
    """Always flees from opponent."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    arena_width = config.arena_width
    direction = jnp.sign(opp_pos - my_pos)

    # Flee away from opponent; at overlap (direction==0), default to fleeing right
    flee_accel = jnp.where(direction > 0, -1.5, 1.5)
    # Wall bounce: don't flee into walls
    accel = jnp.where(my_pos < 1.0, 1.5,
            jnp.where(my_pos > arena_width - 1.0, -1.5,
                       flee_accel))
    return jnp.array([accel, 0])  # neutral stance


def circle_left_jax(state, config):
    """Always circles left (constant -2.0 accel)."""
    my_pos = state.fighter_b.position

    # Bounce off left wall
    accel = lax.cond(
        my_pos < 1.0,
        lambda _: 2.0,  # Bounce right
        lambda _: -2.0,  # Default left
        None
    )

    return jnp.array([accel, 0])  # neutral stance


def circle_right_jax(state, config):
    """Always circles right (constant 2.0 accel)."""
    my_pos = state.fighter_b.position
    arena_width = config.arena_width

    # Bounce off right wall
    accel = lax.cond(
        my_pos > arena_width - 1.0,
        lambda _: -2.0,  # Bounce left
        lambda _: 2.0,  # Default right
        None
    )

    return jnp.array([accel, 0])  # neutral stance


def distance_keeper_1m_jax(state, config):
    """Maintains 1m distance from opponent."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)

    target_distance = 1.0
    tolerance = 0.2

    # Distance control: approach if too far, back away if too close
    accel = jnp.where(distance > target_distance + tolerance, 2.0 * direction,
            jnp.where(distance < target_distance - tolerance, -2.0 * direction,
                       0.0))

    # Use extended stance at optimal range, neutral otherwise
    stance = jnp.where(jnp.abs(distance - target_distance) < tolerance, 1, 0)

    return jnp.array([accel, stance])


def distance_keeper_3m_jax(state, config):
    """Maintains 3m distance from opponent."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)

    target_distance = 3.0
    tolerance = 0.3

    # Too far: approach slowly (1.0). Too close: back away fast (2.0).
    accel = jnp.where(distance > target_distance + tolerance, 1.0 * direction,
            jnp.where(distance < target_distance - tolerance, -2.0 * direction,
                       0.0))

    # Always neutral stance
    return jnp.array([accel, 0])


def distance_keeper_5m_jax(state, config):
    """Maintains 5m distance from opponent."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)

    target_distance = 5.0
    tolerance = 0.8

    accel = jnp.where(distance > target_distance + tolerance, 2.0 * direction,
            jnp.where(distance < target_distance - tolerance, -2.0 * direction,
                       0.0))

    return jnp.array([accel, 0])  # neutral stance


def stamina_efficient_jax(state, config):
    """Conservative stamina management."""
    my_stamina = state.fighter_b.stamina
    max_stamina = state.fighter_b.max_stamina
    stamina_pct = my_stamina / max_stamina

    # Conservative stamina management
    stance = lax.cond(
        stamina_pct > 0.8,
        lambda _: 1,  # High stamina: extended
        lambda _: lax.cond(
            stamina_pct < 0.3,
            lambda _: 2,  # Low stamina: defending
            lambda _: 0,  # Normal: neutral
            None
        ),
        None
    )

    return jnp.array([0.0, stance])  # Stationary


def stamina_waster_jax(state, config):
    """Always uses extended stance to waste stamina."""
    return jnp.array([0.0, 1])  # Stationary extended


def stamina_cycler_jax(state, config):
    """Cycles through stances based on stamina level."""
    my_stamina = state.fighter_b.stamina
    max_stamina = state.fighter_b.max_stamina
    stamina_pct = my_stamina / max_stamina

    # Cycle stances based on stamina
    stance = lax.cond(
        stamina_pct > 0.66,
        lambda _: 1,  # extended (drains stamina)
        lambda _: lax.cond(
            stamina_pct > 0.33,
            lambda _: 0,  # neutral
            lambda _: 2,  # defending (regens stamina)
            None
        ),
        None
    )

    return jnp.array([0.0, stance])


def charge_on_approach_jax(state, config):
    """Charges when opponent is approaching."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)

    # Charge (extended stance) when close
    stance = lax.cond(
        distance < 2.0,
        lambda _: 1,  # extended when close
        lambda _: 0,  # neutral when far
        None
    )

    return jnp.array([0.0, stance])


def reactive_defender_jax(state, config):
    """Stationary opponent that defends when opponent is close."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)

    # Defend when close, neutral when far
    stance = lax.cond(
        distance < 1.5,
        lambda _: 2,  # defending when close
        lambda _: 0,  # neutral when far
        None
    )

    return jnp.array([0.0, stance])


# --- Bridge level (simplified expert) fighters ---

def jab_and_move_jax(state, config):
    """Simplified Boxer: approach to jab range, attack with stamina, retreat when tired."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)
    stamina_pct = state.fighter_b.stamina / state.fighter_b.max_stamina

    def tired(_):
        return jnp.array([-direction * 2.0, 0])  # retreat, neutral

    def approach(_):
        return jnp.array([direction * 2.5, 0])  # approach, neutral

    def in_range(_):
        s = lax.cond(stamina_pct > 0.40, lambda _: 1, lambda _: 0, None)
        return jnp.array([0.0, s])  # stay, attack if stamina

    return lax.cond(
        stamina_pct < 0.25,
        tired,
        lambda _: lax.cond(distance > 1.2, approach, in_range, None),
        None
    )


def wait_and_counter_jax(state, config):
    """Simplified Counter Puncher: defend by default, counter when close."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)
    stamina_pct = state.fighter_b.stamina / state.fighter_b.max_stamina

    def too_tired(_):
        return jnp.array([0.0, 2])  # hold, defend

    def counter(_):
        return jnp.array([direction * 1.0, 1])  # step in, extended

    def default(_):
        return jnp.array([0.0, 2])  # hold, defend

    return lax.cond(
        stamina_pct < 0.20,
        too_tired,
        lambda _: lax.cond(distance < 1.0, counter, default, None),
        None
    )


def hit_and_run_jax(state, config):
    """Simplified Out-Fighter: approach, strike at mid range, retreat when close."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)

    def approach(_):
        return jnp.array([direction * 3.0, 0])  # approach, neutral

    def strike(_):
        return jnp.array([direction * 0.5, 1])  # drift in, extended

    def retreat(_):
        return jnp.array([-direction * 3.0, 0])  # retreat, neutral

    return lax.cond(
        distance > 1.5,
        approach,
        lambda _: lax.cond(distance > 1.0, strike, retreat, None),
        None
    )


def pressure_fighter_jax(state, config):
    """Simplified Slugger: constant forward pressure, attack in range."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)
    stamina_pct = state.fighter_b.stamina / state.fighter_b.max_stamina

    def critical(_):
        return jnp.array([direction * 0.6, 2])  # press, defend

    def in_range(_):
        return jnp.array([direction * 0.6, 1])  # press, extended

    def closing(_):
        return jnp.array([direction * 0.6, 0])  # press, neutral

    return lax.cond(
        stamina_pct < 0.15,
        critical,
        lambda _: lax.cond(distance < 2.0, in_range, closing, None),
        None
    )


def close_range_brawler_jax(state, config):
    """Simplified Swarmer: rush in aggressively, always extended."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)
    stamina_pct = state.fighter_b.stamina / state.fighter_b.max_stamina

    def emergency(_):
        return jnp.array([0.0, 2])  # stop, defend

    def very_close(_):
        return jnp.array([direction * 0.3, 1])  # ease off, extended

    def closing(_):
        return jnp.array([direction * 0.8, 1])  # rush, extended

    return lax.cond(
        stamina_pct < 0.10,
        emergency,
        lambda _: lax.cond(distance < 0.5, very_close, closing, None),
        None
    )


def wall_hugger_left_jax(state, config):
    """Stays near left wall."""
    my_pos = state.fighter_b.position

    # Move toward left wall (position 0)
    accel = lax.cond(
        my_pos < 1.5,
        lambda _: 0.0,  # At wall, stop
        lambda _: -1.5,  # Move to wall
        None
    )

    return jnp.array([accel, 0])  # neutral stance


def wall_hugger_right_jax(state, config):
    """Stays near right wall."""
    my_pos = state.fighter_b.position
    arena_width = config.arena_width

    # Move toward right wall
    accel = lax.cond(
        my_pos > arena_width - 1.5,
        lambda _: 0.0,  # At wall, stop
        lambda _: 1.5,  # Move to wall
        None
    )

    return jnp.array([accel, 0])  # neutral stance


# Shuttle patterns (back and forth movement)
def shuttle_slow_jax(state, config):
    """Shuttles back and forth slowly."""
    my_pos = state.fighter_b.position
    arena_width = config.arena_width

    # Bounce between walls
    accel = lax.cond(
        my_pos < 2.0,
        lambda _: 1.0,  # Near left wall -> go right
        lambda _: lax.cond(
            my_pos > arena_width - 2.0,
            lambda _: -1.0,  # Near right wall -> go left
            lambda _: lax.cond(
                state.fighter_b.velocity > 0,
                lambda _: 1.0,  # Moving right -> continue
                lambda _: -1.0,  # Moving left -> continue
                None
            ),
            None
        ),
        None
    )

    return jnp.array([accel, 0])  # neutral stance


def shuttle_medium_jax(state, config):
    """Shuttles back and forth at medium speed."""
    my_pos = state.fighter_b.position
    arena_width = config.arena_width

    accel = lax.cond(
        my_pos < 2.0,
        lambda _: 1.8,
        lambda _: lax.cond(
            my_pos > arena_width - 2.0,
            lambda _: -1.8,
            lambda _: lax.cond(
                state.fighter_b.velocity > 0,
                lambda _: 1.8,
                lambda _: -1.8,
                None
            ),
            None
        ),
        None
    )

    return jnp.array([accel, 0])


# ---------------------------------------------------------------------------
# Curriculum opponents — directional movement toward/away from learner
# ---------------------------------------------------------------------------

def forward_mover_jax(state, config):
    """Always moves toward learner at accel 2.0. Extended when close."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)

    accel = 2.0 * direction
    stance = jnp.where(distance < 1.5, 1, 0)
    return jnp.array([accel, stance])


def backward_mover_jax(state, config):
    """Always moves away from learner at accel 2.0. Defending when close."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)

    accel = 2.0 * (-direction)
    stance = jnp.where(distance < 2.0, 2, 0)
    return jnp.array([accel, stance])


def sideways_mover_smooth_jax(state, config):
    """Smooth sine oscillation based on position. Stance varies by distance."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)

    accel = 2.0 * jnp.sin(my_pos * 1.5)
    stance = jnp.where(distance < 1.0, 2,
             jnp.where(distance < 2.0, 1, 0))
    return jnp.array([accel, stance])


def aggressive_stance_switcher_jax(state, config):
    """Charges toward learner, cycles extended/neutral on 10-tick period."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    direction = jnp.sign(opp_pos - my_pos)
    stamina = state.fighter_b.stamina

    accel = 0.9 * direction * config.max_acceleration
    cycle_pos = state.tick % 10
    # First 5 ticks: extended (1), next 5: neutral (0)
    base_stance = jnp.where(cycle_pos < 5, 1, 0)
    # Override to defending when stamina critically low
    stance = jnp.where(stamina < 2.0, 2, base_stance)
    return jnp.array([accel, stance])


def defensive_stance_switcher_jax(state, config):
    """Mostly defending (15/20 cycle). Extended when close in attack window."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)

    cycle_pos = state.tick % 20
    in_attack_window = cycle_pos >= 15  # last 5 ticks of 20

    # Approach when close, hold otherwise
    accel = jnp.where(distance < 2.0, 1.5 * direction, 0.0)
    # First 15 ticks: defending. Attack window: extended if close, neutral if far.
    stance = jnp.where(~in_attack_window, 2,
             jnp.where(distance < 2.0, 1, 0))
    return jnp.array([accel, stance])


def forward_charger_jax(state, config):
    """Constant full-accel pressure toward learner. Extended when close."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)

    accel = config.max_acceleration * direction
    stance = jnp.where(distance < 1.5, 1, 0)
    return jnp.array([accel, stance])


def oscillator_jax(state, config):
    """Sine wave movement. Stance varies by distance."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)

    accel = config.max_acceleration * jnp.sin(state.tick * 0.15)
    stance = jnp.where(distance < 0.8, 2,
             jnp.where(distance < 1.5, 1, 0))
    return jnp.array([accel, stance])


def retreater_jax(state, config):
    """Always retreats. Defending when close. Wall bounce."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)
    arena_width = config.arena_width

    retreat_accel = -0.8 * direction
    # Wall bounce: reverse if near walls
    accel = jnp.where(my_pos < 1.0, jnp.abs(retreat_accel),
            jnp.where(my_pos > arena_width - 1.0, -jnp.abs(retreat_accel),
                       retreat_accel))
    stance = jnp.where(distance < 1.0, 2, 0)
    return jnp.array([accel, stance])


def strategic_retreater_jax(state, config):
    """Multi-zone retreat: close=fast retreat+defend, mid=moderate+neutral, far=slow approach+extended."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)

    # Zone-based behavior
    accel = jnp.where(distance < 1.0, -3.0 * direction,
            jnp.where(distance < 3.0, -1.5 * direction,
                       0.5 * direction))
    stance = jnp.where(distance < 1.0, 2,
             jnp.where(distance < 3.0, 0, 1))
    return jnp.array([accel, stance])


# ---------------------------------------------------------------------------
# New fighters — advanced behaviors
# ---------------------------------------------------------------------------

def approach_extended_jax(state, config):
    """Approach at accel 1.5, always extended stance."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    direction = jnp.sign(opp_pos - my_pos)

    accel = 1.5 * direction
    return jnp.array([accel, 1])


def flee_defending_jax(state, config):
    """Flee at accel 2.0, always defending. Wall bounce."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    direction = jnp.sign(opp_pos - my_pos)
    arena_width = config.arena_width

    # Flee away from opponent; pick accel=1.0 at overlap (direction==0)
    flee_accel = jnp.where(direction == 0, 1.0, -direction * 2.0)
    # Wall awareness overrides flee direction
    accel = jnp.where(my_pos < 1.0, 2.0,
            jnp.where(my_pos > arena_width - 1.0, -2.0,
                       flee_accel))
    return jnp.array([accel, 2])


def stamina_burner_jax(state, config):
    """Stamina-aware: high stamina pursues aggressively, low rests."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    direction = jnp.sign(opp_pos - my_pos)
    stamina_pct = state.fighter_b.stamina / state.fighter_b.max_stamina

    accel = jnp.where(stamina_pct > 0.4, 3.0 * direction,
            jnp.where(stamina_pct < 0.2, 0.0,
                       1.5 * direction))
    stance = jnp.where(stamina_pct > 0.4, 1, 0)
    return jnp.array([accel, stance])


def hp_adaptive_jax(state, config):
    """Adapts based on HP advantage: ahead=pursue, behind=retreat, even=maintain 2m."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)

    my_hp_pct = state.fighter_b.hp / state.fighter_b.max_hp
    opp_hp_pct = state.fighter_a.hp / state.fighter_a.max_hp
    hp_diff = my_hp_pct - opp_hp_pct

    # Ahead: pursue extended. Behind: retreat defending. Even: maintain 2m neutral.
    target = 2.0
    tolerance = 0.3
    maintain_accel = jnp.where(distance > target + tolerance, direction * 1.0,
                     jnp.where(distance < target - tolerance, -direction * 2.0, 0.0))

    accel = jnp.where(hp_diff > 0.1, 2.5 * direction,
            jnp.where(hp_diff < -0.1, -2.5 * direction,
                       maintain_accel))
    stance = jnp.where(hp_diff > 0.1, 1,
             jnp.where(hp_diff < -0.1, 2, 0))
    return jnp.array([accel, stance])


def stamina_punisher_jax(state, config):
    """Punishes low-stamina learner. Passive when learner has high stamina."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)
    opp_stamina_pct = state.fighter_a.stamina / state.fighter_a.max_stamina

    # Maintain 3m when learner stamina high, 2m when medium, pursue when low
    maintain_3m = jnp.where(distance > 3.5, direction * 1.0,
                  jnp.where(distance < 2.5, -direction * 2.0, 0.0))
    maintain_2m = jnp.where(distance > 2.3, direction * 1.0,
                  jnp.where(distance < 1.7, -direction * 2.0, 0.0))

    accel = jnp.where(opp_stamina_pct < 0.4, 3.0 * direction,
            jnp.where(opp_stamina_pct > 0.7, maintain_3m,
                       maintain_2m))
    stance = jnp.where(opp_stamina_pct < 0.4, 1, 0)
    return jnp.array([accel, stance])


def range_switcher_jax(state, config):
    """Cycles: 45 ticks at 3m neutral, then 15 ticks rushing extended."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)

    cycle_pos = state.tick % 60
    in_rush = cycle_pos >= 45

    # Maintain 3m during passive phase
    maintain_3m = jnp.where(distance > 3.5, direction * 1.0,
                  jnp.where(distance < 2.5, -direction * 2.0, 0.0))

    accel = jnp.where(in_rush, 4.0 * direction, maintain_3m)
    stance = jnp.where(in_rush, 1, 0)
    return jnp.array([accel, stance])


def comeback_fighter_jax(state, config):
    """Ramps aggression linearly over first 200 ticks."""
    opp_pos = state.fighter_a.position
    my_pos = state.fighter_b.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)

    aggression = jnp.minimum(1.0, state.tick / 200.0)

    # Low aggression: retreat defending
    retreat_accel = -2.0 * direction
    # Mid aggression: maintain 2m, extended when close
    maintain_2m = jnp.where(distance > 2.3, direction * 1.0,
                  jnp.where(distance < 1.7, -direction * 2.0, 0.0))
    mid_stance = jnp.where(distance < 2.0, 1, 0)
    # High aggression: full pursue extended
    pursue_accel = config.max_acceleration * direction

    accel = jnp.where(aggression < 0.3, retreat_accel,
            jnp.where(aggression < 0.7, maintain_2m,
                       pursue_accel))
    stance = jnp.where(aggression < 0.3, 2,
             jnp.where(aggression < 0.7, mid_stance, 1))
    return jnp.array([accel, stance])


# ---------------------------------------------------------------------------
# Example fighters — advanced boxing styles
# ---------------------------------------------------------------------------

def boxer_jax(state, config):
    """Classic boxer: quick jabs, good footwork, disciplined stamina management."""
    my_pos = state.fighter_b.position
    opp_pos = state.fighter_a.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)
    my_stamina_pct = state.fighter_b.stamina / state.fighter_b.max_stamina
    opp_hp_pct = state.fighter_a.hp / state.fighter_a.max_hp

    JAB_RANGE = 1.2
    SAFE_DISTANCE = 2.5

    # Low stamina: defend and back off
    low_stam_accel = jnp.where(distance < SAFE_DISTANCE, -0.5 * direction, 0.0)
    low_stam_stance = 2  # defending

    # High stamina: approach and jab
    high_stam_accel = jnp.where(distance > JAB_RANGE + 0.3, 0.7 * direction,
                      jnp.where(distance < JAB_RANGE, 0.3 * direction, 0.0))
    high_stam_stance = jnp.where(distance > JAB_RANGE + 0.3, 0,  # neutral if far
                       jnp.where(distance < JAB_RANGE, 1, 1))  # extended if close or perfect

    # Medium stamina: conservative
    med_accel = jnp.where(distance < JAB_RANGE, 0.0,
                jnp.where(distance < SAFE_DISTANCE, -0.3 * direction, 0.0))
    med_stance = jnp.where(distance < JAB_RANGE, 1,
                 jnp.where(distance < SAFE_DISTANCE, 0, 2))

    # Select by stamina level
    accel = jnp.where(my_stamina_pct < 0.3, low_stam_accel,
            jnp.where(my_stamina_pct > 0.7, high_stam_accel, med_accel))
    stance = jnp.where(my_stamina_pct < 0.3, low_stam_stance,
             jnp.where(my_stamina_pct > 0.7, high_stam_stance, med_stance))

    # Pursuit override: chase low HP opponent
    pursue = (opp_hp_pct < 0.3) & (my_stamina_pct > 0.5)
    pursue_accel = jnp.where(distance > JAB_RANGE, 1.0 * direction, 0.5 * direction)
    pursue_stance = jnp.where(distance > JAB_RANGE, 0, 1)

    accel = jnp.where(pursue, pursue_accel, accel)
    stance = jnp.where(pursue, pursue_stance, stance)

    return jnp.array([accel, stance])


def counter_puncher_jax(state, config):
    """Counter puncher: defensive, waits for openings, counters vulnerable opponents."""
    my_pos = state.fighter_b.position
    opp_pos = state.fighter_a.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)
    opp_velocity = state.fighter_a.velocity
    my_stamina_pct = state.fighter_b.stamina / state.fighter_b.max_stamina
    opp_stamina_pct = state.fighter_a.stamina / state.fighter_a.max_stamina

    COUNTER_RANGE = 0.9
    COMFORT_ZONE = 2.0

    # Detect opponent charging (good counter opportunity)
    opponent_charging = (jnp.abs(opp_velocity) > 1.5) & (distance < 2.5)

    # Counter opportunity: low opp stamina, charging, or close and moving
    opponent_vulnerable = (
        (opp_stamina_pct < 0.4) |
        opponent_charging |
        ((distance < 1.5) & (jnp.abs(opp_velocity) > 0.5))
    )

    # Counter execution (when vulnerable and we have stamina)
    can_counter = opponent_vulnerable & (my_stamina_pct > 0.4)
    counter_accel = jnp.where(distance < COUNTER_RANGE, 0.4 * direction,
                    jnp.where(distance < COUNTER_RANGE + 0.5, 0.6 * direction,
                              -0.2 * direction))
    counter_stance = jnp.where(distance < COUNTER_RANGE, 1,
                     jnp.where(distance < COUNTER_RANGE + 0.5, 1, 2))

    # Distance management (default)
    dist_accel = jnp.where(distance < COMFORT_ZONE - 0.5, -0.5 * direction,
                 jnp.where(distance > COMFORT_ZONE + 1.0, 0.3 * direction, 0.0))
    dist_stance = jnp.where(distance < COMFORT_ZONE - 0.5, 2,
                  jnp.where(distance > COMFORT_ZONE + 1.0, 0, 2))

    # Select counter vs distance management
    accel = jnp.where(can_counter, counter_accel, dist_accel)
    stance = jnp.where(can_counter, counter_stance, dist_stance)

    # Stamina advantage press override
    stam_press = (my_stamina_pct > 0.8) & (opp_stamina_pct < 0.3)
    press_accel = jnp.where(distance > COUNTER_RANGE, 0.7 * direction, 0.3 * direction)
    press_stance = jnp.where(distance > COUNTER_RANGE, 0, 1)

    accel = jnp.where(stam_press, press_accel, accel)
    stance = jnp.where(stam_press, press_stance, stance)

    return jnp.array([accel, stance])


def out_fighter_jax(state, config):
    """Out-fighter: maintains distance, hit and move, excellent footwork."""
    my_pos = state.fighter_b.position
    opp_pos = state.fighter_a.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)
    my_velocity = state.fighter_b.velocity
    my_stamina_pct = state.fighter_b.stamina / state.fighter_b.max_stamina
    my_hp = state.fighter_b.hp
    opp_hp = state.fighter_a.hp
    arena_width = config.arena_width

    STRIKE_RANGE = 1.3
    SAFE_DISTANCE = 2.8
    DANGER_ZONE = 0.8

    # Emergency escape if too close
    emergency_accel = -1.0 * direction
    emergency_stance = 0  # neutral

    # Hit and run (high stamina)
    moving_toward = (my_velocity * direction) > 0.5
    strike_accel = jnp.where(moving_toward, -0.7 * direction, 0.3 * direction)

    hi_stam_accel = jnp.where(
        (distance > STRIKE_RANGE + 0.2) & (distance < SAFE_DISTANCE), 0.8 * direction,
        jnp.where(
            (distance <= STRIKE_RANGE) & (distance > DANGER_ZONE), strike_accel,
            jnp.where(distance > SAFE_DISTANCE, 0.4 * direction,
                       -0.3 * direction)))
    hi_stam_stance = jnp.where(
        (distance <= STRIKE_RANGE) & (distance > DANGER_ZONE), 1,  # extended
        0)  # neutral

    # Low stamina: distance and recovery
    lo_stam_accel = jnp.where(distance < SAFE_DISTANCE, -0.6 * direction, 0.0)
    lo_stam_stance = jnp.where(
        distance < SAFE_DISTANCE,
        jnp.where(my_stamina_pct < 0.3, 2, 0),
        2)

    # Select by stamina
    accel = jnp.where(distance < DANGER_ZONE, emergency_accel,
            jnp.where(my_stamina_pct > 0.5, hi_stam_accel, lo_stam_accel))
    stance = jnp.where(distance < DANGER_ZONE, emergency_stance,
             jnp.where(my_stamina_pct > 0.5, hi_stam_stance, lo_stam_stance))

    # Arena edge awareness: if near wall and moving away from opponent, reverse
    arena_edge_close = (my_pos < 2.0) | (my_pos > arena_width - 2.0)
    moving_away = (accel * direction) < 0
    edge_override = arena_edge_close & moving_away
    edge_accel = 0.5 * direction
    edge_stance = jnp.where(distance < STRIKE_RANGE, 1, 0)

    accel = jnp.where(edge_override, edge_accel, accel)
    stance = jnp.where(edge_override, edge_stance, stance)

    # Endgame: winning on HP -> stay safe
    winning = my_hp > opp_hp * 1.2
    safe_accel = jnp.where(distance < SAFE_DISTANCE, -0.5 * direction, accel)
    safe_stance = jnp.where(my_stamina_pct < 0.6, 2, 0)

    accel = jnp.where(winning, safe_accel, accel)
    stance = jnp.where(winning, safe_stance, stance)

    return jnp.array([accel, stance])


def slugger_jax(state, config):
    """Slugger: heavy hits, aggressive forward pressure, trades hits for damage."""
    my_pos = state.fighter_b.position
    opp_pos = state.fighter_a.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)
    my_stamina_pct = state.fighter_b.stamina / state.fighter_b.max_stamina
    my_hp_pct = state.fighter_b.hp / state.fighter_b.max_hp
    opp_hp_pct = state.fighter_a.hp / state.fighter_a.max_hp

    POWER_RANGE = 1.0
    CHARGE_DISTANCE = 3.0

    # Critical stamina: must defend
    crit_accel = 0.0
    crit_stance = 2  # defending

    # Build momentum: far away
    charge_accel = 1.0 * direction
    charge_stance = 0  # neutral

    # Approach: mid distance
    approach_accel = 0.9 * direction
    approach_stance = jnp.where(my_stamina_pct < 0.5, 0, 1)

    # Power hit zone: close
    power_accel = jnp.where(my_stamina_pct > 0.3, 0.5 * direction, 0.2 * direction)
    power_stance = 1  # extended

    # Default aggressive
    default_accel = 0.8 * direction
    default_stance = 1  # extended

    # Select by distance (and stamina)
    accel = jnp.where(my_stamina_pct < 0.15, crit_accel,
            jnp.where(distance > CHARGE_DISTANCE, charge_accel,
            jnp.where(distance > POWER_RANGE + 0.5, approach_accel,
            jnp.where(distance <= POWER_RANGE, power_accel,
                       default_accel))))
    stance = jnp.where(my_stamina_pct < 0.15, crit_stance,
             jnp.where(distance > CHARGE_DISTANCE, charge_stance,
             jnp.where(distance > POWER_RANGE + 0.5, approach_stance,
             jnp.where(distance <= POWER_RANGE, power_stance,
                        default_stance))))

    # Berserk mode: go all out when either fighter is low
    berserk = ((opp_hp_pct < 0.25) | (my_hp_pct < 0.25)) & (my_stamina_pct > 0.2)
    accel = jnp.where(berserk, 1.0 * direction, accel)
    stance = jnp.where(berserk, 1, stance)

    return jnp.array([accel, stance])


def swarmer_jax(state, config):
    """Swarmer: constant forward pressure, high work rate, overwhelms with volume."""
    my_pos = state.fighter_b.position
    opp_pos = state.fighter_a.position
    distance = jnp.abs(opp_pos - my_pos)
    direction = jnp.sign(opp_pos - my_pos)
    my_stamina_pct = state.fighter_b.stamina / state.fighter_b.max_stamina
    my_hp_pct = state.fighter_b.hp / state.fighter_b.max_hp
    opp_hp_pct = state.fighter_a.hp / state.fighter_a.max_hp

    SWARM_RANGE = 0.8
    PRESSURE_RANGE = 1.5

    # Distance-based tactics
    dist_accel = jnp.where(distance < SWARM_RANGE, 0.2 * direction,
                 jnp.where(distance < PRESSURE_RANGE, 0.6 * direction,
                            1.0 * direction))
    dist_stance = jnp.where(distance < SWARM_RANGE, 1,
                  jnp.where(distance < PRESSURE_RANGE, 1, 0))

    # Stamina management overrides
    accel = jnp.where(my_stamina_pct < 0.1, 0.0,
            jnp.where(my_stamina_pct < 0.25, 0.5 * direction,
                       dist_accel))
    stance = jnp.where(my_stamina_pct < 0.1, 2,
             jnp.where(my_stamina_pct < 0.25, 0,
                        dist_stance))

    # Chase override: far away and have stamina
    chase = (distance > SWARM_RANGE) & (my_stamina_pct > 0.2)
    accel = jnp.where(chase, 1.0 * direction, accel)

    # Berserk mode: either fighter hurt
    berserk = ((my_hp_pct < 0.4) | (opp_hp_pct < 0.4)) & (my_stamina_pct > 0.15)
    accel = jnp.where(berserk, 1.0 * direction, accel)
    stance = jnp.where(berserk, 1, stance)

    return jnp.array([accel, stance])


# Opponent registry with integer IDs
JAX_OPPONENT_REGISTRY = {
    # Level 1: Fundamentals (stationary, 3-stance system)
    "stationary_neutral": (0, stationary_neutral_jax),
    "stationary_extended": (1, stationary_extended_jax),
    "stationary_defending": (2, stationary_defending_jax),

    # Level 2: Basic Skills (simple movement)
    "approach_slow": (3, approach_slow_jax),
    "flee_always": (4, flee_always_jax),
    "shuttle_slow": (5, shuttle_slow_jax),
    "shuttle_medium": (6, shuttle_medium_jax),
    "circle_left": (7, circle_left_jax),
    "circle_right": (8, circle_right_jax),

    # Level 3: Intermediate (distance/stamina)
    "distance_keeper_1m": (9, distance_keeper_1m_jax),
    "distance_keeper_3m": (10, distance_keeper_3m_jax),
    "distance_keeper_5m": (11, distance_keeper_5m_jax),
    "stamina_waster": (12, stamina_waster_jax),
    "stamina_cycler": (13, stamina_cycler_jax),
    "stamina_efficient": (14, stamina_efficient_jax),
    "charge_on_approach": (15, charge_on_approach_jax),
    "reactive_defender": (39, reactive_defender_jax),
    "jab_and_move": (40, jab_and_move_jax),
    "wait_and_counter": (41, wait_and_counter_jax),
    "hit_and_run": (42, hit_and_run_jax),
    "pressure_fighter": (43, pressure_fighter_jax),
    "close_range_brawler": (44, close_range_brawler_jax),
    "wall_hugger_left": (16, wall_hugger_left_jax),
    "wall_hugger_right": (17, wall_hugger_right_jax),

    # Level 4: Curriculum opponents (directional movement)
    "forward_mover": (18, forward_mover_jax),
    "backward_mover": (19, backward_mover_jax),
    "sideways_mover_smooth": (20, sideways_mover_smooth_jax),
    "aggressive_stance_switcher": (21, aggressive_stance_switcher_jax),
    "defensive_stance_switcher": (22, defensive_stance_switcher_jax),
    "forward_charger": (23, forward_charger_jax),
    "oscillator": (24, oscillator_jax),
    "retreater": (25, retreater_jax),
    "strategic_retreater": (26, strategic_retreater_jax),

    # Level 5: Advanced behaviors
    "approach_extended": (27, approach_extended_jax),
    "flee_defending": (28, flee_defending_jax),
    "stamina_burner": (29, stamina_burner_jax),
    "hp_adaptive": (30, hp_adaptive_jax),
    "stamina_punisher": (31, stamina_punisher_jax),
    "range_switcher": (32, range_switcher_jax),
    "comeback_fighter": (33, comeback_fighter_jax),

    # Example fighters (advanced styles)
    "boxer": (34, boxer_jax),
    "counter_puncher": (35, counter_puncher_jax),
    "out_fighter": (36, out_fighter_jax),
    "slugger": (37, slugger_jax),
    "swarmer": (38, swarmer_jax),
}


def create_multi_opponent_func(opponent_paths, config, n_envs=250):
    """
    Create a JAX function that selects different opponents based on environment index.

    Args:
        opponent_paths: List of opponent file paths
        config: WorldConfig
        n_envs: Number of parallel environments (used for env→opponent mapping)

    Returns:
        Tuple of (jit_compiled_vmap_func, resolved_names: List[str])
    """
    from pathlib import Path

    # Map opponent paths to JAX functions (strict — no silent fallbacks)
    opponent_funcs = []
    resolved_names = []
    for path in opponent_paths:
        name = Path(path).stem
        if name in JAX_OPPONENT_REGISTRY:
            opponent_funcs.append(JAX_OPPONENT_REGISTRY[name][1])
            resolved_names.append(name)
        else:
            raise ValueError(
                f"No JAX implementation for opponent '{name}' (path: {path}). "
                f"Add it to JAX_OPPONENT_REGISTRY in opponents_jax.py. "
                f"Available: {sorted(JAX_OPPONENT_REGISTRY.keys())}"
            )

    # Log resolved opponents for observability
    import logging
    logger = logging.getLogger("opponents_jax")
    logger.info(f"Resolved {len(resolved_names)} JAX opponents for {n_envs} envs: {resolved_names}")

    n_opponents = len(opponent_funcs)

    # Create wrapper functions that capture config in closure
    wrapped_funcs = [
        lambda s, cfg=config, func=f: func(s, cfg)
        for f in opponent_funcs
    ]

    # Build default uniform assignment: each opponent gets n_envs/n_opponents envs
    _n_envs = int(n_envs)
    _envs_per_opponent = max(1, _n_envs // n_opponents)
    default_assignment = jnp.array([
        min(i // _envs_per_opponent, n_opponents - 1) for i in range(_n_envs)
    ], dtype=jnp.int32)

    # Mutable container so reweight_opponents can update the assignment
    assignment_holder = [default_assignment]

    # Create vmapped selector for batch processing
    def single_opponent_decide(state, env_idx):
        """Select and execute opponent logic for a single environment."""
        opponent_idx = assignment_holder[0][env_idx]
        return lax.switch(opponent_idx, wrapped_funcs, state)

    func = jax.jit(jax.vmap(single_opponent_decide))

    def reweight(env_to_opponent_idx):
        """Update env→opponent assignment and re-JIT."""
        nonlocal func
        assignment_holder[0] = jnp.array(env_to_opponent_idx, dtype=jnp.int32)
        func = jax.jit(jax.vmap(single_opponent_decide))

    func.reweight = reweight
    func.get_assignment = lambda: assignment_holder[0]
    func.n_opponents = n_opponents

    return func, resolved_names
