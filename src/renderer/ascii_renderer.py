"""
Atom Combat - ASCII Renderer

Renders matches as ASCII visualization.
"""

from typing import Dict, Any, List, Optional
import time


class AsciiRenderer:
    """
    Renders match replays as ASCII art in the terminal.

    Supports:
    - Live rendering during matches
    - Replay from telemetry
    - Configurable playback speed
    """

    def __init__(self, arena_width: float = 12.0, display_width: int = 50):
        """
        Initialize renderer.

        Args:
            arena_width: Physical arena width (from config)
            display_width: Character width for display
        """
        self.arena_width = arena_width
        self.display_width = display_width
        self.scale = arena_width / display_width

        # Stance visual characters
        self.stance_chars = {
            'neutral': '●',
            'extended': '▶',
            'retracted': '◀',
            'defending': '■'
        }

    def render_tick(self, tick_data: Dict[str, Any], dt: float = 0.0842):
        """
        Render a single tick of telemetry.

        Args:
            tick_data: Single tick telemetry data
            dt: Seconds per tick (for time display)
        """
        tick = tick_data["tick"]
        fighter_a = tick_data["fighter_a"]
        fighter_b = tick_data["fighter_b"]
        events = tick_data.get("events", [])

        # Calculate positions on display
        pos_a = int(fighter_a["position"] / self.scale)
        pos_b = int(fighter_b["position"] / self.scale)

        # Create position bar
        bar = ['-'] * self.display_width

        # Get stance characters
        char_a = self.stance_chars.get(fighter_a["stance"], '●')
        char_b = self.stance_chars.get(fighter_b["stance"], '●')

        # Place fighters (handle overlap and bounds)
        pos_a = min(max(0, pos_a), self.display_width - 1)
        pos_b = min(max(0, pos_b), self.display_width - 1)

        if pos_a == pos_b:
            bar[pos_a] = '⚔'  # Both at same position
        else:
            bar[pos_a] = char_a
            bar[pos_b] = char_b

        arena_bar = ''.join(bar)

        # Check if collision occurred
        collision = any(e['type'] == 'COLLISION' for e in events)

        # Print header
        print(f"\n[Tick {tick:3d} | {tick*dt:5.2f}s]")
        print("┌" + "─" * 52 + "┐")

        # Print arena
        if collision:
            print(f"│ {'💥 COLLISION!':^50s} │")
        else:
            print(f"│ {' ':50s} │")

        print(f"│ |{arena_bar}| │")
        print("└" + "─" * 52 + "┘")

        # Print fighter stats
        name_a = fighter_a["name"]
        name_b = fighter_b["name"]
        print(f"\n{name_a:^26s} │ {name_b:^26s}")
        print("─" * 26 + "┼" + "─" * 26)

        # HP bars
        hp_a = fighter_a["hp"]
        max_hp_a = fighter_a["max_hp"]
        hp_b = fighter_b["hp"]
        max_hp_b = fighter_b["max_hp"]
        hp_a_pct = hp_a / max_hp_a if max_hp_a > 0 else 0
        hp_b_pct = hp_b / max_hp_b if max_hp_b > 0 else 0
        hp_bar_a = self._make_bar(hp_a_pct, 20, '█', '░')
        hp_bar_b = self._make_bar(hp_b_pct, 20, '█', '░')
        print(f"HP  {hp_bar_a} {hp_a:5.1f} │ HP  {hp_bar_b} {hp_b:5.1f}")

        # Stamina bars
        stam_a = fighter_a["stamina"]
        max_stam_a = fighter_a["max_stamina"]
        stam_b = fighter_b["stamina"]
        max_stam_b = fighter_b["max_stamina"]
        stam_a_pct = stam_a / max_stam_a if max_stam_a > 0 else 0
        stam_b_pct = stam_b / max_stam_b if max_stam_b > 0 else 0
        stam_bar_a = self._make_bar(stam_a_pct, 20, '▓', '░')
        stam_bar_b = self._make_bar(stam_b_pct, 20, '▓', '░')
        print(f"STA {stam_bar_a} {stam_a:5.1f} │ STA {stam_bar_b} {stam_b:5.1f}")

        # Stats
        vel_a = fighter_a["velocity"]
        vel_b = fighter_b["velocity"]
        mass_a = fighter_a["mass"]
        mass_b = fighter_b["mass"]
        stance_a = fighter_a["stance"]
        stance_b = fighter_b["stance"]

        print(f"Vel {vel_a:+5.2f} m/s          │ Vel {vel_b:+5.2f} m/s")
        print(f"Pos {fighter_a['position']:5.2f}m             │ Pos {fighter_b['position']:5.2f}m")
        print(f"Mass {mass_a:.0f}kg [{stance_a:9s}] │ Mass {mass_b:.0f}kg [{stance_b:9s}]")

        # Event details
        if collision:
            for event in events:
                if event['type'] == 'COLLISION':
                    print(f"\n  Damage: {event['damage_to_a']:.1f} to {name_a}, {event['damage_to_b']:.1f} to {name_b}")
                    print(f"  Relative velocity: {event['relative_velocity']:.2f} m/s")

    def _make_bar(self, percentage: float, width: int, filled_char: str = '█', empty_char: str = '░') -> str:
        """Create a visual percentage bar."""
        filled = int(percentage * width)
        empty = width - filled
        return filled_char * filled + empty_char * empty

    def render_summary(self, match_result: Any, spectacle_score: Optional[Any] = None):
        """
        Render match summary.

        Args:
            match_result: MatchResult object
            spectacle_score: Optional SpectacleScore object
        """
        print("\n" + "=" * 54)
        print("MATCH COMPLETE".center(54))
        print("=" * 54)

        print(f"\nWinner: {match_result.winner}")
        print(f"Duration: {match_result.total_ticks} ticks")
        print(f"Final HP: {match_result.final_hp_a:.1f} vs {match_result.final_hp_b:.1f}")

        collision_count = len([e for e in match_result.events if e["type"] == "COLLISION"])
        print(f"Total Collisions: {collision_count}")

        if spectacle_score:
            print("\n" + "-" * 54)
            print("SPECTACLE SCORE".center(54))
            print("-" * 54)
            print(f"  Duration:            {spectacle_score.duration:.3f}")
            print(f"  Close Finish:        {spectacle_score.close_finish:.3f}")
            print(f"  Stamina Drama:       {spectacle_score.stamina_drama:.3f}")
            print(f"  Comeback Potential:  {spectacle_score.comeback_potential:.3f}")
            print(f"  Positional Exchange: {spectacle_score.positional_exchange:.3f}")
            print(f"  Pacing Variety:      {spectacle_score.pacing_variety:.3f}")
            print(f"  Collision Drama:     {spectacle_score.collision_drama:.3f}")
            print(f"\n  OVERALL:             {spectacle_score.overall:.3f}")

            if spectacle_score.overall >= 0.8:
                assessment = "EXCELLENT ⭐⭐⭐⭐⭐"
            elif spectacle_score.overall >= 0.6:
                assessment = "GOOD ⭐⭐⭐⭐"
            elif spectacle_score.overall >= 0.4:
                assessment = "FAIR ⭐⭐⭐"
            else:
                assessment = "POOR ⭐⭐"

            print(f"\n  {assessment}")

        print("\n" + "=" * 54 + "\n")

    def play_replay(
        self,
        telemetry: Dict[str, Any],
        match_result: Any,
        spectacle_score: Optional[Any] = None,
        playback_speed: float = 1.0,
        skip_ticks: int = 1,
        show_all_ticks: bool = False
    ):
        """
        Play a replay from telemetry with animation.

        Args:
            telemetry: Match telemetry
            match_result: MatchResult object
            spectacle_score: Optional SpectacleScore
            playback_speed: Speed multiplier (1.0 = realtime, 2.0 = 2x speed)
            skip_ticks: Show every Nth tick (1 = all, 5 = every 5th)
            show_all_ticks: If True, show all ticks (ignore skip_ticks)
        """
        ticks = telemetry.get("ticks", [])

        if not ticks:
            print("No telemetry data to render")
            return

        # Get dt from config or use default
        dt = telemetry.get("config", {}).get("dt", 0.0842)

        print("\n" + "=" * 54)
        print("REPLAY START".center(54))
        print("=" * 54)

        for i, tick_data in enumerate(ticks):
            # Skip ticks if configured
            if not show_all_ticks and i % skip_ticks != 0:
                continue

            self.render_tick(tick_data, dt)

            # Sleep for animation (unless last tick)
            if i < len(ticks) - 1:
                time.sleep(dt / playback_speed)

        # Show summary
        self.render_summary(match_result, spectacle_score)
