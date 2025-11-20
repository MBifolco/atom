#!/usr/bin/env python3
"""
Interactive coaching interface for Atom Combat.

Usage:
    python coach_fight.py fighter_a.py fighter_b.py --coach 0

    During fight:
    - Press 'a' for aggressive
    - Press 'd' for defensive
    - Press 'b' for balanced
    - Press 'r' for rush (cooldown: 20 ticks)
    - Press 'e' for retreat (cooldown: 15 ticks)
    - Press 'c' for counter mode (cooldown: 25 ticks)
    - Press 'q' to quit
"""

import sys
import time
import threading
from pathlib import Path
from typing import Dict, Optional
import importlib.util

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.arena.match_orchestrator import MatchOrchestrator
from src.arena.arena import Arena
from src.coaching.coaching_wrapper import CoachingWrapper


def load_fighter(fighter_path: str):
    """Load a fighter from a Python file."""
    spec = importlib.util.spec_from_file_location("fighter", fighter_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Fighter()


class InteractiveCoachingSession:
    """Interactive coaching session manager."""

    def __init__(self, fighter_a_path: str, fighter_b_path: str, coach_index: int = 0):
        # Load fighters
        self.fighter_a = load_fighter(fighter_a_path)
        self.fighter_b = load_fighter(fighter_b_path)

        # Wrap the coached fighter
        if coach_index == 0:
            self.fighter_a = CoachingWrapper(self.fighter_a)
            self.coached_fighter = self.fighter_a
            self.coached_name = "Fighter A"
        else:
            self.fighter_b = CoachingWrapper(self.fighter_b)
            self.coached_fighter = self.fighter_b
            self.coached_name = "Fighter B"

        # Create arena and orchestrator
        self.arena = Arena()
        self.orchestrator = MatchOrchestrator(
            arena=self.arena,
            fighter_a=self.fighter_a,
            fighter_b=self.fighter_b,
            max_ticks=500
        )

        self.running = False
        self.match_thread = None
        self.last_status = ""

    def display_banner(self):
        """Display the coaching interface banner."""
        print("\n" + "="*60)
        print("         ATOM COMBAT - INTERACTIVE COACHING MODE")
        print("="*60)
        print(f"\nCoaching: {self.coached_name}")
        print("\nCommands:")
        print("  [A]ggressive  - Increase offensive pressure")
        print("  [D]efensive   - Play it safe, conserve energy")
        print("  [B]alanced    - Reset to neutral stance")
        print("  [R]ush        - All-out attack (20 tick cooldown)")
        print("  [E]scape      - Full retreat (15 tick cooldown)")
        print("  [C]ounter     - Wait and counter (25 tick cooldown)")
        print("  [Q]uit        - End match")
        print("\n" + "="*60 + "\n")

    def display_status(self):
        """Display current match status."""
        if self.orchestrator.current_tick < self.orchestrator.max_ticks:
            # Get current state
            state = {
                "tick": self.orchestrator.current_tick,
                "fighter_a_hp": self.orchestrator.fighter_a_hp,
                "fighter_b_hp": self.orchestrator.fighter_b_hp,
                "fighter_a_stamina": self.orchestrator.fighter_a_stamina,
                "fighter_b_stamina": self.orchestrator.fighter_b_stamina,
                "distance": abs(self.orchestrator.fighter_a_position - self.orchestrator.fighter_b_position)
            }

            # Get coaching stats
            coach_stats = self.coached_fighter.get_coaching_stats()

            # Build status line
            status = (
                f"\rTick: {state['tick']:3d}/500 | "
                f"HP: A={state['fighter_a_hp']:3.0f} B={state['fighter_b_hp']:3.0f} | "
                f"Stamina: A={state['fighter_a_stamina']:2.0f} B={state['fighter_b_stamina']:2.0f} | "
                f"Dist: {state['distance']:3.1f}m | "
                f"Mode: {coach_stats['mode']:10s} | "
                f"CD: {coach_stats['cooldown']:2d} "
            )

            # Only print if changed to reduce flicker
            if status != self.last_status:
                print(status, end="", flush=True)
                self.last_status = status

    def process_command(self, key: str) -> bool:
        """
        Process a coaching command.

        Returns True if command was processed, False to quit.
        """
        key = key.lower()

        if key == 'q':
            return False

        command = None
        message = None

        if key == 'a':
            command = "AGGRESSIVE"
            message = ">> AGGRESSIVE MODE ACTIVATED"
        elif key == 'd':
            command = "DEFENSIVE"
            message = ">> DEFENSIVE MODE ACTIVATED"
        elif key == 'b':
            command = "BALANCED"
            message = ">> BALANCED MODE ACTIVATED"
        elif key == 'r':
            command = "RUSH"
            message = ">> RUSH ATTACK!"
        elif key == 'e':
            command = "RETREAT"
            message = ">> TACTICAL RETREAT!"
        elif key == 'c':
            command = "COUNTER"
            message = ">> COUNTER MODE!"

        if command:
            success = self.coached_fighter.receive_coaching(command)
            if success and message:
                print(f"\n{message}")
            elif not success:
                print(f"\n!! Command on cooldown !!")

        return True

    def run_match_thread(self):
        """Run the match in a separate thread."""
        self.running = True
        while self.running and self.orchestrator.current_tick < self.orchestrator.max_ticks:
            self.orchestrator.step()
            time.sleep(0.02)  # 50 FPS

    def run(self):
        """Run the interactive coaching session."""
        self.display_banner()

        # Start match in background thread
        self.match_thread = threading.Thread(target=self.run_match_thread)
        self.match_thread.start()

        # Main coaching loop
        try:
            while self.running and self.orchestrator.current_tick < self.orchestrator.max_ticks:
                # Display status
                self.display_status()

                # Check for input (non-blocking would be better, but keeping it simple)
                try:
                    # Simple input for cross-platform compatibility
                    import select
                    import sys

                    # Check if input is available
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1)
                        if not self.process_command(key):
                            self.running = False
                            break
                except:
                    # Fallback for Windows or if select isn't available
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n\nMatch interrupted by user")
            self.running = False

        # Wait for match thread to finish
        if self.match_thread:
            self.match_thread.join(timeout=1.0)

        # Display final results
        self.display_results()

    def display_results(self):
        """Display match results and coaching statistics."""
        print("\n\n" + "="*60)
        print("                    MATCH COMPLETE!")
        print("="*60)

        # Determine winner
        if self.orchestrator.fighter_a_hp <= 0:
            winner = "Fighter B"
        elif self.orchestrator.fighter_b_hp <= 0:
            winner = "Fighter A"
        elif self.orchestrator.fighter_a_hp > self.orchestrator.fighter_b_hp:
            winner = "Fighter A"
        elif self.orchestrator.fighter_b_hp > self.orchestrator.fighter_a_hp:
            winner = "Fighter B"
        else:
            winner = "Draw"

        print(f"\nWinner: {winner}")
        print(f"Final HP: A={self.orchestrator.fighter_a_hp:.1f}, B={self.orchestrator.fighter_b_hp:.1f}")
        print(f"Duration: {self.orchestrator.current_tick} ticks")

        # Coaching statistics
        stats = self.coached_fighter.get_coaching_stats()
        print(f"\n{'='*30}")
        print("     COACHING STATISTICS")
        print(f"{'='*30}")
        print(f"Decisions Made: {stats['decisions_made']}")
        print(f"Coaching Overrides: {stats['coaching_overrides']}")
        print(f"Override Rate: {stats['override_rate']:.1f}%")
        print(f"Final Mode: {stats['mode']}")
        print(f"Commands Issued: {len(stats['command_history'])}")

        if stats['command_history']:
            print("\nLast 5 Commands:")
            for cmd, tick in stats['command_history'][-5:]:
                print(f"  Tick {tick}: {cmd}")

        print("\n" + "="*60 + "\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive coaching interface for Atom Combat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands during match:
  A - Aggressive mode
  D - Defensive mode
  B - Balanced mode
  R - Rush attack
  E - Escape/retreat
  C - Counter mode
  Q - Quit match

Example:
  python coach_fight.py fighters/examples/balanced.py fighters/examples/tank.py --coach 0
        """
    )

    parser.add_argument("fighter_a", help="Path to fighter A Python file")
    parser.add_argument("fighter_b", help="Path to fighter B Python file")
    parser.add_argument(
        "--coach",
        type=int,
        default=0,
        choices=[0, 1],
        help="Which fighter to coach (0=A, 1=B, default: 0)"
    )

    args = parser.parse_args()

    # Verify fighter files exist
    if not Path(args.fighter_a).exists():
        print(f"Error: Fighter A file not found: {args.fighter_a}")
        sys.exit(1)

    if not Path(args.fighter_b).exists():
        print(f"Error: Fighter B file not found: {args.fighter_b}")
        sys.exit(1)

    # Run coaching session
    session = InteractiveCoachingSession(
        args.fighter_a,
        args.fighter_b,
        args.coach
    )

    try:
        session.run()
    except Exception as e:
        print(f"\nError during match: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()