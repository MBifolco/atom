#!/usr/bin/env python3
"""
Atom Combat - Fighter Training CLI

Train AI fighters using reinforcement learning.

Usage:
    # Train against single opponent
    python train_fighter.py --opponent fighters/tank.py --output trained_fighter

    # Train against multiple opponents
    python train_fighter.py --opponents fighters/tank.py fighters/rusher.py --output my_fighter

    # Quick training test
    python train_fighter.py --opponent fighters/tank.py --episodes 1000 --output test_fighter

    # Full training with all options
    python train_fighter.py \\
        --opponents fighters/*.py \\
        --output champion \\
        --episodes 50000 \\
        --cores 10 \\
        --mass 70 \\
        --patience 10
"""

# Add parent directory to path FIRST before any imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import glob

from training.src.trainers import train_fighter_ppo, train_curriculum_ppo, train_fighter_sac, train_curriculum_sac
from training.src.onnx_fighter import export_to_onnx, create_fighter_wrapper


def main():
    parser = argparse.ArgumentParser(
        description='Train an AI fighter using reinforcement learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Opponent configuration
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--opponent', help='Single opponent fighter file')
    group.add_argument('--opponents', nargs='+', help='Multiple opponent fighter files (can use wildcards)')
    group.add_argument('--curriculum', action='store_true', help='Use curriculum learning (automatic progression)')

    # Output
    parser.add_argument('--output', required=True, help='Output name (will create .zip and .onnx)')

    # Training configuration
    parser.add_argument('--episodes', type=int, default=10000, help='Target number of episodes (default: 10000)')
    parser.add_argument('--cores', type=int, default=10, help='CPU cores to use for parallel training (default: 10)')
    parser.add_argument('--mass', type=float, default=70.0, help='Fighter mass in kg (default: 70)')
    parser.add_argument('--opponent-mass', type=float, default=None, help='Opponent mass in kg (default: same as --mass)')
    parser.add_argument('--max-ticks', type=int, default=1000, help='Max ticks per episode (default: 1000)')

    # Stopping criteria
    parser.add_argument('--patience', type=int, default=5, help='Plateau patience (default: 5 checks)')

    # Output options
    parser.add_argument('--checkpoint-freq', type=int, default=10000, help='Checkpoint frequency (default: 10000 steps)')
    parser.add_argument('--tensorboard', help='TensorBoard log directory (optional)')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')

    # Post-training
    parser.add_argument('--create-wrapper', action='store_true', help='Create standalone .py wrapper')

    # Algorithm selection
    parser.add_argument('--algorithm', choices=['ppo', 'sac'], default='ppo',
                       help='Training algorithm: ppo (default, best for curriculum) or sac (better exploration)')

    args = parser.parse_args()

    # Default opponent mass to same as fighter mass if not specified
    if args.opponent_mass is None:
        args.opponent_mass = args.mass

    # Select training functions based on algorithm
    if args.algorithm == 'sac':
        train_fighter = train_fighter_sac
        train_curriculum = train_curriculum_sac
        if args.cores > 1:
            print(f"⚠️  Note: SAC works best with single environment, adjusting --cores from {args.cores} to 1")
            args.cores = 1  # SAC is off-policy, doesn't benefit from parallel envs
    else:  # ppo
        train_fighter = train_fighter_ppo
        train_curriculum = train_curriculum_ppo

    # Handle curriculum mode
    if args.curriculum:
        print(f"Using curriculum learning mode with {args.algorithm.upper()}...")

        # Set up output directory
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        output_base = output_dir / args.output

        try:
            train_curriculum(
                output_base=output_base,
                episodes_per_level=args.episodes,
                fighter_mass=args.mass,
                opponent_mass=args.opponent_mass,
                max_ticks=args.max_ticks,
                graduation_tests=20,
                verbose=not args.quiet,
                create_wrappers=args.create_wrapper
            )
        except KeyboardInterrupt:
            print("\n\nCurriculum training interrupted!")
            sys.exit(1)
        except Exception as e:
            print(f"\n\nError during curriculum training: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
        return

    # Handle opponent(s)
    opponent_files = []
    if args.opponent:
        opponent_files = [args.opponent]
    else:
        # Expand wildcards
        for pattern in args.opponents:
            matches = glob.glob(pattern)
            if not matches:
                print(f"Warning: No files matched pattern: {pattern}", file=sys.stderr)
            opponent_files.extend(matches)

    if not opponent_files:
        print("Error: No opponent files found!", file=sys.stderr)
        sys.exit(1)

    # Validate opponent files
    for filepath in opponent_files:
        if not Path(filepath).exists():
            print(f"Error: Opponent file not found: {filepath}", file=sys.stderr)
            sys.exit(1)

    # Output paths - create a directory for this model
    output_dir = Path(__file__).parent / "outputs" / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Model directory: {output_dir}")

    model_path = output_dir / "model.zip"
    onnx_path = output_dir / "model.onnx"

    # Train
    try:
        model = train_fighter(
            opponent_files=opponent_files,
            output_path=str(model_path),
            episodes=args.episodes,
            n_envs=args.cores,
            fighter_mass=args.mass,
            opponent_mass=args.opponent_mass,
            max_ticks=args.max_ticks,
            checkpoint_freq=args.checkpoint_freq,
            patience=args.patience,
            verbose=not args.quiet,
            tensorboard_log=args.tensorboard
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted!")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during training: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Export to ONNX
    print(f"\nExporting to ONNX: {onnx_path}")
    try:
        export_to_onnx(str(model_path), str(onnx_path))
    except Exception as e:
        print(f"Warning: ONNX export failed: {e}", file=sys.stderr)
        print("Model saved as .zip only")

    # Create wrapper if requested
    if args.create_wrapper and onnx_path.exists():
        wrapper_path = output_dir / "fighter.py"
        print(f"\nCreating fighter wrapper: {wrapper_path}")
        create_fighter_wrapper(str(onnx_path), str(wrapper_path))

        print(f"\n✓ Trained fighter ready!")
        print(f"\nTest it:")
        print(f"  python atom_fight.py {wrapper_path} fighters/tank.py --html replay.html")
    else:
        print(f"\n✓ Training complete!")
        print(f"\nFiles created:")
        print(f"  {model_path} - Stable-Baselines3 model")
        if onnx_path.exists():
            print(f"  {onnx_path} - ONNX model")


if __name__ == "__main__":
    main()
