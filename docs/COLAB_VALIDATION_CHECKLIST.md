# Colab Validation Checklist

Use this checklist for milestone validation gates so Colab stays a high-value
final check, not part of the daily inner loop.

## Before Any Gate

1. Open [`notebooks/Atom_Training_Colab.ipynb`](../notebooks/Atom_Training_Colab.ipynb).
2. Run cells 1-4 (mount + config + bootstrap + sanity checks).
3. Confirm preflight is green for the stage you are about to run.

Preflight commands (also used by the notebook):

```bash
python -m src.atom.training.utils.colab_preflight --stage bootstrap --strict
python -m src.atom.training.utils.colab_preflight --stage smoke --output-dir /content/drive/MyDrive/atom_runs/quick_test --require-gpu --strict
python -m src.atom.training.utils.colab_preflight --stage full --output-dir /content/drive/MyDrive/atom_runs/run1 --require-gpu --strict
python -m src.atom.training.utils.colab_preflight --stage resume --checkpoint-dir /content/drive/MyDrive/atom_runs/run1 --require-gpu --strict
```

## Gate A: End of Phase 1 (Signal Parity)

Goal:
- Verify refactored signal paths still run end-to-end in Colab.

Run:
1. Notebook quick smoke cell (streaming).

Pass criteria:
- Bootstrap succeeds without manual stash/reset intervention.
- Sanity check reports CUDA runtime when GPU is requested.
- Quick run starts, collects rollouts, and writes logs/checkpoints to Drive.

Artifacts to keep:
- Quick run output directory (`ATOM_SMOKE_OUTPUT_DIR`).
- Any preflight output for failed attempts.

## Gate B: End of Phase 3 (Recovery Hardening)

Goal:
- Validate checkpoint/recovery behavior on real Colab runtime.

Run:
1. Start curriculum/full run with checkpointing enabled.
2. Interrupt runtime mid-training.
3. Re-run bootstrap and resume workflow.

Suggested command (if not using notebook cell):

```bash
python train_progressive.py \
  --mode curriculum \
  --device auto \
  --use-vmap \
  --timesteps 2000000 \
  --checkpoint-interval 100000 \
  --output-dir /content/drive/MyDrive/atom_runs/recovery_gate
```

Resume:

```bash
python train_progressive.py \
  --mode curriculum \
  --device auto \
  --use-vmap \
  --resume-curriculum \
  --checkpoint-interval 100000 \
  --output-dir /content/drive/MyDrive/atom_runs/recovery_gate
```

Pass criteria:
- Resume starts from latest checkpoint bundle (non-zero progress), not from scratch.
- No manual checkpoint surgery required.

Artifacts to keep:
- `curriculum/models/checkpoint_*` bundles.
- Resume logs proving non-zero restart.

## Gate C: End of Phase 5 (Operational Readiness)

Goal:
- Validate production Colab workflow: bootstrap, full run, and resume.

Run:
1. Notebook full training cell (streaming).
2. Notebook resume cell (streaming), or equivalent CLI.

Pass criteria:
- Preflight catches misconfiguration before training starts.
- Full run and resume commands are copy-paste runnable with env defaults.
- Logs stream in notebook while the process is running.
- Outputs persist to Drive and are reusable across runtime restarts.

Artifacts to keep:
- Full run directory (`ATOM_FULL_OUTPUT_DIR`).
- Resume log output and resulting checkpoints.

## Failure Triage

If a gate fails:
1. Save full console output from the failing cell.
2. Re-run the matching preflight command for targeted diagnostics.
3. Log failure category:
   - bootstrap/config
   - GPU/runtime
   - training stability
   - checkpoint/resume
4. Fix locally first when possible, then re-run only the failed gate in Colab.
