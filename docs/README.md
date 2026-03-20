# Atom Combat Documentation

Technical documentation for the Atom Combat AI fighting game.



## 📚 Training & AI

- **[HOW_TRAINING_WORKS.md](HOW_TRAINING_WORKS.md)** - Overview of RL training
- **[REWARD_STRUCTURE.md](REWARD_STRUCTURE.md)** - Complete reward system
- **[PROGRESSIVE_TRAINING.md](PROGRESSIVE_TRAINING.md)** - Progressive curriculum
- **[POPULATION_TRAINING.md](POPULATION_TRAINING.md)** - Population-based training
- **[TRAINING_REFACTOR_ROADMAP.md](TRAINING_REFACTOR_ROADMAP.md)** - Stability refactor phases + local-first testing strategy
- **[LOCAL_TESTING_WORKFLOW.md](LOCAL_TESTING_WORKFLOW.md)** - Local deterministic workflow to reduce Colab dependency
- **[COLAB_SETUP.md](COLAB_SETUP.md)** - Google Colab workflow with persistent Drive cache
- **[COLAB_VALIDATION_CHECKLIST.md](COLAB_VALIDATION_CHECKLIST.md)** - Milestone gate checklist for Colab validation (Phases 1/3/5)
- **[../notebooks/Atom_Training_Colab.ipynb](../notebooks/Atom_Training_Colab.ipynb)** - Ready-to-run Colab notebook
- **[CONTINUAL_LEARNING.md](CONTINUAL_LEARNING.md)** - Catastrophic forgetting prevention
- **[STAMINA_ISSUE.md](STAMINA_ISSUE.md)** - Stamina exhaustion fix
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)** - Historical improvements

## 🏗️ Architecture & Systems

- **[PLATFORM_ARCHITECTURE.md](PLATFORM_ARCHITECTURE.md)** - System architecture
- **[TRAINING_CONFIGURATION.md](TRAINING_CONFIGURATION.md)** - Training config guide
- **[TESTING_SYSTEM.md](TESTING_SYSTEM.md)** - Testing framework
- **[TEST_COVERAGE.md](TEST_COVERAGE.md)** - Test coverage report

## 🔧 Project Planning

- **[VISION_GAP_ANALYSIS.md](VISION_GAP_ANALYSIS.md)** - Vision vs implementation
- **[IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md)** - Development checklist
- **[MASS_CONVERGENCE_PROBLEM.md](MASS_CONVERGENCE_PROBLEM.md)** - Mass convergence fix

## 📁 Original Vision

The `original_vision/` directory contains the original design documents representing the initial vision before implementation.

## 🔗 Other Documentation

- **[../README.md](../README.md)** - Main project README
- **[../training/README.md](../training/README.md)** - Training usage guide
- **[../fighters/README.md](../fighters/README.md)** - Fighter examples
- **[../tests/README.md](../tests/README.md)** - Test suite docs
- **[../archived/README.md](../archived/README.md)** - Archived benchmarks/tests

---

**Latest**: JAX optimization complete - 77x speedup with GPU (AMD ROCm 7.1)
