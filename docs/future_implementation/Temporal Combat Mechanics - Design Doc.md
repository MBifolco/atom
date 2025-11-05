Temporal Combat Mechanics - Design Document

  Core Concept: Neural Bandwidth as Physical Constraint

  All fighters have the same "neural clock speed" but allocate it differently. This isn't a game mechanic - it's
  physics. Just like mass affects acceleration and stamina affects damage, cognitive load affects reaction time.

  Three Cognitive Styles (World-Enforced)

  Like weight classes in boxing, fighters operate in one of three cognitive modes:

  1. REACTIVE Fighter

  - History Window: 2 ticks (minimal memory)
  - Reaction Delay: 1 tick (near-instant)
  - Action Commitment: 1 tick (can change every tick)
  - Physical Cost: 25% higher stamina burn from constant adjustments
  - Fighting Style: Twitchy, opportunistic, exhausting

  2. ANALYTICAL Fighter

  - History Window: 10 ticks (pattern recognition)
  - Reaction Delay: 3 ticks (processing lag)
  - Action Commitment: 2 ticks (moderate flexibility)
  - Physical Cost: Normal stamina burn
  - Fighting Style: Predictive, strategic, vulnerable to rushes

  3. COMMITTED Fighter

  - History Window: 4 ticks (basic awareness)
  - Reaction Delay: 2 ticks (moderate lag)
  - Action Commitment: 4 ticks (locked in)
  - Physical Cost: 30% stamina savings from momentum
  - Fighting Style: Powerful charges, efficient, predictable

  Physical Implementation in Arena

  The Arena enforces these constraints through physics:

  1. Reaction Delay = Actions execute N ticks after decision
  2. Action Commitment = Can't change course for N ticks
  3. Stamina Costs = Direction changes cost extra, commitment saves energy

  Natural Counter System

  - Reactive beats Committed: Dodges the predictable charges
  - Committed beats Analytical: Too fast to analyze, breaks through
  - Analytical beats Reactive: Predicts twitches, baits exhaustion

  Why This Design Works

  1. No Configuration Complexity: Fighters don't choose parameters
  2. Physics-Based: Emerges from energy/information constraints
  3. Strategic Depth: Each style requires different training approaches
  4. Natural Balance: Stamina costs create real tradeoffs

  Training Implications

  Each cognitive style needs different training:
  - Reactive: High APM, stamina management, opportunistic strikes
  - Analytical: Pattern recognition, prediction, trap-setting
  - Committed: Momentum management, timing, decisive strikes

  Future Integration Points

  - Tournaments: Run separate leagues per cognitive style
  - Mixed Battles: Style advantages create interesting dynamics
  - Spectacle: Different styles produce varied fight choreography
  - Evolution: Population training can discover optimal strategies per style

  ---
  This system creates depth through physics rather than configuration. The constraints feel natural because they mirror 
  real combat - you can't simultaneously have perfect awareness, instant reactions, and committed power. The Arena 
  enforces these tradeoffs through stamina and timing mechanics.

  Let's get back to this after we see how the current population training performs!
  