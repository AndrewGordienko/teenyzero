<div align="center">

<picture>
  <img alt="teenyzero logo" src="/docs/logo.png" width="40%" height="40%">
</picture>

### teenyzero: Modeling Geometry for Abductive Search.
*Preparing for a Summer 2026 Research Internship*
</div>

---

**teenyzero** is an experimental AlphaZero-style reinforcement learning stack focused on narrowing the gap between statistical search and logical abductive reasoning.

While traditional MCTS relies on heavy simulation counts to "stumble" upon tactical truths, teenyzero explores **Target-Driven Pathing**—using relational geometry to identify mate structures and prune the search space before a single simulation is run.

---

## Docs

- [Running Guide](./docs/running.md): setup, quick starts, and which launcher to use for each workflow
- [Architecture Guide](./docs/architecture.md): how self-play, training, arena evaluation, runtime profiles, and visualizers fit together
- [Autotune Results](./docs/autotune_results.md): promoted hardware/runtime recommendations from measured autotune sweeps and the one-command autotune pipeline

---

## Research Areas

### 1. AlphaFold-Inspired "Attack Geometry"
Traditional move priors are often flat. We are exploring a relational network—analogous to AlphaFold’s Evoformer—that maps board positions to a latent **pairwise geometry**. 
- **Goal:** Identify "King Zones" and masking irrelevant pieces to collapse the search window.
- **Mechanism:** Mapping board states to latent attack vectors → identifying mate structures → generating high-certainty move priors.

### 2. LLM-Infused Search Heuristics
Mapping the landscape of using a Large Language Model's internal world model as a heuristic. By leveraging the context inherent in these models, we aim to tighten traditional MCTS selection logic (PUCT) with "logical weight" derived from LLM-based backward chaining.

### 3. Target-Driven Pathing
Modeling "mate structure" as a target state. Instead of searching forward blindly, the engine learns to bridge the gap between the current state and a logically sound target, moving toward effective backward-chaining.

### 4. Autonomous Experimentation (The "Overnight" Agent)
A simplified single-GPU implementation of **nanochat** allows an agent to experiment autonomously.
- **Process:** The agent modifies its own training code → executes a 5-minute micro-train → evaluates deltas → iterates.
- **Output:** Wake up to a log of successful architectural mutations and optimized weights.

---

## Roadmap

- [ ] **Search Engine Core**
  - [ ] Implement MCTS (Monte Carlo Tree Search)
  - [ ] Enable network-driven move selection
- [ ] **Interactive Interface**
  - [ ] Develop simple visual dashboard for play
  - [ ] Integrate engine with web-based UI

---
