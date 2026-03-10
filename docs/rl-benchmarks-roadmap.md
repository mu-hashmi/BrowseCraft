# RL Benchmark Roadmap (Deferred)

External benchmarks are intentionally deferred until the core HUD + simulator RL loop is stable.

## Deferred Benchmarks

- SpartQA
- StepGame
- FloorplanQA
- MineBench

## Entry Criteria

Start benchmark integration only after all conditions are true:

- T1-T6 task generation is deterministic and covered by tests.
- HUD scenarios for T1-T6 run locally via `hud dev`.
- Grader and reward components are stable under repeated runs.
- Baseline eval script runs end-to-end on Claude Sonnet 4.6 and Opus 4.6.
- Trajectory export and RFT task export are validated.

## Planned Integration Sequence

1. Define benchmark adapter interfaces and strict dataset schema checks.
2. Add one benchmark at a time, starting with FloorplanQA.
3. Add transfer evaluation reports that compare base vs trained models.
4. Fold benchmark metrics into the baseline summary output.

No benchmark loaders or fake stub evaluators are added in this phase.
