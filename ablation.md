# giga ablation study

All runs on commit `fca73946` (detached HEAD), env `puffer_giga` (privileged vector
observations, `obs_mode = vector`), config `pufferlib/config/giga/giga.ini`.

Started 2026-08-20. The reference point for the whole study is the best historical
`puffer_giga_3cam` run (`0ejcqldx`, also `fca73946`), which reached
`score 0.739 / completion_rate 0.854` at 600-step episodes.

---

## 1. Why `fca73946`

Training on `main` (`cc4fb56b`) produced agents that ignore road boundaries and other
vehicles. `fca73946` is the last commit with a known-good result, so it is used as the
control. Differences from `cc4fb56b`:

| difference | `fca73946` | `cc4fb56b` |
|---|---|---|
| `EGO_FEATURES_JERK` | `11 + 13 = 24` | `9 + 13 = 22` (checkpoints incompatible) |
| reward clip | hardcoded `torch.clamp(r, -1, 1)`, active | `reward_clip = False` |
| off-road test | `ROAD_EDGE` intersection only | union with `min_lane_distance > lane_width/2` |
| `ALPHA_COMFORT` | `U(0, 0.05)` | `0` |
| `ALPHA_L_ALIGN` | `U(0.00025, 0.025)` | `0` |
| `ALPHA_L_CENTER` | `U(0.00025, 0.075)` | `0` |
| `V_GOAL` | observed, `U(0, 20)` | constant `3.0`; slot reused for `is_final` |
| LSTM state on `done` | not zeroed | zeroed (`models.py:140,191`) |

Note: the historical runs never set `goal_behavior` to respawn. Every one of the 13
runs from 2026-08-12/13 used `--env.goal-behavior 3` (GOAL_REMOVE).

---

## 2. Method

Metrics come from the `puffer` dashboard, one flush per epoch, 1909 flushes per run.

Two rules, both learned the hard way during this study:

0. **Deduplicate the dashboard before averaging.** The dashboard reprints the previous
   `self.stats` when an epoch produces no new env log (`pufferl.py:874-877`), and the env
   log only fires when an episode ends. The repeat factor scales with episode length:
   measured 1.0 / 1.7 / 2.6 / 5.0 at 100 / 200 / 300 / 600 steps, and ~10 at 1200. Means
   are unaffected, but any standard deviation computed over raw flushes is understated by
   roughly the square root of that factor. Standard deviations reported in sections 4-9
   for runs with episodes longer than 100 steps are therefore too small, and the claim
   that run G was "the tightest run in the study" is an artifact of this, not a property
   of the run.

1. **Compare at matched epoch, never against another run's endpoint.** Early training
   looks nothing like late training; at 6% of budget every arm looks identical.
2. **Never read a single 60-flush window.** The flush-to-flush standard deviation of
   `completion_rate` is **0.053**, which is the same size as the effects being measured.
   Four separate mid-run conclusions during this study were later contradicted by more
   data. Final numbers use the mean of the **last 300 flushes**, cross-checked against
   six non-overlapping 100-flush blocks.

Source of the noise: `resample_frequency = 2560` swaps the map set roughly every 13
episodes, and `agents_per_map ~ U{1,120}` makes map difficulty highly variable.

**`lane_alignment_rate` as logged is biased downward in proportion to
`completion_rate`.** `add_log` divides the accumulated `LANE_ALIGNED_IDX` by the full
episode length, but a removed agent — and with `goal_behavior = 3` that includes every
agent that *succeeds* — contributes zeros to the numerator for the rest of the episode
while the denominator stays fixed. The better a policy is, the earlier its agents leave,
and the worse this metric looks. Measured against live agent-steps only, the ranking
inverts (see section 8). Do not compare this metric across runs with different
completion rates.

`episode_return` is **not comparable across arms** with different reward functions:
an arm missing `r_center`/`r_align` is not charged those per-step penalties at all.
Only `completion_rate`, `score`, `collisions_per_agent`, `offroad_per_agent`,
`lane_alignment_rate` and `dnf_rate` are cross-comparable — and even those break down
when agent removal truncates metric accrual (see run E).

---

## 3. Runs

Shared settings unless stated: `episode_length` and `goal_behavior` via CLI,
`total_timesteps = 2e9` (1908 epochs), `num_agents = 1024`, `batch_size = 1048576`,
`minibatch_size = 32768`, `rollout_horizon = 128`, `adamw`, `lr 5e-4` (annealed),
`gamma 0.999`, `gae_lambda 0.95`, `ent_coef 0.01`, `collision_behavior = 0`,
`offroad_behavior = 0`, reward clip `[-1, 1]` active.

| id | run | episode | start | conditioning | behaviors | checkpoint |
|---|---|---|---|---|---|---|
| A | stage 1 | 100 | cold | full | ignore/ignore | `puffer_giga_20260820_173038_x6c2mvje.pt` |
| B | stage 2 | 200 | warm ← A | full | ignore/ignore | `puffer_giga_20260820_190357_nnvni1n3.pt` |
| C | stage 3 | 600 | warm ← B | full | ignore/ignore | `puffer_giga_20260820_204607_9toyx4uf.pt` |
| D | tri-ablation | 100 | cold | **4,5,7 = 0** | ignore/ignore | `puffer_giga_20260820_220901_rx02td80.pt` |
| E | comfort-only | 100 | cold | **4 = 0** | ignore/ignore | `puffer_giga_20260821_065015_uiebg6eh.pt` |
| F | remove-behavior | 100 | cold | full | **remove/remove** | `puffer_giga_20260821_082823_7u26rtc4.pt` |
| G | remove @300 | **300** | cold | full | **remove/remove** | `puffer_giga_20260821_100538_omyawtlq.pt` |
| H | **matched control** | 300 | cold | full | ignore/ignore | `puffer_giga_20260821_120041_qt3lw4zq.pt` |
| I | remove @600 | 600 | cold | full | remove/remove | `puffer_giga_20260821_141617_og6aw9wl.pt` |
| J | remove @1200 | 1200 | cold | full | remove/remove | `puffer_giga_20260821_153252_9w0caqko.pt` |

Conditioning slot numbers refer to `conditioning.h`:
`4 = ALPHA_COMFORT`, `5 = ALPHA_L_ALIGN`, `7 = ALPHA_L_CENTER`.
Zeroing 5 also makes `6 = ALPHA_VEL_ALIGN` inert (it multiplies inside `r_align`);
zeroing 7 also makes `8 = ALPHA_CENTER_BIAS` inert (it lives inside `r_center`).

### Commands

```bash
# A
puffer train puffer_giga --env.episode-length 100 --env.goal-behavior 3
# B
puffer train puffer_giga --env.episode-length 200 --env.goal-behavior 3 \
  --load-model-path experiments/puffer_giga_20260820_173038_x6c2mvje.pt
# C
puffer train puffer_giga --env.episode-length 600 --env.goal-behavior 3 \
  --load-model-path experiments/puffer_giga_20260820_190357_nnvni1n3.pt
# D, E  (edit conditioning.h, then: touch binding.c && python setup.py build_ext --inplace)
puffer train puffer_giga --env.episode-length 100 --env.goal-behavior 3
# F
puffer train puffer_giga --env.episode-length 100 --env.goal-behavior 3 \
  --env.collision-behavior 2 --env.offroad-behavior 2
# G
puffer train puffer_giga --env.episode-length 300 --env.goal-behavior 3 \
  --env.collision-behavior 2 --env.offroad-behavior 2
# H  (matched control: G with one variable changed back)
puffer train puffer_giga --env.episode-length 300 --env.goal-behavior 3
```

---

## 4. Results (mean of last 300 flushes)

| run | episode | completion | score | coll/agent | offroad/agent | lane_align | dnf | return |
|---|---|---|---|---|---|---|---|---|
| A stage 1 | 100 | 0.191 | 0.168 | 2.447 | 0.687 | 0.778 | 0.653 | -5.979 |
| B stage 2 | 200 | 0.426 | 0.392 | 2.630 | 0.601 | 0.690 | 0.545 | -6.183 |
| C stage 3 | 600 | **0.765** | **0.652** | 3.730 | 0.618 | 0.472 | 0.248 | -9.173 |
| D tri-abl | 100 | 0.135 | 0.117 | 2.520 | 0.707 | 0.783 | 0.680 | -5.639 |
| E comfort0 | 100 | 0.145 | 0.124 | 2.660 | 0.983 | 0.748 | 0.634 | -6.375 |
| F remove | 100 | 0.454 | 0.528 | 0.102* | 0.028* | 0.640 | 0.603 | +0.226* |
| G remove | 300 | **0.755** | **0.736** | 0.109* | 0.038* | 0.400 | 0.267 | +0.514* |
| H control | 300 | 0.189 | 0.160 | 3.091 | 1.641 | 0.546 | 0.579 | -9.192 |
| *3cam ref* | *600* | *0.854* | *0.739* | *2.397* | *0.572* | *0.416* | — | *-6.026* |

`collisions_per_agent` and `offroad_per_agent` are **frame counts** and scale with
episode length; divide by episode length to compare across rows A/B/C.

\* Run F's starred cells are not comparable to the other rows: removal freezes the
agent, so the counters become event counts rather than frame counts and `episode_return`
stops accruing. Only `completion_rate`, `score`, `lane_alignment_rate` and `dnf_rate`
carry the same meaning as elsewhere — and see the caveat in section 7.

### Curriculum (A → B → C)

The 100 → 200 → 600 curriculum reproduces the historical result within reach but does
not match it: `score 0.652` vs `0.739`, `completion 0.765` vs `0.854`. Normalized bad
frames: 0.50/100 steps for the 3cam reference vs 0.72/100 steps for C.

A single stage cannot substitute for the curriculum — run A alone, given the full 2e9
budget, reaches only `completion 0.191`.

### Conditioning ablation (A vs D vs E)

Matched-epoch, last 300 flushes, relative to A:

| metric | D (4,5,7 = 0) | E (4 = 0 only) |
|---|---|---|
| completion_rate | **-29%** | **-24%** |
| score | **-30%** | **-26%** |
| collisions_per_agent | +3% | +9% |
| offroad_per_agent | +3% | **+43%** |
| lane_alignment_rate | +1% | -4% |
| dnf_rate | +4% | -3% |

Completion deficit across six non-overlapping 100-flush blocks:

| block | 1 | 2 | 3 | 4 | 5 | 6 | mean |
|---|---|---|---|---|---|---|---|
| D | -20% | -22% | -12% | -24% | -29% | -35% | **-24%** |
| E | -20% | -5% | -7% | -11% | -31% | -28% | **-17%** |

---

## 5. Conclusions

**Supported by all six independent blocks:**

1. Zeroing the dense shaping terms costs **20–30% of completion_rate and score**.
2. `ALPHA_COMFORT` alone accounts for most of that. The block distributions of D and E
   overlap heavily (blocks 1, 5, 6 are nearly equal), so **the additional contribution
   of `ALPHA_L_ALIGN` + `ALPHA_L_CENTER` on top of comfort is not measurable here**.
3. **Lane discipline does not come from the lane terms.** `lane_alignment_rate` lands at
   0.75–0.78 in all three arms, including the arm where nothing rewards it. What
   produces it is not established. It is *not* the off-road penalty geometry: on
   `fca73946` the only thing that sets `OFFROAD` is a bounding-box intersection with a
   `ROAD_EDGE` segment (`drive.h:1350-1353`) — the `min_lane_distance > lane_width/2`
   test does not exist on this commit, it was added later in `c2b13e67`. The most
   likely remaining cause is `r_velocity`, which pays for forward progress only while
   `lane_valid` (within 4 m of a centerline) and scales with `cos(lane_heading_error)`.
   Note the logged values here are all deflated by the bias described in section 2;
   the arms are comparable to each other only because their completion rates are
   similar (0.191 / 0.135 / 0.145).

4. **`collision_behavior`/`offroad_behavior` = remove is by far the largest effect in
   the study.** Run F reaches `completion 0.454` / `score 0.528` against run A's
   `0.191` / `0.168` — the six blocks are 0.440–0.475 for F and 0.180–0.205 for A,
   completely non-overlapping. Collision events drop to 0.102 per agent. But the
   attribution is confounded; see section 7.

**Not explained:**

- ~~Run F is less lane-aligned than the baseline (0.640 vs 0.778).~~ **Resolved:**
  this is the `lane_alignment_rate` bias of section 2, not a behavioural difference.
  Run F completes 2.4x more often than run A, so 2.4x more of its agents are frozen
  out early and contribute zeros.
- `offroad_per_agent` is non-monotone: +43% when only comfort is removed, +3% when all
  three are removed. Consistent across both the 300- and 600-flush windows, so probably
  not pure noise, but there is no mechanism for it. Most likely single-seed variance.

**Bearing on the original problem:** the conditioning ablation produces a *more passive*
policy (higher `dnf_rate`, lower completion), which is the **opposite** of the observed
"drives through everything" behaviour on `cc4fb56b`. Zeroing 4/5/7 is worth 20–30% of
performance but is **not** the cause of that behaviour.

---

## 6. Limitations

- **One seed per arm.** Within-arm block-to-block spread (D: -12% to -35%; E: -5% to
  -31%) is comparable to the between-arm effect. Every percentage above is a point
  estimate with roughly ±10 points of uncertainty.
- All runs use `obs_mode = vector`; the historical reference is a camera policy with a
  different architecture (4.0M params vs 595K) and `num_agents = 2048`.
- `episode_return` is not comparable across arms with different reward functions.

---

## 7. Open questions

Remaining differences between `fca73946` and `cc4fb56b`, ranked by suspicion for the
"drives through everything" behaviour:

1. **`goal_behavior = 0` (respawn) vs `3` (remove).** Respawn sets `terminals[i] = 1`
   (`drive.h:3318`) without a bootstrap, so `nextnonterminal = 0` in the advantage
   kernel. With a net-negative reward stream, reaching the goal cancels the remaining
   discounted penalty — an escape worth far more than the nominal `+1`. This predicts
   exactly the observed behaviour. **Next experiment: `--env.goal-behavior 0`, cold, 100
   steps, everything else as run A.**
2. `episode_length` 1280 vs 100 — amplifies (1): a longer life means a more negative
   `V`, so the escape is worth more.
3. The lane-distance off-road test — increases off-road frames sharply, but pushes the
   policy toward caution, not recklessness.
4. Ego observation width 24 → 22 — no expected behavioural direction.

### Note on `REMOVE` (run F)

`collision_behavior = 2` / `offroad_behavior = 2` set `x = y = -10000.0f`, which equals
`INVALID_POSITION`, so `compute_agent_metrics` returns early and the agent accrues **no
further penalty** — and, unlike the goal path, **no `terminals[i]`** is set. Removal is
therefore a free, permanent exit from the penalty stream.

Two consequences that make run F hard to interpret:

- Metric semantics change: `collisions_per_agent` becomes an event count rather than a
  frame count, so it is not comparable to runs A–E.
- Collisions remove **both** vehicles (`drive.h:1443-1444`). Measured under a random
  policy, 39 of 64 agents are gone within 100 steps, so traffic density collapses and
  the task becomes much easier mid-episode.

Final result (last 300 flushes, matched epoch against run A):

| metric | A baseline | F remove |
|---|---|---|
| completion_rate | 0.191 | **0.454** |
| score | 0.168 | **0.528** |
| collisions_per_agent | 2.447 (frames) | 0.102 (events) |
| offroad_per_agent | 0.687 (frames) | 0.028 (events) |
| lane_alignment_rate | 0.778 | 0.640 |
| episode_return | -5.979 | +0.226 |

The degenerate "crash immediately" policy predicted beforehand did **not** appear. The
reason is a sign flip: that prediction assumed a net-negative reward stream, under which
removal is a free escape. Once the policy is good enough to collect goals, the stream
turns positive (`episode_return = +0.226`) and removal instead forfeits all remaining
goal reward — it becomes the harshest penalty in the environment rather than an exit.
The transition is self-reinforcing, because collisions remove both vehicles, so early
crashes thin the traffic and make driving profitable sooner.

**The attribution is not settled.** Run F changes three things at once: collisions become
effectively terminal, metric semantics change, and traffic density collapses mid-episode.
The completion/score gain could be a genuinely better driving policy or simply an easier
task.

### Run G — remove at 300 steps, cold

`completion 0.755 ± 0.025`, `score 0.736 ± 0.030`, six blocks spanning 0.749–0.763 and
0.724–0.744 — the tightest run in the study. Cold-started, it reaches in one stage what
the baseline curriculum needed three stages and 600-step episodes to approach
(run C: `completion 0.765`, `score 0.652`).

Two things this does **not** establish:

- `completion_rate` rises mechanically with episode length under baseline behaviours too
  (A@100 `0.191` → B@200 `0.426` → C@600 `0.765`). Run G's `0.755` at 300 steps sits on
  that curve; there is no 300-step baseline-behaviour run to compare against.
- Traffic thinning is worse at 300 steps than at 100. `collisions_per_agent = 0.109`
  means only 11% of agents are ever removed, and removal is concentrated early, so most
  of a 300-step episode is driven on a near-empty map.

~~**`lane_alignment_rate` is the one clear cost.**~~ **Retracted.** The logged `0.400`
is the section-2 metric bias, not lane-keeping degradation: run G completes 4x more
often than run A, so 4x more of its agents are frozen out early. Measured over live
agent-steps in the cross-evaluation below, run G is the *most* lane-aligned policy in
the study (86.4%). The off-network-driving hypothesis is also refuted: only 1.1% of
run G's live steps have `lane_valid == 0`.

**Proposed test (cheap, no retraining):** evaluate the run F and run G checkpoints in the *baseline*
environment (`collision_behavior = 0`, `offroad_behavior = 0`, everything else as run A)
and compare against run A's numbers. Same traffic density, same metric semantics.
If they still win, `remove` is a useful training technique; if they fall back to the
baseline level, the advantage was the easier environment. `lane_alignment_rate` under
full traffic is the sharpest single number to look at.

---

## 8. Cross-evaluation (2026-08-21)

Three checkpoints rolled out in **one identical environment** — baseline behaviours
(`collision_behavior = 0`, `offroad_behavior = 0`), 300-step episodes, full traffic,
`goal_behavior = 3`, 256 agents over 500 maps, seed 11, one episode each. No training.

| policy | completion | score | coll/agent | offroad/agent | lane_align (logged) | **lane-aligned, live steps** | **off lane graph** |
|---|---|---|---|---|---|---|---|
| A (100, baseline) | 0.296 | 0.258 | 1.758 | 0.480 | 0.684 | 77.0% | 13.0% |
| C (600, curriculum) | 0.612 | 0.609 | 1.992 | 0.453 | 0.653 | 82.8% | 5.2% |
| **G (300, remove)** | **0.809** | **0.727** | 5.473 | 5.285 | 0.556 | **86.4%** | 6.5% |

### Findings

1. **Run G's advantage is the policy, not an easier environment.** In the same full-traffic
   environment it beats the three-stage curriculum endpoint on completion (0.809 vs 0.612)
   and on score (0.727 vs 0.609). The confound raised in section 7 does not survive.

2. **The `lane_alignment_rate` bias is confirmed and inverts the ranking.** Logged value
   falls monotonically with completion (A 0.684 → C 0.653 → G 0.556) while the live-step
   measurement rises (77.0% → 82.8% → 86.4%). Every cross-run comparison of this metric
   elsewhere in this document is unreliable for the same reason.

3. **Run G has no failure-recovery behaviour.** Collision frames 5.473 vs C's 1.992;
   off-road frames 5.285 vs C's 0.453 — 11.7x. Its `score = 0.727` against
   `completion = 0.809` means only ~27% of its agents ever touch anything, so it fails
   rarely — but when it does it stays stuck for tens of frames. Trained under
   `remove`, it never experienced a state after contact, so it has no policy for one.
   Irrelevant for a completion/score benchmark; a real problem for transfer.

### Limitation

256 agent-episodes per policy, one seed, one 300-step rollout each — far less data than
the training logs. The ordering (0.809 / 0.612 / 0.296) is far larger than plausible
noise, but the individual figures should not be read as precise.

### Script

`scratchpad/lanecheck.py` — loads a checkpoint via `pufferl.load_policy`, rolls out with
`enable_debug_trace()`, and counts `lane_valid` / `lane_aligned` over agent-steps where
`x > -9999` (i.e. excluding removed and finished agents), plus the env log at truncation.

---

## 9. Run H — the matched control (2026-08-21)

Identical to run G in every respect (300-step episodes, cold start, full conditioning,
`goal_behavior = 3`, same budget) except `collision_behavior` and `offroad_behavior`,
which are back to `0` (ignore). This isolates the removal behaviour as a single variable.

| metric | H (ignore/ignore) | G (remove/remove) | ratio |
|---|---|---|---|
| completion_rate | 0.189 | **0.755** | **4.0x** |
| score | 0.160 | **0.736** | **4.6x** |
| collisions_per_agent | 3.091 (frames) | 0.109 (events) | — |
| offroad_per_agent | 1.641 (frames) | 0.038 (events) | — |
| dnf_rate | 0.579 | 0.267 | — |
| episode_return | -9.192 | +0.514 | — |

Completion over six non-overlapping 100-flush blocks:

| block | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| H | 0.205 | 0.197 | 0.171 | 0.173 | 0.178 | 0.216 |
| G | 0.751 | 0.754 | 0.763 | 0.754 | 0.761 | 0.749 |

Ranges 0.171–0.216 and 0.749–0.763. No overlap, not remotely close. **This is the
largest and cleanest single-variable effect measured in the study.**

### Conclusion

Setting `collision_behavior = 2` and `offroad_behavior = 2` is worth **4x completion
and 4.6x score** at 300-step episodes, everything else held fixed. Combined with the
cross-evaluation in section 8 — where run G's policy also beats the three-stage
baseline curriculum in a *full-traffic, no-removal* environment (completion 0.809 vs
0.612, score 0.727 vs 0.609) — the effect is a property of the learned policy, not of
an easier training environment.

The mechanism is credit assignment on the collision signal. Under `ignore`, a collision
costs `-(alpha + 0.1v)` per frame of contact, clipped to `-1`, and the agent drives on.
Under `remove` it forfeits the entire remaining episode including the `+1` goal reward.
The penalty is two orders of magnitude larger and, more importantly, unambiguous — one
event, one consequence, no credit to spread over a contact lasting tens of frames.

### The one real cost

Run G has **no failure-recovery behaviour** (section 8, finding 3): put back in an
environment where contact does not end the episode, it accumulates 5.473 collision
frames and 5.285 off-road frames against run C's 1.992 and 0.453. It rarely fails, but
it cannot get out of a failure, because it never saw a post-contact state in training.

For a completion/score benchmark this does not matter. For transfer it does. A plausible
fix is a curriculum on the behaviour flag itself — train under `remove` to get the
avoidance policy cheaply, then fine-tune under `ignore` so the policy learns recovery
without losing what it learned. **Untested.**

---

## 10. Episode length under `remove` (2026-08-21)

All cold-started, full conditioning, `goal_behavior = 3`, same 2e9 budget. Figures are
means of the last 100 **unique** readings.

| run | episode | behaviors | completion | score | coll/agent | offroad/agent | goals_reached |
|---|---|---|---|---|---|---|---|
| A | 100 | ignore | 0.185 | 0.164 | 2.171 | 0.657 | 0.200 |
| F | 100 | **remove** | 0.458 | 0.534 | 0.095 | 0.029 | 0.619 |
| H | 300 | ignore | 0.190 | 0.161 | 2.964 | 1.647 | 0.205 |
| **G** | **300** | **remove** | **0.756** | **0.736** | 0.109 | 0.038 | 1.279 |
| I | 600 | **remove** | 0.637 | 0.549 | 0.181 | 0.143 | 0.956 |
| *C* | *600* | *ignore* | *0.754* | *0.643* | *3.695* | *0.614* | *1.251* |
| J | 1200 | **remove** | 0.287 | 0.255 | 0.267 | 0.271 | 0.363 |

*Run C is warm-started through the 100 → 200 → 600 curriculum; every other row is cold.*

### Findings

1. **300 steps is the optimum of the four tested, and the peak is sharp.** Under cold
   start with `remove`, completion goes `0.458` (100) → **`0.756`** (300) → `0.637` (600)
   → `0.287` (1200). Score has the same shape: `0.534` → `0.736` → `0.549` → `0.255`.
   Run J at 1200 steps — the length `cc4fb56b` actually trains at (1280) — recovers less
   than 40% of run G's completion on the same budget.

2. **This corrects an earlier inference.** Section 7 read the baseline sequence
   `0.191 / 0.426 / 0.765` at 100 / 200 / 600 as an episode-length effect and used it to
   discount run G's completion. That sequence is the *curriculum* — B and C were
   warm-started. Cold-started, completion is not monotone in episode length, so the
   discount was wrong and run G's result was better than section 7 credited.

3. **`remove`'s advantage shrinks as episodes lengthen.** Matched cold pairs:
   2.5x at 100 steps (A → F), 4.0x at 300 (H → G). At 600 there is no cold baseline to
   pair with, but run I (`0.637`) does not beat the warm curriculum's run C (`0.754`),
   whereas run G at 300 beats it while costing a third of the compute.

4. **The safety advantage persists at every length.** Run I still ends at 0.181 collision
   events and 0.143 off-road events per agent, against run C's 3.695 and 0.614 frames.

### Why long episodes fail cold

Run J's learning curve is the diagnostic: completion `0.109` (6% of budget) → `0.119`
(26%) → `0.156` (46%) → `0.247` (66%) → `0.287` (final). It spends the first quarter of
its budget going nowhere. Over the same span `off-road events per agent` falls `0.467` →
`0.400` → `0.322` → `0.235`, so what it is doing in that quarter is learning not to leave
the road at all.

The mechanism is that `remove` and long episodes interact badly at initialization. A
cold policy leaves the road within tens of steps, and removal ends its episode there, so
the *effective* episode is short no matter what `episode_length` says — but the budget is
still charged for the full nominal length. The longer the nominal episode, the larger
the fraction of the rollout buffer that is spent on agents that are already frozen.
`remove` converts a strong learning signal at 300 steps into premature termination at
1200.

This also predicts that a curriculum on episode length should rescue the long setting,
since the cost is entirely in the cold-start phase. Untested.

### Caveat

`I` vs `C` is not a clean single-variable comparison — it differs in both the behaviour
flag and cold vs curriculum start. A cold `ignore` run at 600 steps would close that gap;
it has not been run.
