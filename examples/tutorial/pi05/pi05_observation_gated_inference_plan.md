# PI05 Observation-Gated Inference Reuse Plan

## 1. Goal

This document proposes a practical optimization path for the current `pi05` embodied control loop:

- keep sensor acquisition at high frequency
- reduce expensive VLA inference frequency
- reuse prior inference results when observations change only slightly
- preserve safety by forcing replan on large state changes or near-obstacle events

The target scenario is edge or robot-side deployment with a single robot and `batch=1`, not a high-concurrency serving setup like `vLLM`.

## 2. Why This Direction

### 2.1 Current measured bottleneck

Based on the recent `call_pi05_server.py` timing logs:

| Stage | Typical time |
| --- | --- |
| Observation load | `23-34 ms` |
| Server inference + RPC | `344-360 ms` |
| Transport / RPC overhead | `5-10 ms` |
| Cmd send | `0.14-0.33 ms` |
| Cmd execution hold + settle | `150 ms` |
| Total cycle | `520-540 ms` |

The bottleneck is clearly model inference, not observation capture or command transmission.

### 2.2 This is not a `vLLM` problem

In the current embodied setting:

- there is usually only one robot or a small number of robots
- the control loop is strongly sequential
- `batch=1` is the normal case
- adjacent observations are temporally correlated

So the first optimization target should be reducing how often we invoke the model, not improving multi-tenant throughput.

## 3. Key Insights From VLA-Perf

This plan is consistent with the paper `How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf` (`arXiv:2602.18397v1`, February 20, 2026).

The most relevant takeaways for the current system are:

1. Reducing denoising steps matters much more than shrinking action chunk size.
2. Action chunking is valuable because it reduces how often inference must run.
3. Asynchronous execution can improve throughput by overlapping inference and execution or communication.
4. Dual-system designs work by lowering the expensive model invocation rate and letting a cheaper fast loop react more often.
5. For real robots, end-to-end performance must be considered jointly across sensing, inference, transport, and execution.

For the current `pi05` setup, the practical interpretation is:

- do not insist on running VLA for every camera frame
- accept bounded observation staleness
- reuse prior actions when the scene and robot state have not changed enough to justify a new inference

## 4. Current System Observations

### 4.1 Existing loop structure

The current loop in `examples/tutorial/pi05/call_pi05_server.py` is approximately:

1. load observation
2. send full observation to inference server
3. receive action chunk
4. convert first action to `cmd_vel`
5. send command
6. hold for `execution_horizon_sec`
7. optionally settle for `execution_settle_sec`
8. repeat

### 4.2 Important implementation detail

In `examples/tutorial/pi05/pi05_runtime.py`, the server currently calls `predict_action_chunk(...)` and only then slices the returned result to `requested_actions`.

That means with `actions_per_chunk=1`, the system is likely still paying for full chunk inference while consuming only the first action. This should be fixed or at least exploited before introducing more complex reuse logic.

### 4.3 What recent similarity logs showed

Recent CSV analysis showed:

- command changes are often small across adjacent cycles
- image similarity alone is not reliable enough to decide reuse
- high image similarity can still coincide with noticeable command change

Examples observed in the current logs:

- some transitions had `image_similarity_corr >= 0.9` while `cmd_delta_l2 > 0.1`
- one transition even had `image_similarity_corr = 1.0` and `image_mean_abs_diff = 0.0`, but command change was still non-trivial

So an image-only cache is too risky.

## 5. Proposed Architecture

### 5.1 Core idea

Use an observation-gated control loop:

- sense every cycle
- infer only when needed
- otherwise reuse an already available action

This is not exact memoization. It is bounded-staleness replan control.

### 5.2 Two-rate system

Split the loop into two logical rates:

- fast loop:
  - capture camera / lidar / proprioception
  - compute cheap similarity metrics
  - execute cached action or cached command
- slow loop:
  - run `pi05` inference
  - refresh the action chunk cache

### 5.3 Reuse priority

The reuse order should be:

1. consume remaining actions from the last inferred action chunk
2. if the chunk is exhausted but the scene is still stable, briefly reuse the last final `cmd_vel`
3. force re-inference when change exceeds thresholds or safety events occur

This is safer than reusing the same `cmd_vel` immediately for many cycles.

## 6. Decision Signals

Reuse decisions should be based on multiple modalities, not image similarity alone.

### 6.1 Camera signals

Keep the current cheap image metrics:

- `image_similarity_l1`
- `image_similarity_corr`
- `image_mean_abs_diff`

These are useful as coarse change detectors, but not sufficient by themselves.

### 6.2 State signals

Add state-based metrics for `observation.state`:

- `state_delta_l2`
- `state_max_abs_diff`
- optional per-group deltas:
  - joint position delta
  - joint velocity delta
  - IMU delta
  - contact delta count

This is critical because the policy consumes state as part of the prompt.

### 6.3 Radar / obstacle signals

For locomotion, obstacle-related changes must have higher priority than image similarity:

- `front_obstacle_distance_delta_m`
- optional lidar sector deltas if available later

When an obstacle gets closer quickly, inference should not be skipped.

### 6.4 Safety triggers

The following should force re-inference immediately:

- first cycle
- no cached chunk available
- obstacle distance crosses a stop or slow boundary
- state delta exceeds threshold
- command reuse count exceeds limit
- periodic refresh timeout reached

## 7. Proposed Control Policy

### 7.1 Minimal viable policy

```text
for each control cycle:
    obs = load_observation()
    change = compute_change_metrics(obs, prev_obs)

    if first_cycle:
        run_inference()
        cache_action_chunk()
        execute_next_action()
        continue

    if safety_triggered(change) or force_refresh():
        run_inference()
        cache_action_chunk()
        execute_next_action()
        continue

    if observation_is_stable(change) and cached_chunk_has_next():
        execute_cached_next_action()
        continue

    if observation_is_stable(change) and can_briefly_reuse_last_cmd():
        execute_last_cmd()
        continue

    run_inference()
    cache_action_chunk()
    execute_next_action()
```

### 7.2 Initial conservative thresholds

The first version should be intentionally conservative:

- allow at most `1-2` consecutive skipped inference cycles
- force replan near obstacles
- force replan every fixed wall-clock interval
- prefer false negatives over false positives

In other words, skip only when the system is very confident the scene is stable.

## 8. Implementation Roadmap

### Phase 0: Better observability

Before enabling reuse, add the following logs:

- `state_delta_l2`
- `state_max_abs_diff`
- `front_obstacle_distance_delta_m`
- `reuse_decision`
- `reuse_reason`
- `reused_cmd_count`
- `reused_chunk_action_index`

Output them both in console and the metrics CSV.

### Phase 1: Fully use action chunks

Before any skip logic:

- increase real usage of `action_chunk`
- avoid paying full chunk inference cost but consuming only one action
- verify action smoothness and closed-loop stability

This is the lowest-risk gain.

### Phase 2: Conservative observation-gated reuse

Add a feature flag such as:

- `--reuse-when-static`

And thresholds such as:

- `--reuse-image-corr-threshold`
- `--reuse-image-mad-threshold`
- `--reuse-state-l2-threshold`
- `--reuse-front-distance-delta-threshold`
- `--max-reused-cycles`
- `--force-replan-interval-sec`

The first version should only reuse if all enabled thresholds pass.

### Phase 3: Asynchronous overlap

Once gated reuse is stable:

- start next inference while executing current action chunk
- overlap communication and compute where possible
- reduce idle time between action execution and next decision

This follows the paper's asynchronous inference direction, but should come after conservative reuse.

### Phase 4: Model-level reduction

After system-level gating is in place, optimize model cost:

- reduce `num_inference_steps`
- consider lower precision / quantization
- inspect `past_key_values` copy overhead
- evaluate whether smaller policy variants are acceptable

## 9. Expected Benefit

With the current measured loop:

- each skipped inference can save roughly `~350 ms`
- observation and command overheads are already small

Simple estimates:

| Inference skip hit rate | Approx. average cycle time | Approx. control frequency |
| --- | --- | --- |
| `0%` | `~530 ms` | `~1.9 Hz` |
| `30%` | `~425 ms` | `~2.35 Hz` |
| `50%` | `~355 ms` | `~2.8 Hz` |

These are rough estimates and assume the current execution hold and settle timings remain unchanged.

## 10. Evaluation Plan

### 10.1 Offline analysis

Use recorded CSV traces to estimate:

- how often the gate would skip inference
- whether skipped cycles would likely have produced similar commands
- whether skip candidates cluster around safe or unsafe obstacle regimes

### 10.2 Online A/B test

Compare:

- baseline: infer every cycle
- reuse-v1: conservative gated reuse

Track:

- average cycle time
- effective control frequency
- inference calls per minute
- skip hit rate
- forced replan count
- emergency stop count
- qualitative smoothness

### 10.3 Safety review

Specially inspect cases with:

- fast obstacle approach
- sharp turns
- recovery from drift
- slips or contact changes

Any reuse strategy that fails in these cases should be tightened before broader use.

## 11. Risks and Mitigations

### Risk 1: Image-only false positives

Problem:

- the image may look similar while state or obstacle geometry has changed

Mitigation:

- require state and obstacle thresholds in addition to image thresholds

### Risk 2: Model stochasticity

Problem:

- diffusion inference may produce slightly different actions even for similar inputs

Mitigation:

- treat reuse as a control policy decision, not exact output caching
- evaluate closed-loop stability, not only one-step command distance

### Risk 3: Over-reuse causes stale behavior

Problem:

- the robot may continue executing outdated intentions

Mitigation:

- cap maximum consecutive reuse cycles
- force periodic refresh
- force replan when obstacle distance decreases quickly

### Risk 4: Chunk reuse without feedback correction

Problem:

- longer open-loop execution reduces responsiveness

Mitigation:

- start with short horizons
- re-check observation every cycle even when reusing actions

## 12. Recommended Next Steps

The immediate next steps should be:

1. add state and obstacle delta metrics to `call_pi05_server.py`
2. expose reuse-related thresholds as CLI flags
3. implement chunk-first reuse under a feature flag
4. run offline threshold selection on recorded traces
5. run a short online safety-first trial

## 13. Bottom Line

For the current `pi05` embodied system, the most promising near-term optimization is not high-concurrency serving, but reducing inference frequency through observation-gated reuse.

The correct principle is:

- sense every cycle
- infer only when needed
- reuse cautiously
- replan aggressively near safety boundaries

This is the most practical path to improve real-time behavior before deeper model or systems refactoring.
