# PI05 基于观测门控的推理复用方案

## 1. 目标

本文档针对当前 `pi05` 具身控制闭环提出一条务实的优化路径：

- 保持传感器采集高频运行
- 降低高成本 VLA 推理的触发频率
- 当观测变化很小时复用上一轮推理结果
- 在状态突变或接近障碍物时强制重规划，保证安全性

该方案面向边端或机器人端部署场景，假设单机器人、`batch=1`，而不是 `vLLM` 那类高并发 serving 场景。

## 2. 为什么选这个方向

### 2.1 当前测到的瓶颈

根据最近 `call_pi05_server.py` 的计时日志：

| 阶段 | 典型耗时 |
| --- | --- |
| Observation 获取 | `23-34 ms` |
| Server 推理 + RPC | `344-360 ms` |
| 传输 / RPC 额外开销 | `5-10 ms` |
| Cmd 发送 | `0.14-0.33 ms` |
| Cmd 执行保持 + settle | `150 ms` |
| 单轮总耗时 | `520-540 ms` |

很明显，瓶颈在模型推理，不在 observation 获取，也不在命令发送。

### 2.2 这不是 `vLLM` 类问题

在当前具身场景里：

- 通常只有一台机器人或极少量机器人
- 控制闭环强时序、强顺序依赖
- `batch=1` 是常态
- 相邻时刻的观测高度时序相关

因此，第一优先级应该是减少模型调用次数，而不是去优化多租户并发吞吐。

## 3. VLA-Perf 给出的关键启发

这个方案和论文 `How Fast Can I Run My VLA? Demystifying VLA Inference Performance with VLA-Perf` 一致，该论文对应 `arXiv:2602.18397v1`，发布日期是 2026 年 2 月 20 日。

对当前系统最相关的几点结论是：

1. 降低 denoising steps 的收益，远大于缩小 action chunk size。
2. Action chunking 的价值在于减少推理调用频率。
3. 异步执行可以通过重叠推理、执行和通信来提升系统吞吐。
4. 双系统设计的核心是降低昂贵模型的调用频率，同时让一个更便宜的快环高频响应。
5. 机器人系统要看端到端性能，不能只盯着模型本身。

对于当前 `pi05` 链路，这些结论可以翻译成一句话：

- 不要执着于每一帧 camera 都跑一次 VLA
- 可以接受有边界的 observation 陈旧
- 当场景和机器人状态变化不大时，应优先复用已有动作

## 4. 当前系统的观察

### 4.1 现有控制闭环结构

`examples/tutorial/pi05/call_pi05_server.py` 当前的大致流程是：

1. 加载 observation
2. 把完整 observation 发给推理 server
3. 接收 action chunk
4. 取第一个 action 转成 `cmd_vel`
5. 发送命令
6. 保持 `execution_horizon_sec`
7. 可选地等待 `execution_settle_sec`
8. 进入下一轮

### 4.2 一个重要实现细节

在 `examples/tutorial/pi05/pi05_runtime.py` 中，server 当前是先调用 `predict_action_chunk(...)`，然后再把结果裁成 `requested_actions`。

这意味着当 `actions_per_chunk=1` 时，系统很可能仍然支付了完整 chunk 的推理代价，但只消费了第一个动作。在引入更复杂的复用逻辑前，这个问题本身就值得先利用或修正。

### 4.3 最近相似度日志反映了什么

最近的 CSV 分析说明：

- 相邻轮次的命令变化经常不大
- 但仅靠图像相似度来决定是否复用，并不可靠
- 图像相似度高时，命令仍可能发生明显变化

当前日志里已经观察到的例子包括：

- 某些相邻轮次 `image_similarity_corr >= 0.9`，但 `cmd_delta_l2 > 0.1`
- 甚至有一次 `image_similarity_corr = 1.0` 且 `image_mean_abs_diff = 0.0`，但命令变化依然不可忽略

所以，基于图像的精确 cache 风险太高。

## 5. 建议的系统架构

### 5.1 核心思路

采用一种“基于观测门控的控制闭环”：

- 每轮都感知
- 只在必要时推理
- 其他时候复用已有动作

这不是精确 memoization，而是一种“有边界陈旧度”的重规划控制策略。

### 5.2 双速率系统

可以把系统逻辑上拆成两个速率：

- 快环：
  - 采集 camera / lidar / proprioception
  - 计算廉价的变化指标
  - 执行缓存动作或缓存命令
- 慢环：
  - 运行 `pi05` 推理
  - 刷新 action chunk 缓存

### 5.3 复用优先级

建议的复用顺序是：

1. 优先消费上一轮推理得到但尚未执行完的 action chunk
2. 如果 chunk 已经耗尽，但场景依然稳定，则短暂复用上一轮最终 `cmd_vel`
3. 当变化超过阈值，或触发安全事件时，强制重新推理

这比长时间重复发送同一个 `cmd_vel` 更稳妥。

## 6. 决策信号

复用决策必须基于多模态信号，而不是只看图像相似度。

### 6.1 Camera 信号

保留当前已实现的轻量图像指标：

- `image_similarity_l1`
- `image_similarity_corr`
- `image_mean_abs_diff`

这些指标适合作为粗粒度变化检测器，但不足以单独决定是否复用。

### 6.2 State 信号

需要为 `observation.state` 增加状态变化指标：

- `state_delta_l2`
- `state_max_abs_diff`
- 可选的分组 delta：
  - joint position delta
  - joint velocity delta
  - IMU delta
  - contact delta count

这一步很关键，因为策略本身会把 state 一并作为输入。

### 6.3 雷达 / 障碍信号

对于 locomotion 场景，障碍变化的优先级应高于图像相似度：

- `front_obstacle_distance_delta_m`
- 如果后续接入更多激光雷达信息，可以继续加 lidar sector delta

当障碍物快速靠近时，不应跳过推理。

### 6.4 安全触发条件

下面这些条件应直接触发重新推理：

- 第一轮
- 当前没有可用缓存 chunk
- 障碍距离跨越 stop 或 slow 阈值
- state delta 超过阈值
- 连续复用次数超过上限
- 达到周期性强制刷新时间

## 7. 建议的控制策略

### 7.1 最小可行版本

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

### 7.2 初版阈值应当保守

第一版建议故意保守：

- 最多允许连续跳过 `1-2` 轮推理
- 接近障碍物时强制重规划
- 每隔固定 wall-clock 时间强制重推一次
- 宁可误判为“不复用”，也不要误判为“可以复用”

换句话说，只有在系统非常确定场景稳定时，才允许跳过推理。

## 8. 实施路线

### Phase 0：先把可观测性补齐

在真正启用复用前，先增加这些日志：

- `state_delta_l2`
- `state_max_abs_diff`
- `front_obstacle_distance_delta_m`
- `reuse_decision`
- `reuse_reason`
- `reused_cmd_count`
- `reused_chunk_action_index`

同时打印到控制台和 metrics CSV。

### Phase 1：先把 action chunk 真正用起来

在任何跳推理逻辑之前，先做下面几件事：

- 提高 `action_chunk` 的真实利用率
- 避免系统支付了完整 chunk 的推理代价，却只消费第一个动作
- 验证动作平滑性和闭环稳定性

这是当前风险最低、收益最直接的一步。

### Phase 2：保守版 observation-gated reuse

增加一个特性开关，例如：

- `--reuse-when-static`

并增加一组阈值参数，例如：

- `--reuse-image-corr-threshold`
- `--reuse-image-mad-threshold`
- `--reuse-state-l2-threshold`
- `--reuse-front-distance-delta-threshold`
- `--max-reused-cycles`
- `--force-replan-interval-sec`

第一版建议只有在所有启用的阈值同时满足时，才允许复用。

### Phase 3：引入异步重叠

当门控复用已经稳定后，再做下一步：

- 在当前 action chunk 执行期间启动下一次推理
- 尽可能重叠通信和计算
- 降低动作执行结束到下一次决策之间的空转时间

这和论文里的 asynchronous inference 方向一致，但实现顺序应放在保守版复用之后。

### Phase 4：进一步做模型级降本

在系统级门控稳定后，再考虑优化模型本身：

- 降低 `num_inference_steps`
- 评估更低精度或量化
- 检查 `past_key_values` copy 的额外开销
- 评估更小 policy 变体是否可接受

## 9. 预期收益

按当前实测闭环来看：

- 每跳过一次推理，大约可以节省 `~350 ms`
- observation 和 cmd 的额外开销已经很小

粗略估计如下：

| 推理跳过命中率 | 估计平均单轮耗时 | 估计控制频率 |
| --- | --- | --- |
| `0%` | `~530 ms` | `~1.9 Hz` |
| `30%` | `~425 ms` | `~2.35 Hz` |
| `50%` | `~355 ms` | `~2.8 Hz` |

这些数字只是粗估，默认当前的 `hold` 和 `settle` 参数不变。

## 10. 评估方案

### 10.1 离线分析

利用已经记录的 CSV 轨迹估计：

- 门控策略大概能跳过多少轮推理
- 被跳过的轮次里，命令是否大概率仍然相似
- 这些跳过候选样本是否集中在安全或不安全的障碍距离区间

### 10.2 在线 A/B 测试

对比以下两种模式：

- baseline：每轮都推理
- reuse-v1：保守版门控复用

重点跟踪：

- 平均 cycle time
- 有效控制频率
- 每分钟推理调用次数
- skip hit rate
- forced replan count
- emergency stop count
- 定性动作平滑性

### 10.3 安全性复核

需要重点检查这些场景：

- 障碍物快速靠近
- 急转向
- 偏航后的恢复
- 打滑或 contact 状态突变

如果复用策略在这些场景下表现不稳，就必须先收紧阈值，再考虑扩大使用范围。

## 11. 风险与缓解

### 风险 1：图像相似导致误判

问题：

- 图像看起来很像，但状态或障碍几何关系已经变了

缓解：

- 除图像阈值外，必须同时要求 state 和 obstacle 阈值满足

### 风险 2：模型本身存在随机性

问题：

- 对于相似输入，扩散式推理仍可能产生略有差异的动作

缓解：

- 把复用看成控制策略，而不是精确输出 cache
- 评估重点放在闭环稳定性，而不是单步命令距离

### 风险 3：复用过多导致行为陈旧

问题：

- 机器人可能持续执行已经过时的意图

缓解：

- 限制最大连续复用轮数
- 增加周期性强制刷新
- 当障碍距离快速缩小时强制重规划

### 风险 4：chunk 复用时缺少反馈修正

问题：

- 更长时间的开环执行会降低系统响应性

缓解：

- 从短 horizon 开始
- 即使在复用动作时，也要每轮重新检查 observation

## 12. 建议的下一步

当前最值得马上做的事是：

1. 在 `call_pi05_server.py` 中增加 state 和 obstacle delta 指标
2. 把复用相关阈值暴露成 CLI 参数
3. 在 feature flag 下实现“优先复用 chunk”的逻辑
4. 用离线日志先做阈值筛选
5. 做一轮小规模、以安全为优先的在线验证

## 13. 最终结论

对于当前 `pi05` 具身系统，近期最值得做的优化，不是高并发 serving，而是通过基于观测门控的复用来降低推理频率。

正确的原则应该是：

- 每轮都感知
- 只在必要时推理
- 谨慎复用
- 在安全边界附近积极重规划

在更深层的模型和系统重构之前，这是提升实时性的最务实路径。
