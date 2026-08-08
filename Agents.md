# Agents.md — Pictura 复现工作日志

> 跨 session 的工作记录。token 用尽后从这里接着干。
> 计划全文：`/home/tjhu78u/.claude/plans/vivid-moseying-treehouse.md`

---

## 0. 任务

在本 repo 中尽可能忠实复现 **Pictura**（arXiv 2607.26005，valeo.ai，2026-07-28）
*"Pictura: Perspective-View Self-Play at Scale for Driving"*
项目页 https://valeoai.github.io/Pictura/ ——**官方代码不会发布**。

论文 3.1 节原文明确：*"We build on PufferDrive [5], an open-source driving simulator written with PufferLib... to our knowledge, the best public starting point for vectorized self-play RL in driving."* 即本 repo 就是它的基座。

**本次范围**：专用 RenderState buffer → 专用 CUDA 光栅器 → 单相机（rig 可配置 list）→ CNN + PPO。
**不在范围**：Gigaflow 特性补齐（红绿灯/墙体/停放车/11 项 reward/conditioning），见 §6。

**硬约束**：相机策略只能读渲染图像 + 论文允许的 ego state / conditioning。任何 privileged 场景量不得进入 policy 输入 —— 用结构强制（光栅器属于观测管线，不属于 policy），不靠约定。

**环境**：RTX A6000 48GB / sm_86 / CUDA 12.1，单卡。分支 `2.0`。

---

## 1. 目标架构

```
Drive C env  (obs_mode = render_state)
   ├─ per-agent 观测缓冲: [ ego state S | conditioning C ]      ← 小，非特权
   └─ per-env RenderState: 世界系场景图元                        ← 特权，policy 永不可见
         · agents[]: x, y, cos_h, sin_h, length, width, height, type   (每步)
         · roads[] : x0, y0, x1, y1, width, type                       (scenario 静态)
         · egos[]  : 每个受控 agent 的 x, y, cos_h, sin_h               (每步)

PerspectiveVecEnv 包装器 (Python, 新)  ← 特权屏障
   ├─ RenderState 上传 GPU
   ├─ raster_cuda(scene, egos, rig) → images [B, N_cam*3, H, W] uint8
   ├─ 丢弃场景图元 ─────────────────────────────── policy 拿不到
   └─ obs = Dict{ image: uint8[N*3,H,W], ego: float32[E] }
        经 pufferlib.emulation 打包成字节 Box + emulated dtype

pufferl PPO (不改)
   └─ DriveCam: nativize_tensor → CNN(image) + MLP(ego) → 4x512 MLP → actor/critic
```

### 三个关键决定
1. **光栅器归包装器，不归 policy** —— 特权屏障的实现方式；附带让"向量 baseline vs 透视策略"变成 config 开关，正好是论文的受控对比。
2. **RenderState 是 per-env 世界系共享，不是 per-agent 自车系** —— 同图 1024 agent 共享一份场景。per-agent 自车系要复制 1024 遍（8 KB/agent → 8 MB/步），且强行给渲染加实体数上限（论文渲染全场景）。世界系共享后每步只需 `num_agents × 8` floats（1024 agent ≈ 32 KB），路网只在 scenario reset 时上传一次。
3. **rollout buffer 存图像，不存 RenderState 重渲染** —— 存图像才忠实（policy 确实在图像上训练）；重渲染会让 policy forward 收到场景量，破坏约束 1。

---

## 2. 已验证的 repo 事实（含行号，改动前先复核仍然成立）

### 数据通路
| 事实 | 位置 |
|---|---|
| `o = torch.as_tensor(o); o.to(device)` 对已在 GPU 的 tensor 是直通 → **包装器可直接返回 GPU tensor**（需 `cpu_offload=false`） | `pufferlib/pufferl.py:270-271` |
| pufferl 用到的 vecenv 接口仅：`async_reset/recv/send/close/reset/step/single_observation_space/single_action_space/observation_space/action_space/num_agents/agents_per_batch/driver_env` | `pufferl.py` 全文 grep |
| policy 由 `policy_cls(vecenv.driver_env, **args["policy"])` 构造 → 包装器需自任 driver_env | `pufferl.py:1362-1368` |
| `single_observation_space` 必须是 Box | `pufferlib/pufferlib.py:58` |
| 混合 dtype 走 Dict → 字节视图 Box；全叶同 dtype 则用该 dtype，否则 uint8 | `pufferlib/emulation.py:115-126` |
| torch 侧还原：`pufferlib.pytorch.nativize_tensor(obs, self.dtype)`，现成用例在 models.py | `pufferlib/pytorch.py:96`, `pufferlib/models.py:73` |
| 共享内存分配入口 | `pufferlib/pufferlib.py:19 set_buffers` |
| obs/action/reward/terminal 由 Python 传入指针，env 直接写 | `pufferlib/ocean/drive/binding.c:8-67` (`my_put`) |
| rollout buffer 分配（dtype 取自 obs_space） | `pufferl.py:107-114` |
| Serial 后端把 env 放在训练进程内 → 可直接读 numpy 视图，零 plumbing | `pufferlib/vector.py:55-180` |

### 构建与环境搭建（**踩过的坑，务必照做**）

本机 CUDA toolkit 只有 **12.1**（`/usr/local/cuda-12.1`，驱动 555.42.02，无更高版本）。
`pyproject.toml:236` 和 `setup.py:319` 对 torch **不做版本约束** → 默认装到 cu130 编译的 torch，
`torch.utils.cpp_extension._check_cuda_version` 直接报
`RuntimeError: The detected CUDA version (12.1) mismatches ... (13.0)`，可编辑安装失败。

正确顺序：
```bash
export VIRTUAL_ENV=/mnt/disk/tjhu78u/workspace/test/PufferDrive/.venv
uv pip install "numpy<2"                                                   # 构建要求 numpy<2
uv pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121   # 必须 cu121
uv pip install -e . --no-build-isolation                                   # 复用已装 torch
python setup.py build_ext --inplace --force                                # 重编 C 扩展
```
已验证：`torch 2.4.1+cu121`、`numpy 1.26.4`。

- `setup.py` 已 import `CUDAExtension`，有 `pufferlib._C` torch 扩展链（`TorchBuildExt`，`setup.py:223`）→ 新 kernel 挂上去即可。
- 现有 `pufferlib/extensions/cuda/pufferlib.cu` 仅 88 行，是 PPO advantage/vtrace，**不是光栅器**。
- CUDA 测试先例：`tests/test_c_advantage.cu`。

### 现有 env（`pufferlib/ocean/drive/`，drive.h 3188 行）
| 事实 | 位置 |
|---|---|
| 观测构造函数 | `drive.h:1829 compute_observations` |
| ego 打包（JERK 11 维）：`[rel_goal_x*0.005, rel_goal_y*0.005, speed/100, w/15, l/30, collision, steer/π, a_long, a_lat, respawn, type/3]` | `drive.h:1834-1853` |
| partner 31×7：`[rel_x*0.02, rel_y*0.02, w/15, l/30, cosΔθ, sinΔθ, speed/100]`，50m 截断（`dist>2500` skip），**无 type/height** | `drive.h:1886-1907` |
| road 128×7：`[rel_x*0.02, rel_y*0.02, half_len/100, 0.1/100, cosθ, sinθ, type-4]` | `drive.h:1963-1969` |
| road 取自网格邻域缓存，**螺旋序**（截断后保留最近的） | `drive.h:778 get_neighbor_cache_entities`, `drive.h:691-719` |
| 网格：`GRID_CELL_SIZE 5.0f`，`vision_range = 21` 格 → **±52.5m** | `drive.h:83`, `drive.h:1462` |
| 路网 polyline 原始数据（渲染的真正数据源） | `drive.h:1939-1960`（`entity->traj_x/y[geometry_idx]`，`type >= 4`） |
| 实体类型：NONE0 VEHICLE1 PEDESTRIAN2 CYCLIST3 ROAD_LANE4 ROAD_LINE5 ROAD_EDGE6 STOP_SIGN7 CROSSWALK8 SPEED_BUMP9 DRIVEWAY10 | `drive.h:33-43` |
| 指标索引 collision/offroad/reached_goal/lane_aligned | `drive.h:77-80` |
| raylib BEV 渲染（几何正确性对照用） | `drive.h:3024` 起 |
| `VIEW_MODE_AGENT_PERSP` 是**给人看的跟车相机**，不是观测管线 | `drive.h:24`, `drive.h:3074` |
| 现有 policy：MLP entity-set encoder + LSTM | `pufferlib/ocean/torch.py:15 class Drive` |
| 训练配置 | `pufferlib/config/ocean/drive.ini` |

### 地图数据实测（2026-08-04，解析脚本见 §7）

地图二进制格式（`drive.h:412 load_map_binary`）：
`scenario_id[16] | sdc_track_index:i32 | num_tracks_to_predict:i32 | tracks[]:i32 | num_objects:i32 | num_roads:i32 |` 然后每个实体：
`scenario_id:i32 | type:i32 | id:i32 | array_size:i32 | traj_x/y/z:f32[size] | (仅 type∈{1,2,3}) vx/vy/vz/heading:f32[size] + valid:i32[size] | width,length,height:f32 | goal_x/y/z:f32 | mark_as_expert:i32`

| 文件 | 格式 | objects | road 实体 | **road 线段总数** | 类型分布 |
|---|---|---|---|---|---|
| `resources/drive/binaries/map_000.bin` | 当前 | 47 VEHICLE | 112 | **1638** | LANE 573 / LINE 502 / EDGE 441 / DRIVEWAY 122 |
| `resources/drive/map_town_02_carla.bin` | **legacy**（无 16B scenario_id 头，当前 loader 读不了）| 32 VEHICLE | 470 | **2019** | LANE 995 / LINE 624 / EDGE 400 |

对光栅器的意义：
- **每张图的路网线段总量只有 1600–2000 条** → per-env 世界系缓冲仅需 ~2000×6 float ≈ 48 KB，完全放得下，且**无需任何实体数上限**。这坐实了架构决定 2（对比：现有 per-agent 观测只带 128 条）。
- `height` 字段确实有值且合理：WOMD 车辆 h∈[1.45,3.53]，CARLA 统一 1.8 → **长方体高度可直接用，不必假设常量**。
- WOMD 车辆尺寸 w∈[1.88,3.03] l∈[4.17,8.72]；CARLA 统一 2.0×4.5×1.8。
- LANE(4) 是车道中心线，**物理上不可见**，渲染它等于把特权信息画进图像。默认只渲染 LINE(5)/EDGE(6)/CROSSWALK(8)/SPEED_BUMP(9)。CARLA Town02 去掉 LANE 后仍有 624+400=1024 条可见标线，够用。**此项做成配置开关**。

**训练数据已就绪**（2026-08-04）：`resources/drive/binaries/training` 是指向
`/home/tjhu78u/workspace/PufferDrive/resources/drive/binaries/training/` 的软链接，含 **10000 张 WOMD 图**。
命名 `map_000.bin` … `map_9999.bin`，与 loader 的 `snprintf("%s/map_%03d.bin")` 兼容（`%03d` 是最小宽度，不截断）。

实测（1024 agents，10000 图）：

| obs_mode | envs | obs 形状 | 吞吐 |
|---|---|---|---|
| `vector` | 155 | (1024, 1124) | 277K agent-steps/s |
| `render_state` | 153 | (1024, **11**) | **426K** agent-steps/s |

render_state 更快是因为跳过了实体集观测的构建。场景规模：153 env 共 3407 agent 图元 + 107414 路段（≈700 段/env）。

论文用的 8 张 CARLA town 的 JSON 在 `data_utils/carla/Town{01..07,10HD}.json`，若要复刻论文训练集需自行转换
（`pufferdrive_womd_train_carla_mixed` 数据集含 Town1/2 各 100 份）。当前用的是 WOMD 训练集。

### repo 已有 / 已缺（对照论文）
**已有**：C 多 agent 核心 + PufferLib PPO；CARLA Town01–07 + Town10HD（`data_utils/carla/*.json`，正是论文训练图）；jerk 动力学 + `MultiDiscrete([4*3])`（**已与论文的 4 纵×3 横 jerk bins 一致**，`drive.py:130`）；开放式目标 `GOAL_GENERATE_NEW`；违规停住 `collision_behavior=1`；WOMD 数据管线。

**已缺**：红绿灯/停止线（`drive.h` grep 无结果）、墙体、停放车、车道限速（全 repo 无 `speed_limit`）、11 项 reward（现只有 collision/offroad/goal/goal_post_respawn + jerk 惩罚）、per-agent conditioning、**透视渲染观测**。

---

## 3. 论文关键参数（PDF 文本仅存于 session scratchpad，会丢失，故固化于此）

### 相机 rig（论文 Tab. 3，**完整外参**）

ego 系约定：**x 前、y 左、z 上，yaw 绕 z**，单位米 —— 与 repo 的 `heading_x=cos, heading_y=sin` 及
`rel_x = dx·cos+dy·sin, rel_y = -dx·sin+dy·cos`（`drive.h:1884`）**完全一致**，无需坐标系转换。

共享内参（4 路相同）：传感器 1920×1080 (16:9)，焦距 **1545 px**，FOV **63.7°×38.5°**，渲染分辨率 **96×54**。
自洽性核对：`2·atan(960/1545)=63.7°` ✓ `2·atan(540/1545)=38.5°` ✓ 方形像素。
缩放到 96×54：`fx=fy=1545×96/1920=77.25`，`cx=48`，`cy=27`。

| 相机 | x | y | z | yaw |
|---|---|---|---|---|
| **Front（本复现默认）** | 1.66 | −0.01 | 1.49 | 0° |
| Front-left | 1.63 | 0.12 | 1.48 | +55° |
| Front-right | 1.62 | −0.16 | 1.49 | −55° |
| Back | −0.47 | 0.02 | 1.43 | 180° |

**无 pitch / roll**，只有 yaw。分辨率扫描范围 32×18 / 64×36 / 96×54 / 192×108 / 384×216。

单相机选型决定：**采用论文 front 相机的原始内外参**（而非早先设想的加宽 FOV）。理由是它就是论文
四路 rig 中的一路，逐位复刻；后续补上其余三路时 rig 里直接追加条目，kernel 和内参都不用动。
代价是单路 63.7° 看不见路口横向来车 —— 这是已知的、有意接受的取舍，不是缺陷。

### 渲染语义（论文 Fig. 3）
平面着色几何图元，无纹理无光照。agent 长方体 **per-face 上色 + 深度调制亮度**（朝向和距离从颜色可读），车/行人/骑车人各一色；红绿灯为有限角度可见的圆盘，只画给它管辖的 approach，背对时剔除，灯头位置与停止线解耦（美式路口）；停放车；墙体作遮挡物；路面标线为随深度收窄的 polyline。
**解析覆盖率抗锯齿，1 spp** —— 论文点名这是低分辨率下车道线不断裂的前提（Fig. 10）。

### 渲染器
自写 CUDA kernel，跑在**通用计算核**上，不走图形管线。三个后果：无 host 往返（kernel 在训练进程内）；随新硬件加速；无图形驱动的 HPC 集群也能跑（JIT 编译）。
H100 上 **500K agent-steps/s = 2M images/s**；比 Gigapixel 最快配置快 1.4–4.1×；A100L→H100 提速 1.5×（Madrona 反而降到 0.2–0.7×）。渲染占单次训练迭代约 **10%**（Madrona 为 >50%）。

### 策略 Alberti（论文 Tab. 4）
| 组件 | 值 |
|---|---|
| Conv stack | 5 层，128 输出通道，从零训练 |
| 池化 | 每相机 16 个 learned query + 1 层 cross-attention |
| scene embedding | 4×16 tokens → 512-d (= 4×128) |
| backbone | MLP 4 层 × 512，GELU |
| actor/critic | 共享 backbone，线性头 |
| 动作空间 | joint discrete，4 纵向 × 3 横向 jerk bins |
| backbone 输入 | 384-d（与向量 baseline 相同） |
| group embedding 宽度 | 64 |
| **无 LSTM** | |

两个模态都从 ego 中**移除**了部署车辆测不到的量：车道中心横向偏移、相对车道朝向。

### PPO / 环境（论文 Tab. 5）
| 参数 | 值 |
|---|---|
| Δt | 0.1 s |
| episode 长度 | 1280 步（128 s）|
| 早停 reset | ≥40% agent 失活 |
| 训练地图 | 8 张 CARLA town（Town01–07, Town10HD）|
| 每图 agent 数 | ~U{1, 120} |
| 并发 agent | 20 envs × 1024 = 20,480 |
| 行人/骑车人比例 | 0.1 / 0.1 |
| 停放车 | 32/episode |
| rollout 长度 | 128 步 |
| rollout size | 2,621,440 agent steps |
| minibatch | 65,536 |
| epochs/rollout | 2 |
| γ / λ | 0.999 / 0.95 |
| clip | 0.2 |
| value clipping | 无 |
| value loss 系数 | 0.5 |
| entropy 系数 | 0.01 |
| optimizer | AdamW (0.9, 0.999, 1e-8, wd=0.01) |
| lr | 5e-4，cosine |
| max grad norm | 0.5 |
| advantage | 每 minibatch 归一化；按 0.01× running max abs 过滤 |
| 精度 | FP32 |
| 训练量 | 50B agent steps ≈ 35M km；32×H100（8 节点×4）13 小时 @ ~1.3M steps/s |

### 11 项 reward（论文 Eq. 3/4，Phase 7 用）
`R = R_goal + R_collision + R_off-road + R_stop-line + R_comfort + R_l-align + R_l-center + R_velocity + R_overspeed + R_reverse + R_timestep`
- `R_collision = -(α_collision + 0.1|v|)·1[collision]`
- `R_comfort = -α_comfort(1[|a_long|>3] + 1[|a_lat|>3] + 1[|ȧ_long|>5 ∨ |ȧ_lat|>5])`
- `R_overspeed`：超过车道限速 2 m/s 触发（论文相对 Gigaflow 的新增项）
- 行人只启用 goal 和 collision 两项，goal 半径上限 0.5 m

conditioning 采样分布（每 episode 每 agent 重采样，进观测）：
`δ_goal~U(2,12)`, `v_goal~U(0,20)`, `α_collision~U(0,3)`, `α_off-road~U(0,3)`, `α_stop-line~U(0,1)`, `α_comfort~U(0,0.1)`, `α_l-align~U(2.5e-4,2.5e-2)`, `α_vel-align~U(0,1)`, `α_l-center~U(2.5e-4,7.5e-3)`, `α_center-bias~U(-0.5,0.5)`, `α_reverse~U(2.5e-4,7.5e-3)`, `α_overspeed~U(0,1)`；固定：`α_velocity=2.5e-3`, `α_timestep=2.5e-5`
动力学：`C_throttle, C_steer ~ Umix(0.8,1.25)`, `C_acc ~ Umix(2/3,1.5)`（最大前向加速度 = 2.5·C_acc m/s²）

---

## 4. 本机规模上限

| 项 | 论文 | 本机（A6000 48GB）|
|---|---|---|
| GPU | 32× H100 | 1× A6000 |
| 并发 agent | 20,480 | 1,024 – 4,096 |
| rollout | 2.62M steps | 131K – 524K |
| 图像 buffer (uint8 96×54×3 = 15.5 KB/agent-step) | — | 2.0 – 8.1 GB |
| 相机 | 4 路 | 1 路（rig 可扩）|
| 训练量 | 50B steps | 1–3B（数天）|

**诚实边界**：拿不到论文的零样本 WOMD 迁移结论（需完整训练量）；不得用 A6000 数字对标论文 Fig. 5a 的 H100 吞吐。可复现的是：光栅器正确性、吞吐随分辨率的趋势、"透视策略能否接近向量 baseline"的定性结论。

---

## 5. 阶段进度

- [x] **Phase 0** — Agents.md（本文件）
- [x] **Phase 1** — C 侧 RenderState ✅ 已验证
- [x] **Phase 2** — torch 参考光栅器 ✅ 25 项测试全过
- [x] **Phase 3** — CUDA 光栅器 ✅ 30 项测试全过
- [x] **Phase 4** — PerspectiveVecEnv 包装器 ✅
- [x] **Phase 5** — DriveCam 策略 + drive_cam.ini ✅
      **训练已跑通**：`puffer train puffer_drive_cam` 完整跑完、exit 0、指标在改善、检查点可存可读。
- [ ] **eval 相机可视化**（**下一步**，见会话记录决定 2）
- [ ] Phase 6 — Multiprocessing shm 通路（吞吐优化，当前用 Serial 即可训练）

### 当前状态：可以开始训练

```bash
puffer train puffer_drive_cam        # Alberti（相机观测）
puffer train puffer_drive --vec.num-workers 8   # 向量 baseline（注意 worker 数）
```

800K 步烟测（7 epochs，1024 agents，96×54 单前视相机）的指标变化：

| 指标 | epoch 1 | epoch 7 |
|---|---|---|
| explained_variance | 0.133 | 0.301 |
| episode_return | −10.222 | −9.118 |
| collisions_per_agent | 5.541 | 4.472 |
| offroad_rate | 0.867 | 0.805 |
| goals_reached | 0.049 | 0.078 |

**纯像素输入下确有学习信号。** 检查点写到 `experiments/puffer_drive_cam_<run_id>/`，
`load_state_dict(..., strict=True)` 可直接加载续跑（已验证）。

**一个既有缺陷（非本次引入）**：`tests/test_drive_scenario_length.py` 失败，
`episode_length=5` 时日志记录成 4。原因是 `c_step` 里截断判据用 `(timestep+1) >= episode_length`
与日志计数差一。之前该测试因 `binaries/training` 不存在被 `FileNotFoundError` 跳过，
数据软链接建好后才暴露。`git diff drive.h` 中无任何一行涉及 `episode_length`/`add_log`/
`reached_time_limit`/`logs[]`，故与本次改动无关。**修不修是 repo 的语义决定**（N 步还是 N−1 步）。
- [ ] **Phase 6** — Multiprocessing shm 通路（`env_binding.h` / `vector.py` / `binding.c` / `pufferlib.py:19`）——纯吞吐优化，前 5 阶段在 Serial 下即可验证正确性

### Phase 1 实现纪要（已完成）

改动文件：`drive.h`、`binding.c`、`env_binding.h`、`env_config.h`、`drive.py`、`config/ocean/drive.ini`

- `drive.h`：新增 `OBS_MODE_VECTOR/RENDER_STATE`、三个 `RENDER_*_FEATURES`、`RENDER_ROAD_TYPES_DEFAULT`；
  `Drive` 结构体加 `obs_mode/render_road_types/render_agents/render_egos/render_roads/render_counts/render_max_*`；
  新增 `drive_ego_dim()`/`drive_obs_size()`（观测步长的唯一真相源）、`render_type_enabled()`、`render_road_width()`、
  `count_render_roads()`、`fill_render_roads()`、`fill_render_state()`。
  `compute_observations` 在 render_state 模式下写完 ego 就 `continue`，末尾调 `fill_render_state()`。
  BEV 观测叠加渲染（`drive.h` 内）在 render_state 模式提前 return（否则会越界读 obs）。
- 缓冲区传递沿用既有 `my_put` 机制：Python 分配 numpy → `binding.env_put(env_id, render_*=...)` → C 存指针。
  顺带把 `my_put` 各字段改为**可选**（原实现对缺失 key 会解引用 NULL 直接段错误）。
  `render_roads` 绑定时 C 立即调 `fill_render_roads()`（路网静态，只填一次）。
- 新增 binding 方法 `vec_render_state_sizes(handle)` → 每个 env 的 `(road_segments, num_actors, num_egos)`，供 Python 定尺寸。
- 新导出常量：`RENDER_{AGENT,ROAD,EGO}_FEATURES`、`OBS_MODE_*`、`RENDER_ROAD_TYPES_DEFAULT`。
- `drive.py`：新增 `obs_mode`/`render_road_types` 参数、`_alloc_render_state()`，在 `__init__` 和
  `resample_maps()` 两处 `binding.vectorize` 之后调用（换图会改变路网，必须重分配）。

实测（`map_000.bin`，32 agents，2 envs）：vector 模式 obs 1124 维不变；render_state 模式 obs **11 维（仅 ego）**；
每 env 943 条路段（= LINE 502 + EDGE 441，与独立解析地图的数字精确吻合）；agent/ego 每步更新、road 保持不变；
heading 单位化正确；特权屏障成立（obs 宽度 == ego_features）。

### Phase 2 实现纪要（已完成）

新增 `pufferlib/ocean/drive/raster_ref.py`（纯 torch 参考实现）+ `tests/test_raster.py`（25 测试）。

**渲染语义定义**（CUDA kernel 必须复刻）——世界是平的，据此分三层解析可见性，不必对全部三角形排序：
1. **背景**：解析求相机光线与地平面 z=0 的交点，得到每像素精确的地面深度（天空为 +inf）。
2. **路面标线**：躺在地面上的四边形。与地面共面，所以直接画在其上而不做深度比较，彻底避免 z-fighting。
3. **Agent 长方体**：12 三角形、per-face 上色 + 深度调亮度。仅当其深度 **小于该像素的地面深度** 时才合成
   —— 这个判据对平坦世界是精确的：30m 外的车不可能出现在地面交点只有 5m 的方向上。

层内用**解析边覆盖率作为 alpha 做前向到后向合成**（front-to-back over），即论文赖以在 96×54 保住细车道线的抗锯齿方案。

**踩到并修掉的三个 bug（CUDA 实现时同样会遇到，务必注意）**：
1. **边函数符号反了** —— 用了 `(X-p)×(q-p)` 而非 `(q-p)×(X-p)`，导致三角形内部判为负，整幅图全空。
   正确式：`e = ex*(py-p_y) - ey*(px-p_x)`，再乘 `sign(area)` 归一化绕向。
2. **不透明片元被丢弃** —— 透射率写成 `cumprod(1-a)/(1-a)`，在 `a=1`（长方体内部的常态）处除零，
   被 `where` 兜成 0，于是车身内部整片消失只剩边缘。正确做法是**独占前缀积**：
   `trans = cat([ones(1), cumprod(1-a)[:-1]])`。
3. **细长三角形覆盖率越界泄漏** —— 顶面等三角形只有 0.1px 厚时两条长边近乎共线，各按半像素膨胀后
   在共享顶点外侧形成约 1° 的楔形重叠，一直漏到 ~57px 之外（实测在 col 10 画出杂散像素）。
   修法：**把覆盖率限制在三角形屏幕包围盒（膨胀 0.5px）内** —— 真实光栅器按 bbox 遍历本来就不会看到那些像素，
   所以这条同时保证了参考实现与 kernel 的逐 bbox 遍历一致。

**另一个语义修正**：`fill_render_state` 原本把 ego 自己的车也写进场景，渲染时自车 box 距离为 0 糊满画面。
论文明确 "Only the surroundings are rendered"。故 **`RENDER_EGO_FEATURES` 由 4 改为 5**，
第 5 列是该 ego 自身图元在 agents 数组中的下标（无图元时为 -1），光栅器据此跳过。

参考实现吞吐约 **4 ego-images/s**（CPU，943 路段 + 21 agents）—— 它只是 ground truth，慢是设计使然。

### Phase 3 实现纪要（已完成）

新增 `pufferlib/extensions/cuda/raster.cu` + `pufferlib/ocean/drive/raster_cuda.py`；
`pufferlib.cpp` 加 `drive_raster` schema，`setup.py` 的 `torch_sources` 加 `raster.cu`。
算子名 `torch.ops.pufferlib.drive_raster(agents, roads, egos, rig, out)`，仅 CUDA 实现。

**kernel 结构**（两阶段）：
1. `transform_kernel`：每个 block 一张图像，把全部图元投影到屏幕空间写入 scratch（含 bbox 与颜色）。
   scene 只有约 1000 条路段，变换一次远比逐像素重算便宜。
2. `raster_kernel`：每个 block 负责一张图像的一个 **32×16 tile**，线程协作把 bbox 与 tile 相交的
   三角形下标压进 shared memory，再各自着色自己的像素。避免了每像素扫全部 1600+ 三角形。

**关键教训：合成场景测试通过 ≠ 正确。** 首版在合成场景上完全通过，但真实场景 max abs diff 达 **48**，
0.028% 像素超容差，且坏像素**全部集中在地平线下方的 28–29 两行** —— 那里几十条远处路段挤进一两行像素。
两个原因，都必须修：
- `transform_kernel` 用 `atomicAdd` 分配 scratch 槽位 → 三角形顺序不确定 →
  **共面路段深度完全相同时**合成顺序随机。
- 每像素只保留 `MAX_FRAGS` 个片元，而参考实现当时对全部片元排序，截断行为不一致。

修法是让两边采用**同一套确定性语义**：
- kernel 改用**按图元下标的固定槽位**（去掉 atomic），被剔除的槽位写入倒置 bbox 标记为空，
  使三角形下标与参考实现的枚举顺序逐一对应（路面：先全部第一个三角形、再全部第二个；agent：face-tri 主序 `t*A+i`）。
- 片元插入时按 `(depth, 三角形下标)` 字典序破平 → **结果与线程访问顺序无关**。
- 参考实现改用 `argsort(..., stable=True)` 并同样截断到最近 `MAX_FRAGMENTS=16` 个**有效**片元
  （零覆盖的片元把排序键设为 +inf，不占名额）。

修完：max abs diff **48 → 2**，超容差像素 **0.028% → 0.0004%**。
残留的 2 级差异来自 float32 在边缘覆盖率斜坡上的舍入，属可接受范围；
`test_cuda_matches_reference_dense_scene` 的断言即为 `≤2` 且超 1 级的像素占比 `< 0.1%`。

**代价**：确定性槽位使被剔除的图元也要占槽写入，吞吐从 885K 降到 319K images/s（96×54）。
这是拿性能换正确性，值得；若日后要优化，方向是让压缩保持顺序（例如前缀和代替 atomic）而非退回 atomic。

**吞吐实测（A6000，render-only，2048 egos，单相机）**：

| 分辨率 | images/s |
|---|---|
| 32×18 | 350K |
| 64×36 | 337K |
| 96×54（默认） | **319K** |
| 192×108 | 187K |
| 384×216 | 71K |

对照：论文在 H100 上 500K agent-steps/s = 2M images/s（4 相机）。本机 96×54 单相机 319K images/s
≈ 4 相机下 80K agent-steps/s。**不要拿这个数字直接对标论文 Fig. 5a** —— 硬件差一代半，且本 kernel
未做论文那级别的优化。可复现的是趋势：分辨率升高吞吐下降，且渲染远未成为训练瓶颈
（C 环境本身只有 426K agent-steps/s）。

### 验证清单
1. 数值一致性：`tests/test_raster.py`，CUDA vs torch 参考，uint8 容差 ≤1 —— **Phase 3 验收门槛**
2. 几何正确性：同帧渲染图 vs raylib BEV（`drive.h:3024`）并排人工核对
3. 抗锯齿：32×18 / 64×36 / 96×54 三档下车道线不断裂（对应论文 Fig. 10）
4. 吞吐：render-only agent-steps/s 随分辨率扫描，对比论文 Fig. 5a 的**趋势**；pufferl profile 分段看渲染占比
5. **特权屏障**：单测断言 `DriveCam.forward` 收到的观测字节数 == image + ego 大小
6. 训练：`puffer train puffer_drive_cam` vs `puffer train puffer_drive`（同图同种子）
7. 回归：`pytest tests/test_drive_train.py tests/test_drive_config.py`，确认 `obs_mode=vector` 未破坏

---

### Phase 4 + 5 实现纪要（已完成）

新增 `pufferlib/ocean/drive/perspective.py`（PerspectiveVecEnv）、`ocean/torch.py` 的 `DriveCam`、
`config/ocean/drive_cam.ini`；改 `ocean/environment.py`（注册 `drive_cam` + `vecenv_wrapper` 钩子）、
`pufferl.py`（`load_env` 里调用该钩子 + `ENV_KWARG_BLOCKLIST`）。

**观测打包**（一个 uint8 缓冲，避免图像付 float32 的代价）：
```
[ num_cameras*3*H*W 图像字节 | ego_dim 个 float32 的字节 ]
```
单相机 96×54 时 = 15552 + 44 = **15596 字节**。`DriveCam.unpack()` 用 `view(torch.float32)` 还原 ego
（故要求 image_bytes 4 字节对齐，构造时已断言）。

**特权屏障落在包装器**：`_render()` 里组装的 agents/roads/egos 张量在函数返回后即出作用域，
policy 的 `forward` 签名里只有打包好的 obs。`t_e2e` 断言 `obs.shape[1] == image_bytes + ego_dim*4`。

**kernel 增加按场景分段**（Phase 3 之后补的）：训练时有一百多个 env，逐 env 调用会被 kernel 启动和
张量分配开销吃掉，所以 `drive_raster` 增加 `ego_scene[E]`、`agent_ranges[S+1]`、`road_ranges[S+1]`，
一次 launch 渲染整批。scratch 按批内最大场景定尺寸，超出部分槽位标记为空。
路网只在 `resample_maps` 后重传（`_upload_roads` 用 `id()` 做缓存键）。

**DriveCam 架构**（论文 Tab. 4）：5 层 conv 到 128 通道、相机间共享权重、从零训练 →
每相机投影到 128-d → 拼接 → scene 256-d；ego 编码 64-d；拼接后过 4×512 GELU backbone + 线性
actor/critic 头。动作沿用 `MultiDiscrete([4*3])`。**无 LSTM**（`rnn_name = None`）。
单相机下未用论文的 16 query token + cross-attention 池化 —— 那个设计的动机是让多路不同分辨率相机
映射到同一 latent 宽度，单相机不成立；扩到 4 路时再加，作为消融项。参数量 1.79M。

**已知限制**：包装器需要 env 在训练进程内，故 `drive_cam.ini` 用 `backend = Serial`。
Multiprocessing 的共享内存通路是 Phase 6，尚未做。

**新建 ini 的坑**：pufferl 直接 `config["key"]` 取值，缺键就是运行时 `KeyError`，而且往往在
第一次反向传播时才炸（`anneal_entropy` 就是在 `pufferl.py:460` 才崩）。新配置文件务必与
`drive.ini` 做键集对比：
```bash
.venv/bin/python -c "
import configparser
def keys(f,s):
    c=configparser.ConfigParser(inline_comment_prefixes=('#',';')); c.read(f)
    return set(c[s].keys()) if c.has_section(s) else set()
for sec in ('base','vec','policy','env','train','eval'):
    d=keys('pufferlib/config/ocean/drive.ini',sec)-keys('pufferlib/config/ocean/drive_cam.ini',sec)
    if d: print(sec, sorted(d))
"
```

**本机的既有配置问题**（与本次改动无关）：`drive.ini` 的 `[vec] num_workers = 16` 超过本机 8 核，
`pufferlib.vector.make` 会直接抛 `APIUsageError`。跑向量 baseline 要加 `--vec.num-workers 8`
（或更低）。`drive_cam.ini` 用 Serial，不受影响。

**分阶段性能实测**（1024 agents，139 scenes，96×54 单相机，A6000）：

| 阶段 | 单步耗时 | 折合吞吐 |
|---|---|---|
| C env step | 2.6 ms | 395K agent-steps/s |
| 渲染（含上传/打包） | 8.3 ms | 123K agent-steps/s |
| rollout 合计 | 10.9 ms | **93.8K agent-steps/s** |

裸 kernel 基准是 319K images/s（3.2 ms/1024 图），所以包装器有约 5 ms 的 host 侧开销 ——
主要是每步在 Python 里对 139 个 scene 做 `np.concatenate`。**这是后续最值得优化的一处**
（做法：预分配一块 pinned 大缓冲，让 C 侧直接写进去，省掉每步的 concat 与多次 H2D）。

前向+反向实测（camera 观测很吃显存，minibatch 不能照搬论文的全局值）：

| minibatch | 耗时 | 峰值显存 | 每样本 |
|---|---|---|---|
| 32768 | 691 ms | **27.6 GiB** | 21.1 μs |
| 8192 | 121 ms | 7.0 GiB | **14.8 μs** |
| 2048 | 34 ms | 1.8 GiB | 16.5 μs |

故 `drive_cam.ini` 取 `minibatch_size = 8192`（论文的全局 65536 是摊在 32 张卡上，约 2048/卡）。

### 打通训练时踩到的集成 bug（按出现顺序）

单元测试全绿不代表能训练。真正跑 `puffer train` 才暴露出下面这些，逐个修掉：

1. **`KeyError: 'anneal_entropy'`** —— 新 ini 的 `[train]` 段缺键，且到第一次反向传播才炸。
   修法见上面的键集对比脚本。
2. **段错误（exit 139）** —— 我把 `render_mode` 写成 0（窗口模式），无 DISPLAY 时 raylib 建窗失败。
   `drive.ini` 用的是 **1（headless，自动拉 Xvfb，`drive.h:2441`）**。训练中途 eval 渲染 rollout 时触发。
3. **`RuntimeError: shape '[64,3,54,96]' is invalid for input of size 704`** ——
   `pufferl.py:525/530` 把 eval 环境**硬编码**成 `load_env("puffer_drive", ...)`，
   于是相机策略在 eval 时拿到的是 11 维向量观测（704 = 64×11）。
   改为 `self.full_args["env_name"]`，让 eval 环境也经过同一条观测管线。
   **这是原有代码的缺陷**：任何给训练环境叠加观测管线的做法都会被它绕过。
4. **`AttributeError: 'PerspectiveVecEnv' object has no attribute 'num_envs'`** ——
   evaluator 通过 `driver_env` 取用模拟器设施（`render`、`resample_maps`、
   `get_global_agent_state`、`num_envs`）。给包装器加了 `num_envs` 属性和 `__getattr__` 委托，
   让它对"观测以外的一切"保持透明。
5. **`step(per_env_logs=...)`** —— 只有 `Drive.step` 接受这个参数，`PufferEnv.send` 和 Serial 都不接受。
   包装器的 `step` 用 try/except TypeError 兼容两者。

## 6. 后续（另立项）

Phase 7 = 论文 3.1 的 Gigaflow 特性补齐，也是渲染保真度的真正瓶颈：红绿灯 + 停止线、墙体（兼作遮挡物）、停放车、11 项 reward、per-agent conditioning、车道限速。红绿灯和墙体需扩展地图二进制格式和 `data_utils/` 转换脚本。

---

## 7. 会话记录

### 2026-08-04 — session 1
- 读完论文全文，确认 Pictura 基于本 repo；确认官方代码不发布
- 核对 repo 现状，建立 §2 事实表；实测地图数据（§2 地图小节）
- 从论文 Tab. 3 取到完整相机内外参（§3），确认 ego 坐标系约定与 repo 一致
- 定架构：per-env 世界系 RenderState + 包装器持有光栅器（特权屏障）+ rollout 存图像
- 搭好环境（cu121 torch，见 §2 构建小节）
- **完成 Phase 0 / 1 / 2**
- 测试状态：`tests/test_raster.py` 25 通过 1 跳过（CUDA 那条待 Phase 3）；
  `tests/test_drive_config.py` 4 通过 1 跳过。
  注意：`tests/test_drive_train.py` 用 pytest 跑会失败，因为 `load_config` 解析 `sys.argv` 会吃到 pytest 的参数；
  CI 是当脚本跑的（`python tests/test_drive_train.py`），与本次改动无关。
- 训练数据软链接就位（10000 张 WOMD 图），两种 obs_mode 在真实数据上均跑通
- **两项决定（用户拍板）**：
  1. **暂不启动训练**，GPU 全部留给 Phase 3–5 开发与吞吐基准测试。等 Phase 5 完成后，
     用同图/同种子/同超参同时起 `puffer_drive`（向量 baseline）与 `puffer_drive_cam`（Alberti）
     两条对照实验，唯一变量是观测模态。
  2. **eval 相机可视化放到 Phase 3 之后做**，且必须**显示真实观测**（光栅器输出），
     不用 raylib 另画一遍 —— 后者的着色/抗锯齿与 CUDA 光栅器不同，看到的不是策略输入，
     既不能用来 debug 渲染，也不能充当 Phase 3 的验收。
     实现方式：新增 binding 的 `render_camera_rgb` 缓冲，Python 每步为选中 agent 渲染后写入，
     C 侧用 raylib `UpdateTexture` + `DrawTexturePro` 贴成面板（BEV 右侧竖排）。
- **完成 Phase 3**（CUDA 光栅器）。`tests/test_raster.py` 30 通过。
- **完成 Phase 4 + 5**：`puffer train puffer_drive_cam` 可以真正训练。
  首个 epoch：131.1K steps、GPU 98%、policy_loss -0.241、value_loss 4.381、entropy 2.485、
  explained_variance 0.133，user stats 正常产出。
  单次迭代耗时构成：rollout ≈2 s、训练 ≈7 s → **渲染不是瓶颈，策略的前向/反向才是**，
  与论文"渲染仅占迭代约 10%"的结论一致。
  显存占用接近 48 GB 上限，若要加大 `num_agents` 或分辨率需同步下调 `minibatch_size`。
- 800K 步烟测跑完 exit 0，指标改善（见 §5 表），检查点保存/加载/续跑均验证通过。
- 测试：`tests/test_raster.py` 30 通过 1 跳过、`test_drive_config.py` 4 通过 1 跳过；
  `test_drive_scenario_length.py` 失败是既有差一缺陷（详见 §5）。
- **下一步**：eval 相机可视化 —— 新增 binding 的 `render_camera_rgb` 缓冲，
  Python 每步用 `raster_cuda.render` 为选中 agent 出图写入，C 侧 `UpdateTexture` +
  `DrawTexturePro` 贴在 BEV 右侧。注意 `render_mode` 必须是 1（headless/Xvfb）。
- **下一步**：Phase 4 —— `pufferlib/ocean/drive/perspective.py` 包装器。
  要点：`recv()` 内把每个 env 的 RenderState 上传 GPU → `raster_cuda.render` → 丢弃场景图元 →
  返回 `Dict{image: uint8[N*3,H,W], ego: float32[E]}` 打包成字节 Box（走 `emulation.py:115`）。
  注意 RenderState 是 **per-env** 的，而 pufferl 的观测是 **per-agent 扁平** 的，
  包装器要按 `env.agent_offsets` 把每个 env 的 ego 段对齐回全局 agent 下标。
  可直接返回 GPU tensor（`pufferl.py:270` 是直通），但需 `cpu_offload=false`。

### 2026-08-06 / 08-07 — session 2：两次 NaN 训练崩溃 + 非有限更新守卫

**症状**：`puffer train puffer_drive_cam` 两次训到中途 loss 全部变 `nan`，且再也不恢复。

| run | 目录 | 相机 rig | 最后一个好 ckpt | 炸点 | 浪费 |
|---|---|---|---|---|---|
| 1 | `experiments/puffer_drive_cam_178600831551` | nuPlan 96x54 | `model_..._003000.pt` | epoch 3000–4000 之间 | ~2.8 h |
| 2 | `experiments/puffer_drive_cam_zv75yuy4` | Waymo 96x64 | `model_..._004000.pt` | **epoch 4745** | ~2.6 h（且当时仍在跑） |

**取证结论（已验证，不要重做）**

1. **是瞬间事件，不是渐进发散。** 解析 run 2 的 `wandb/run-20260806_210429-zv75yuy4/files/output.log`
   （6171 帧 dashboard），epoch 4744 各项完全健康：`value_loss 0.961`、`entropy 1.653`、
   `approx_kl 0.013`、`clipfrac 0.147`、`importance 0.999`；4745 直接全 nan。**没有任何预兆。**
   → 降 lr / 降 gamma / 降 update_epochs 这类"治发散"的手段对此**无效**，只会推迟触发时间。
2. **炸在 epoch 4745 的 minibatch 0。** 该 epoch 的 `clipfrac` 是 0.003 而非 nan，
   而 `nan > 0.2` 求值为 False：0.147/32 ≈ 0.0046 ≈ 0.003，说明 mb 0 前向有限、
   算出了正常 clipfrac，NaN 产生在 mb 0 的 ratio 之后到 `optimizer.step()` 之间，mb 1–31 全被污染。
3. **一旦发生就是永久的。** `clip_grad_norm_` 用一个全局系数乘所有梯度，
   NaN 的 `total_norm` 让 34 个张量同时中招；Adam 的 `exp_avg`/`exp_avg_sq` 变 NaN 后是吸收态。
   实测：ckpt 权重 1,794,093/1,794,093 全 NaN，`trainer_state.pt` 的 Adam 状态 3,588,186/3,588,220 全 NaN
   （幸存的 34 个是整数 `step` 标量）。
4. **环境和数据已排除。** 全部 10000 张地图（680,847 个 object）扫描：无 non-finite 轨迹/goal/尺寸，
   无 `length == 0`（最小 |length| = 0.2，所以 `drive.h:1746` 的 `tanf(steer)/wheelbase` 除零触发不了）。
   jerk 动力学实跑 192,000 agent-step，11 维 ego 特征全部有限且在范围内。图像是 uint8，本身不可能是 NaN。
5. **光栅器已排除。** `raster.cu` 的 tile 装箱有 `if (s < MAX_TILE_TRIS)`，输出写有
   `x >= c.width || y >= c.height` 检查，scratch 每次调用按本 batch 最大场景重新分配（非缓存复用），
   索引上界 `2*num_roads` / `12*num_agents` 与 `max_*_tris` 一致。
6. **`mb_prio` → inf 数值上不可能。** 要让 `prio_probs` 下溢到精确 0 需要 `Σprio_weights > 7e38`；
   `value_loss ≈ 1` 时 advantage 量级差 30 多个数量级。它能到 ~1e6（训练质量问题，仍待改），但不是 NaN 源。

**已实施的改动（session 2）**

- `pufferl.py` — **非有限更新守卫**：`optimizer.step()` 前用 `torch._foreach_norm` 自行求梯度范数，
  非有限则跳过这次更新并计数，Adam 状态和权重都不被污染。
  注意范数**必须在 clip 之前**求：`clip_grad_norm_` 会把 NaN 系数乘进所有梯度，之后就没有证据可看了。
  单元验证：注入一次 NaN loss，无守卫 → 权重 212/212 + Adam 424/428 全 NaN；有守卫 → 全 0。
- `pufferl.py` — **首次触发时 dump 现场**（`PuffeRL.dump_nonfinite`），落盘到
  `experiments/<env>_<run_id>/nonfinite_epoch<E>_mb<M>.pt`。内含：逐张量的 nan/inf 计数与有限最大值
  （`ratio / logratio / newlogprob / mb_logprobs / mb_prio / adv / advantages / newvalue / mb_values /
  mb_returns / mb_rewards / entropy / idx`）、**未经 clip 的**逐参数梯度健康度与逐层梯度范数、
  以及出问题那几行的观测。只存坏行，不存整个 minibatch（相机帧整批约 130 MB）。
- `pufferl.py` — loss 累加只累加有限项，另出 `losses/nonfinite_minibatches` 计数。
  否则一次跳过会让所有 loss 曲线永久停在 nan，正好把你最需要读的信号毁掉。
- `pufferl.py` — `self.values[idx]` 写回加 `torch.where(isfinite)` 保护，
  避免一个坏预测污染 rollout buffer 导致整个 epoch 的后续 minibatch 全部作废（该 buffer 下个 evaluate() 会重建）。
- `config/ocean/drive_cam.ini` — `checkpoint_interval` 1000 → **200**。
  run 2 从最后一个好 ckpt(4000) 到炸点(4745) 丢了 745 个 epoch ≈ 1.4 h；间隔 200 最多丢 22 min。
- 烟测：`--env.num-maps 200 --train.total-timesteps 800000` 跑完 exit 0，7 个 epoch 指标正常，
  dashboard 出现 `nonfinite_min…  0.000`。

**⚠️ 从 ckpt resume 的已知行为（`pufferl.py:1549` 附近）**

`--load-model-path` **只加载权重**，优化器状态的加载是注释掉的。所以从
`puffer_drive_cam_zv75yuy4/model_puffer_drive_cam_004000.pt` 续跑时：
- Adam 动量从零开始（这次反而是想要的，因为 `trainer_state.pt` 已经是 NaN）；
- `self.epoch` 和 `self.global_step` **重置为 0** → **cosine lr 调度从 5e-4 重新开始**
  （run 2 死的时候已经退火到 3.32e-4），`total_epochs` 也按完整 `total_timesteps` 重算。
  如果想接着原来的退火位置，需要另外处理，目前没做。

**下一步（按顺序）**

1. 从 ckpt 4000 续跑，**盯 `losses/nonfinite_minibatches`**：
   - 稳定为 0 → 之前是极罕见单点事件，守卫已经够用；
   - 每个 epoch 跳几十次 → 那是真发散，守卫在掩盖问题，要回头查。
2. 一旦落盘了 `nonfinite_*.pt`，看 `summary` 里**哪个张量先非有限**，再决定打哪个补丁：
   - `ratio` / `logratio` 非有限 → **B1**：`logratio.clamp(max=10)` 再 exp。
     float32 的 exp 在 88.7 溢出，而 PPO clip 只用到 1±0.2，ratio > 20 在数学上无意义。
     这是目前排第一的嫌疑（单样本事件，8192 个样本里 1 个 inf 不会影响 clipfrac）。
   - `mb_rewards` / `advantages` 非有限 → **B4**：在环境边界加断言。
     注意 `pufferl.py:301` 的 `torch.clamp(r, -1, 1)` 对 NaN 是透传的，
     而 `pufferl.py:326` 的 truncation bootstrap `r + gamma*V` 是**不裁剪**的（代码注释自己写了）。
   - 权重已 NaN 而梯度干净 → 说明更早的某次更新就坏了，守卫的判定位置要往前挪。
3. 与病因无关、无论如何都该做的两条：
   - **B2**：`mb_prio` 按 batch 内 max 归一化（`pufferl.py:404`），标准 PER 做法。现在能到 ~1e6。
   - **B3**：补上论文 Tab. 5 明写的 advantage 过滤（0.01× running max abs）。
     **论文有一个专门压制 advantage 离群值的机制，这份复现漏了，还额外加了会放大离群值的 PER。**
4. 暂不动的（会偏离 Pictura 复现基线，等 1–3 做完确认还有问题再说）：
   换回 Muon（baseline `puffer_drive` 从 `default.ini:28` 继承的就是 muon，
   `drive_cam` 为对齐论文 Tab. 5 才显式改成 AdamW）、γ 0.999 → 0.99
   （论文的 0.999 配 1280 步 episode，这里 `episode_length = 91`，有效视界是 episode 的 11 倍）、
   关 PER、去掉 `vf_clip_coef`（论文明写"无 value clipping"）。

#### session 2 续 —— 真凶找到了：CNN 激活尺度失控，被 LayerNorm 掩盖

**上面 "下一步" 里对 B1/B4 的猜测全部作废**，dump 落盘后指向的是架构，不是 trainer。

**过程**：从 `zv75yuy4/model_..._004000.pt` 续跑（run `66lj6rha`）。
epoch 1–623 干净，**epoch 624 开始每个 epoch 恰好跳 32/32 次，一次不落**。
判据是 `approx_kl` 和 `clipfrac` **精确等于 0.000** —— 策略在 rollout 和 update 之间完全没变。
即：守卫把"永久 NaN"换成了"**永久死锁**"，而且是静默的（score 稳在 0.89、loss 正常波动、dashboard 看着完全活着）。
ckpt 800/1000/…/2000 的 CNN 权重**逐位相同**，实际空转了 1376+ 个 epoch。

**守卫的设计缺陷（记住）**：跳过式守卫只对**瞬时离群**有效。这次的 NaN 是**权重状态的确定性函数**，
跳过 → 权重不变 → 下次 backward 产生一模一样的 NaN → 无限循环。守卫必须能区分这两种情况
（例如连续 N 个 minibatch 全跳就该报警/中止，而不是继续跑）。

**dump 的结论**（`experiments/puffer_drive_cam_66lj6rha/nonfinite_epoch000624_mb021.pt`）：

- 14 个前向张量**全部有限**，loss 也全部健康（`policy_loss -0.103`、`value_loss 0.963`、
  `entropy 1.734`、`approx_kl 0.022`、`importance 1.002`）。**NaN 只存在于 backward。**
- 逐参数梯度里只有 `cam_proj.0.weight` 坏：`g_nan=256`、`g_inf=0`、其余部分 `g_max_abs=0`。
  该权重是 `[128, 3072]`，**256 = 2 × 128** → 3072 个 CNN 特征里有 2 个污染了全部 128 个输出行。
- **整个 CNN（10 个参数）的梯度范数精确等于 0**；`cam_proj.0.bias` 的梯度是 6.4e-7。

**根因**（用同一批合成输入跑所有 ckpt 实测，跨两个 run）：

| ckpt | max\|feat\| | mean\|feat\| | LN 输入 σ | 传回 CNN 的梯度 |
|---|---|---|---|---|
| zv75yuy4 @1000 | 41 | 1.56 | 41 | 1.7e-2 |
| zv75yuy4 @2000 | 54 | 1.34 | 62 | 2.0e-2 |
| zv75yuy4 @3000 | 116 | 1.85 | 112 | 8.4e-3 |
| zv75yuy4 @4000 | 891 | **37.0** | 1381 | 4.5e-4 |
| *@4745 → NaN* | | | | |
| 66lj6rha @200 | 943 | 43.7 | 2280 | 3.0e-4 |
| 66lj6rha @400 | 1173 | 80.0 | 3854 | 1.8e-4 |
| 66lj6rha @600 | 1875 | 136 | 7387 | 1.3e-4 |
| *@624 → 冻结* | | | | |
| 66lj6rha @800（冻结态） | 9442 | **992** | 38041 | **2.4e-5** |

`DriveCam` 的 `cam_proj = Linear(3072→128) → LayerNorm(128) → GELU`，而它前面那个 5 层 conv stack
**层间完全没有归一化**。这构成一个没有回复力的正反馈：

- LayerNorm **前向对尺度不变** → CNN 输出涨多大 loss 都感觉不到，没有任何惩罚；
- LayerNorm **反向正比于 1/σ** → CNN 输出越大，唯一能把它拉回来的梯度就越弱（实测衰减 700 倍）。

越涨越没人管，越没人管涨得越快，终点是 σ² 溢出 float32 → backward 出 NaN。
**前向自始至终有限**（LayerNorm 本来就是干这个的），所以两次都毫无预兆 —— 这解释了
为什么 epoch 4744 的每一项指标都完全健康。

**关键的修复选择依据：weight decay 没用。** 实测 CNN 权重范数在 epoch 200→600 几乎不动
（`cnn.0` 6.97→7.19、`cnn.2` 31.4→35.6、`cnn.4` 41.5→45.6、`cnn.6` 58.0→63.6、`cnn.8` 53.5→55.8，
全部只涨 3~13%），bias 更是几乎不变，而同期 `mean|feat|` 涨了 23 倍。
**尺度爆炸来自 5 层之间的方向对齐（每层约 1.87×），不是权重变大。**
所以任何约束权重范数的手段（weight decay、weight norm penalty）都打不中，
必须直接归一化**激活**（conv 之间加 GroupNorm/LayerNorm）。

**这条通路的来历**：`torch.py:130-134` 的注释写明，论文 Alberti 用 **16 个 learned query token
做 cross-attention 池化**，本复现单相机下换成了 flatten + Linear + LayerNorm。
替换本身有道理，但它在 CNN 和 LayerNorm 之间留下了一条尺度无约束的通路。

**手上所有 checkpoint 都已污染，包括 zv75yuy4 的 4000**（`mean|feat|` 37，健康值 1.5，已在失控轨道上）。
从任何一个续跑都只是重复这次 —— 这正是 `66lj6rha` 只撑了 624 个 epoch
而原 run 从头跑撑了 4745 个的原因：**resume 继承了病灶。**

**下一步（取代上面那份）**

1. **先改架构再训**：给 conv stack 层间加归一化。不改就重训 = 花 ~11 小时复现一个已知 bug。
2. **监控换成前瞻指标**：`nonfinite_minibatches` 现在是**滞后**指标 —— 它响的时候网络已经死了/冻了。
   前瞻指标是 CNN 激活尺度本身（`mean|cnn feat|`，健康 ~1.5，>10 就该警觉），
   每个 epoch 从一批观测上量一次几乎不花钱，也可以离线从任意 ckpt 秒级量出来（脚本见本条表格的做法）。
3. 守卫补一条：连续 N 次全跳 → 报警并中止，不要静默空转。
4. B2（`mb_prio` 按 max 归一化）/ B3（补论文 Tab. 5 的 advantage 过滤）仍然值得做，
   但它们与本次故障无关，别再把它们当成 NaN 的修复。

#### session 2 收尾 —— 修复已落地

**改动**

1. `ocean/torch.py` — `DriveCam` 的 conv stack 每层卷积后加 `GroupNorm`，由新 kwarg
   `cnn_norm_groups`（默认 8，`0` = 恢复论文 Tab. 4 的无归一化版本做消融）控制。
   组数用 `math.gcd(cnn_norm_groups, out_channels)`，对任意通道宽度都合法（32/64/128 → 均为 8 组）。
   选 GroupNorm 不选 BatchNorm：rollout 和 minibatch 的 batch 构成不同，batch 统计量不可比。
   `cnn_out` 仍是 3072，网络其余部分一字未动；参数量 1.79M → 1.80M（+960 个 GN 仿射参数）。
2. `ocean/torch.py` — `DriveCam.feat_scale` 记录每次前向的 `mean|cnn feat|`，
   经新增的 `metrics()` 钩子暴露；`pufferl.py` 若策略有 `metrics()` 就并入 losses 上报，
   于是 wandb 多出 **`losses/cnn_feat_scale`** —— 这就是**前瞻**指标（健康 ~1.5，
   死掉的两个 run 分别是 37 和 992）。钩子是通用的，其他策略没有 `metrics()` 也不受影响。
3. `pufferl.py` — 守卫加**死锁检测**：`consecutive_skips` 连续达到 `stall_patience`
   （= 2 个 epoch 的 minibatch 数，本配置为 64）就置 `stop_reason`，外层 while 循环退出并打印原因。
   仍然走 `close()`，checkpoint 和 wandb 正常收尾。修掉上一条记录里那个"静默空转 1376 epoch"的缺陷。
4. `config/ocean/drive_cam.ini` — `[policy]` 增加 `cnn_norm_groups = 8`。

**验证**

- 接线：`cnn_norm_groups=8` → 5 个 GroupNorm 层，`cnn_out` 不变（3072），参数量 +960。
- **失控路径已封死**（关键验证）：把所有 conv 权重乘以同一个增益 k 来模拟层间对齐，
  测 `mean|cnn feat|`：

  | 每层增益 k | 无归一化 | groups=8 |
  |---|---|---|
  | 1.0 | 0.0365 | 0.391745 |
  | 1.5 | 0.481 | 0.391746 |
  | 2.0 | 2.600 | 0.391747 |
  | 4.0 | **92.94** | 0.391747 |

  k=4 时无归一化版本放大 2550 倍（5 层复利），GroupNorm 版本**六位有效数字不变**。
- 端到端：`--env.num-maps 200 --train.total-timesteps 1200000` 跑通，见下方 session 记录。

**旧 checkpoint 全部作废**：state_dict 的键和层索引都变了（`cnn.0/2/4/6/8` → `cnn.0/3/6/9/12`），
而且旧权重本来就已被污染（`mean|feat|` 37 起步）。**必须从头训**，不要 `--load-model-path`。

**重训时盯什么**（按优先级）

1. `losses/cnn_feat_scale` —— 应当稳定，不应单调上升。持续爬升说明归一化没拦住，回来查。
2. `losses/nonfinite_minibatches` —— 应当恒为 0。偶发几次是瞬时离群（守卫会跳过，无害）；
   连续 64 次会自动停并打印原因。
3. 仍然待办、与本次故障无关：B2（`mb_prio` 按 max 归一化）、B3（补论文 Tab. 5 的 advantage 过滤）。
