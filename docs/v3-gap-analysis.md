# V3 重构差距分析

> 对照 `fix_old_img_V3_重构优化建议报告.md` 逐条核验当前代码库的完成情况。
> 生成时间：2026-09-25。基线：`408 passed, 1 skipped`，ruff 全绿。
>
> **2026-09-26 更新**：第一、二轮补齐已完成，见文末「补齐进展」。当前基线
> **545 passed, 1 skipped**，ruff 全绿。

## 总览

按报告「附录 B：V3 发布验收标准」的 37 项逐条核对：

| 类别 | 已完成 | 部分完成 | 未实现 | 小计 |
|---|---:|---:|---:|---:|
| 架构 | 6 | 0 | 0 | 6 |
| 可靠性 | 6 | 0 | 0 | 6 |
| 性能 | 3 | 1 | 2 | 6 |
| API | 5 | 0 | 0 | 5 |
| UI | 5 | 1 | 0 | 6 |
| 工程化 | 7 | 1 | 0 | 8 |
| **合计** | **32** | **3** | **2** | **37** |

按报告的优先级总表（§8）：

| 优先级 | 项目 | 状态 |
|---|---|---|
| P0（5 项） | Domain Model / 统一 Error Code / 测试基线 / Task lease-retry-attempt / Legacy CLI Adapter | 全部完成 |
| P1（6 项） | ModelBackend / GPU Scheduler / Model Manifest / ArtifactStore / SSE / Before-After UI | 全部完成 |
| P2（6 项） | Redis Queue / PostgreSQL / MinIO / Multi-GPU / Model Hot Reload | 热更新与多 GPU 完成；Redis/PostgreSQL/MinIO 有接口待生产验证 |

---

## 一、完全未实现

### 1. §3.5.4 PyTorch 推理优化 — 零实现

报告要求：

```python
with torch.inference_mode():
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(input)
```

并要求把「是否支持 compile」做成 capability 查询，而不是全局打开。

**当前状态**：全仓库 `inference_mode` / `autocast` / `channels_last` / `torch.compile` /
`float16` / `bfloat16` **零命中**。

**差距**：
- 没有 `torch.inference_mode()` 包裹推理（DDColor 走 `pipeline.process`，内部是否用了不可控）
- 没有混合精度开关，也没有 `supports_fp16` 这类 capability 标志
- 没有 `channels_last` / pinned memory / CUDA stream 优化
- 报告特别强调的「按 capability 决定是否 compile」无从实现（因为没有 capability 位）

**影响**：这是验收标准「性能」栏里 `inference_mode` 与 `mixed precision capability flag`
两项未通过的直接原因。属于**纯性能项，不影响正确性**，但报告把它列在 §3.5.4 的核心位置。

**建议落点**：`ModelBackend` 增加 `capabilities` 里的 `fp16` / `compile` 位，
`DDColorBackend._do_infer` 用 `torch.inference_mode()` + 条件 autocast 包裹；
manifest 里声明每个模型支持哪些精度。

---

### 2. §3.6 / §3.7 `warp_back` 阶段缺失 + 语义去重未解决

报告要求把 pipeline 标准化为：

```
NormalizeStage → AnalyzeStage → DecisionStage → RestorationStage
→ ScratchStage(opt) → FaceStage(opt) → ColorizationStage(opt)
→ PostProcessStage → EvaluationStage → ArtifactPublishStage
```

并且 §3.7 明确警告：

> 避免「全局恢复已经做过一次，人脸增强又再做一次」的语义重复。
> 这不是简单的性能优化，更是可复现性和结果一致性问题。

**当前状态**：`GlobalRestoreStage` 通过 `LegacyCliBackend` 调用 `run.py`，
而 `run.py`（`cli/batch.py`）**在一次调用里跑完 4 个 path**：

| path | 实际做的事 |
|---|---|
| 1/4 | 整体质量修复（含可选划痕） |
| 2/4 | 人脸检测 |
| 3/4 | 人脸增强 |
| 4/4 | warp-back 合成 |

也就是说 `global_restore` 这一个 stage 内部已经包含了人脸检测 + 人脸增强 + warp-back。
若计划里同时有 `face_enhancement` stage，人脸链就会被执行**两次**——正是报告警告的场景。

**差距**：
- 没有独立的 `warp_back` stage（`warp_back` 只作为字符串出现在 `cli/batch.py`）
- `global_restore` 的职责边界与报告定义的「只负责全局退化恢复」不符
- Planner 的 `restore` 计划只有 `global_restore` 一个 stage，所以目前**没有实际触发**双重处理；
  但 `face_enhance` / `auto_colorize` 选项一旦组合，语义就会重叠
- `StageResult` 缺报告要求的 `metrics` 字段（当前只有 `image / metadata / duration / artifacts / message`）
- `NormalizeStage` / `AnalyzeStage` / `DecisionStage` / `PostProcessStage` /
  `EvaluationStage` / `ArtifactPublishStage` 都没有实现（相应逻辑散在
  `runtime._normalize_input` / `runtime._evaluate` / `planner.plan_auto_restore` 里）

**影响**：这是**最值得优先补的一项**——它同时关系到验收标准的「架构」栏
（Stage 职责边界）与结果可复现性。当前实现虽然能跑，但「一个 stage 干了四个 stage 的事」
与报告的 Stage Contract 是矛盾的。

**建议落点**：让 `LegacyCliBackend` 支持 `--stage` 参数只跑指定 path
（`Global/test.py` 与 `Face_Detection`、`Face_Enhancement` 本就分属不同目录，可分别调用），
拆出 `warp_back` stage，并在 Planner 里显式声明各 stage 的职责。

---

### 3. §3.10.2 Golden Image 测试 — 零实现

报告要求的断言维度：

```
shape / mode / range / mean / std / PSNR / SSIM / LPIPS / face count / artifact schema
```

**当前状态**：没有任何测试建立「固定输入 → 输出特征基线」。最接近的是
`tests/inference/test_tiler.py:72` 的 `diff.mean() < 0.5`（拼缝往返校验），
以及 `tests/gpu/test_ddcolor_gpu.py` 的形状断言——都不是 golden 基线。

**影响**：模型或预处理改动导致的**输出质量回归**无法被 CI 捕获。
验收标准「工程化」栏的 `benchmark regression` 也因此只有耗时回归、没有质量回归。

**建议落点**：`tests/golden/` 目录 + 小尺寸固定输入（如 128×128 合成图），
断言 `mode == "RGB"`、`size` 不变、像素 range、mean/std 容差；
真实模型部分用 `-m gpu` 标记，CI 走合成 stage 的 golden。

---

### 4. §3.10.1 Inference 测试：Global / Scratch / Face 全部缺失

报告测试矩阵要求：

```
Inference Test
  ├── Global Restore
  ├── Scratch
  ├── Face
  └── DDColor
```

**当前状态**：`tests/gpu/` 下**只有** `test_ddcolor_gpu.py`。
Global / Scratch / Face 三条链既没有 GPU 测试，也没有用打桩 backend 的 CPU 契约测试。

**影响**：`LegacyCliBackend` 的参数拼装（`--with_scratch` / `--HR` / `--GPU` /
`load_size` / `batchSize`）完全没有测试覆盖，改动时只能靠人工验证。

**建议落点**：用 `monkeypatch` 替换 `stream_subprocess`，断言各阶段生成的命令行参数、
输出路径解析、`pipeline_report.json` 的降级信息解析。

---

### 5. §6 `Task.error_code` 字段缺失

报告的数据模型：

```
Task
├── ...
└── error_code        ← 机器可读错误码
```

**当前状态**：`tasks` 表只有 `error_message`（自由文本）。`grep error_code src/` 零命中。

**影响**：客户端无法按错误类型分支重试策略；报告的「统一 Error Code」只做到了
HTTP 响应层（`ErrorCode` 枚举），没有落库。同一类失败（OOM / 权重缺失 / 输入非法）
在历史里无法聚合统计。

**建议落点**：`tasks` 增列 `error_code`，`fail_task()` 接收 code，
`runtime` 从 `AppError.code` 透传，API 的 `TaskStatusResponse` 暴露该字段。

---

### 6. §3.14.1 ModelPlugin 协议 + 动态发现缺失

报告要求：

```python
class ModelPlugin(Protocol):
    id: str
    version: str
    capabilities: set[str]
    def create_backend(self, config): ...

registry.register(GlobalRestorePlugin())
```

**当前状态**：`ModelBackendRegistry` 与 `StageRegistry` 都已存在，注册方式也是显式的
（`build_default_backend_registry()`），但：
- 没有 `ModelPlugin` 协议
- 没有 entry-point / 目录扫描的**第三方插件发现**机制
- 插件只能通过改源码注册

**影响**：报告 §3.14「可扩展多模型支持」的**内部**扩展点已经打通
（实现 backend + 注册 + 声明 capability，无需改 planner），但**外部**扩展
（不改主仓库就能加模型）还不支持。属于平台化能力，非当前阻塞项。

---

### 7. §3.2 `stages/normalize.py` / `analysis.py` / `evaluate.py` 缺失

报告的目标目录结构里 `inference/stages/` 含：
`normalize.py`、`analysis.py`、`restoration.py`、`face_enhance.py`、`colorize.py`、`evaluate.py`。

**当前状态**：只有 `global_restore.py` / `scratch.py` / `face_enhancement.py` /
`colorization.py`。归一化、图像分析、评估都不是 stage。

**说明**：这是 §3.6 拆分的直接后果，与「未实现 #2」是同一件事的不同侧面。
命名差异（`restoration.py` vs `global_restore.py`）本身不重要，**职责边界才重要**。

---

### 8. §4.3 Step 2 SQLAlchemy + Alembic 未引入

**当前状态**：仍是手写 SQLite + 自研轻量迁移 runner
（`infrastructure/db/migrations/runner.py` + `migrations/versions/0001_v3_baseline.py`）。

**影响**：`FIXIMG_DATABASE_URL` 目前只影响配置展示，**切换到 PostgreSQL 不会真正生效**
（`engine.py` 硬编码 `sqlite3.connect`）。`configs/production.yaml` 里
`database_url` 的注释也如实标注了「PostgreSQL is P2」。

**这是报告 P2 的明确内容，需要外部服务才能验证，属于已知未完成。**

---

### 9. §3.11 CI/CD：安全扫描与发布环节缺失

报告链路：

```
ruff / black / mypy → pytest → API integration → build docker image
→ security scan → release
```

**当前状态**：CI 有 `lint` / `typecheck` / `tests` / `benchmark` / `docker` 五个 job。
- 缺 **security scan**（trivy / pip-audit 等）
- 缺 **release** 环节（镜像推送、tag 触发）
- `docker` job 只构建 `docker/api.Dockerfile`，**未构建 worker 镜像**
- `typecheck` 是 `continue-on-error: true`，实际不构成门禁

---

### 10. 多 GPU：显存感知调度缺失

已实现能力级路由（`FIXIMG_GPU_ROUTING`）与 worker 绑定。但没有：
- OOM 检测与任务迁移
- 按显存余量选择设备
- 设备健康检查（某卡掉线后摘除）

报告 §7.1 的 KPI「GPU 利用率 — 按模型 profiling 优化」也未做 profiling。

---

## 二、部分实现

### 1. §3.5.5 Image Size Policy 只有两档

报告要求四档：

| 尺寸 | 策略 |
|---|---|
| ≤ 1024 | direct |
| 1024~2048 | adaptive resize |
| 2048~4096 | tile / staged |
| > 4096 | reject or pre-resize |

**当前状态**：`tiler.tiling_enabled()` 只判断 `long_side > tile_size`，
`runtime._normalize_input` 只做 `max_image_side` 拒绝。**缺「adaptive resize」中间档**，
也没有把 resize 策略从模型里收归统一策略。

### 2. §3.4.2 日志字段不完整

报告要求：`request_id / user_id / task_id / worker_id / gpu_id /
model_name / model_version / stage / attempt / latency_ms`。

**当前状态**（`infrastructure/observability/logging.py`）：

| 字段 | 状态 |
|---|---|
| request_id / user_id / task_id / worker_id / gpu_id | 已有（contextvars 自动附加） |
| stage | 有，但作为 `extra_fields` 的 `stage=`，非固定字段 |
| attempt | 仅 worker 的 "task claimed" 一行有 |
| model_name / model_version | **缺**（只有 artifacts 表有 model_version 列） |
| latency_ms | **缺**（用的是 `duration_ms`，语义相同但命名不符） |

### 3. §6 `TaskStage.started_at` 缺失

表结构用 `created_at` 表示开始时间，报告要求 `started_at`。语义等价，命名不符。
（`stage_version` / `order` / `duration_ms` / `finished_at` 都已具备。）

### 4. §6 `ModelVersion` 持久化 + 版本化权重目录缺失

报告要求 `models/<name>/v1/` 的磁盘布局，以及 `ModelVersion.metadata_json` 持久化。

**当前状态**：`models/` 下**只有 `manifest.yaml`**；权重散落在
`weights/ddcolor/`、`Global/checkpoints/`、`Face_Detection/`、`Face_Enhancement/`。
`ModelVersion` 是内存对象（从 YAML 解析），没有入库。

**影响**：热更新的版本切换目前依赖 manifest 里的 `version` 字符串，
磁盘上并没有真正的「v1 / v2 两套权重目录」，所以「双版本驻留」在生产上
需要运维自行准备两套路径。机制通了，**资产布局没通**。

### 5. §3.12 依赖锁定未用 uv / pip-tools

报告推荐 `uv` 或 pip-tools 做分组锁定。当前 `requirements.lock` 是 `pip freeze`
产物（93 行），与 `pyproject.toml` 的 extras 分组**没有对应关系**，无法按组锁定。

### 6. §3.13 Redis / PostgreSQL / MinIO 未在生产路径验证

三个后端都有完整接口与实现：
- `QueueBackend` + `SqliteQueueBackend` + `RedisQueueBackend`
- `ArtifactStore` + `LocalArtifactStore` + `S3ArtifactStore`

S3 有 fake-client 契约测试；**Redis 后端没有任何测试**（连 fake 都没有）。
`docker/compose.yaml` 的 `platform` profile 可以拉起 redis/postgres/minio，
但没有任何测试或脚本验证过连通性。

### 7. §7.1 GPU peak memory 无 per task/stage 维度

报告的 KPI：「GPU peak memory — 已有基础统计 → 加入每 task/stage 维度」。
当前 `model_manager.gpu_stats()` 只返回进程级
`torch.cuda.max_memory_allocated()`，没有按 task / stage 拆分。

### 8. §3.9.2 Before/After 只有两态

报告建议优先做 slider（已完成），其次提供
`Original / Restored / Mask / Face crop` 四态对比。后三者未做。
划痕检测 tab 有独立的 mask 输出图，但没有和原图做成对比视图。

### 9. §3.9.3 历史面板缺分页与缩略图

报告的 History 结构含 `thumbnail`。当前用 `gr.Dataframe` 文本列
（Task ID / Type / Created / Duration / Status / Progress / Stages / Metrics），
无缩略图列，无分页控件（只有「显示行数」上限）。

### 10. §3.2 `ui/components/` 空目录、`ui/state.py` 缺失

`src/fiximg/ui/components/__init__.py` 是 0 字节空文件。
报告目标结构里的 `ui/state.py` 不存在（等价职责分散在 `task_progress.py`
与 `history_panel.py`）。属于目录整理，无功能影响。

### 11. §5.1 运行时类命名差异

报告建议 `InferenceRuntime.execute(task, context)`；实现是
`PipelineOrchestrator.execute_queued(task_id, task_type)` / `run(...)`。
职责一致，签名与命名不同。属文档与实现的措辞差异，**不建议为此改动代码**。

---

## 三、已完成（对照清单）

为免误判，以下为报告明确要求且**已确认实现**的部分：

**架构（6/6）**
- `api/routes/*` 只依赖 application 层，不 import 任何 `stages` / `backends`
- `ui/*` 只经 `task_service`，不碰 pipeline internals
- `BaseStage` 契约明确禁止 DB / HTTP / Gradio / Redis（`stages/base.py` 文档字符串）
- `ModelBackend` 统一接口（`backends/base.py`，含 load/warmup/infer/unload/health）
- `QueueBackend` 可替换（sqlite / redis）
- `ArtifactStore` 可替换（local / s3）

**可靠性（6/6）**
- 状态机 `queued → running → completed/failed/cancelled` + `is_terminal` / `is_active`
- lease：`lease_until` + 心跳 + `reset_stale_running` 回收
- retry：backoff + `max_attempts` + `requeue_for_manual_retry`
- idempotency：`Idempotency-Key` → 偏唯一索引 → 重复提交返回既有 task
- worker crash 恢复：lease 过期重排（含无 lease 旧行的 `started_at` 回退）
- artifact 回收：`purge_stale_runs` + `ArtifactExpiredError`（HTTP 410）

**性能（3/6）**
- model warmup：`lazy` / `first-use` / `startup` 三种策略 + `ModelManager._warm`
- p50/p95 基线：`benchmark.latency`（冷/热分离）+ CI 门禁
- queue wait 基线：`MetricName.QUEUE_WAIT_SECONDS` + `benchmark.throughput`

**API（5/5）**
- Pydantic request/response（`api/schemas/`，multipart 通过 `openapi_extra` 补文档）
- 统一错误格式（`api/errors.py`，`{"error": {code, message, request_id}}`）
- SSE task events（`/tasks/{id}/events`，含异步评估的有界等待窗口）
- user identity 映射（`Principal`，`FIXIMG_API_TOKEN_USER`）
- OpenAPI 自动文档（21 条 `/api/v1` 路径）

**UI（5/6）**
- Before/After（5 个 tab 各一个 `gr.ImageSlider`）
- realtime progress（`task_progress.py` 逐帧 yield）
- task history（`ui/history_panel.py`）
- retry（`POST /tasks/{id}/retry` + 历史面板 Re-run）
- download（`GET /tasks/{id}/result`）

**工程化（7/8）**
- pytest（408 passed / 1 skipped）
- integration test（`tests/api/test_api_tasks.py`）
- E2E test（`tests/e2e/test_upload_to_result.py`，上传→队列→worker→结果）
- lint（ruff，CI 门禁）
- type check（mypy，CI 中 `continue-on-error`）
- CI（5 个 job）
- Docker build（api 镜像；另有一体化与 worker 镜像文件）
- benchmark regression（CI 跑 latency + throughput，但**不比对基线**）

**P2 中已完成**
- Model Hot Reload（§3.15）：双版本驻留 + 原子切换 + 回滚 + 租约排空
- Multi-GPU（§4.3 Step 4）：能力→设备路由 + worker 绑定 + 能力感知领取 + 按设备分槽

---

## 四、建议的补齐顺序

| 顺序 | 项目 | 理由 |
|---|---|---|
| 1 | §3.6/§3.7 拆分 `warp_back` + 消除语义重复 | 唯一影响**结果正确性/可复现性**的未完成项；同时让 `StageResult.metrics` 等契约补齐 |
| 2 | §3.10.1 Legacy 链的推理测试（打桩 backend） | 补齐覆盖空白，成本低，是 #1 的安全网 |
| 3 | §3.10.2 Golden Image 测试 | 质量回归的唯一防线；与 #2 一起做 |
| 4 | §3.5.4 推理优化（inference_mode + 精度 capability 位） | 验收标准里明确列出的两项；改动集中在一个 backend |
| 5 | §6 `Task.error_code` 落库 | 小改动，补齐可观测性与客户端分支能力 |
| 6 | §3.11 CI：worker 镜像构建 + 安全扫描 | 门禁完整性 |
| 7 | §3.13 Redis 后端契约测试（fake client） | 无需真实服务即可验证，与 S3 测试对齐 |
| 8 | §3.5.5 Image Size Policy 四档 | 消除「每个模型自己管 resize」 |
| 9 | §6 版本化权重目录 `models/<name>/vN/` | 让热更新在生产可用 |
| 10 | §4.3 Step 2 SQLAlchemy + Alembic | 需要真实 PostgreSQL 验证，工程量最大 |

---

## 五、结论

报告 P0 与 P1 的**全部 11 项已完成**，P2 的 6 项中 **2 项完成**（模型热更新、多 GPU）、
3 项有完整接口待生产验证（Redis / PostgreSQL / MinIO）、1 项（多 GPU 显存感知调度）未做。

验收标准 37 项中 **32 项通过、3 项部分通过、2 项未通过**。两项未通过的
（`inference_mode`、`mixed precision capability flag`）都属于 §3.5.4 同一节，
改动范围小；真正需要投入的是 §3.6/§3.7 的 Stage 职责拆分——它是报告
「V3 核心判断」里「消除 legacy subprocess 语义重复」的直接落点。

---

# 补齐进展（2026-09-26）

本轮针对上述未完成项做了两批补齐。基线从 `408 passed` 提升到 **`545 passed, 1 skipped`**，
ruff 全绿，连续两次全量无 flake。

## 一、已完成的未实现项

| 原编号 | 报告章节 | 处理方式 |
|---|---|---|
| 1 | §3.5.4 PyTorch 推理优化 | 新增 `inference/precision.py`：`PrecisionPolicy` 由 manifest 的 `precision` / `channels_last` / `compile` 声明驱动，`torch.inference_mode()` 始终启用，autocast 仅在 CUDA 生效；DDColor 已接入，`FIXIMG_PRECISION` 可全局覆盖 |
| 2 | §3.6/§3.7 warp_back + 语义去重 | CLI 拆成 `--stages 1,2,3,4` 四个可独立执行的阶段；新增 `WarpBackStage`；`GlobalRestoreStage` 只跑 path 1；`StageResult` 增加 `metrics` 字段；`task_stages.metrics_json` 落库；`planner.FACE_CHAIN` 显式声明四阶段职责 |
| 3 | §3.10.2 Golden Image 测试 | 新增 `tests/inference/test_golden_image.py` + `tests/inference/golden/synthetic_restore.json` 基线，断言 mode/size/shape/channels/range/mean/std/PSNR/SSIM/artifact schema |
| 4 | §3.10.1 Legacy 链推理测试 | 新增 `tests/inference/test_legacy_chain.py`（32 例）：命令行拼装、共享根目录、降级报告解析、进度标记流、四阶段端到端、Stage 不写数据库 |
| 5 | §6 `Task.error_code` | `tasks.error_code` 列 + `fail_task(error_code=)` + `error_code_of()` 从 `AppError` 透传；API 暴露；重试时清空 |
| 6 | §3.11 CI | 新增 worker 镜像构建、四个 compose 文件的 `config --quiet` 校验、Trivy 镜像扫描、`pip-audit` 依赖审计 |
| 7 | §3.13 Redis 后端测试 | 新增 `tests/unit/test_redis_queue.py`（23 例，fake client）。**测试抓到 3 个真实缺陷**（见下） |
| 8 | §3.5.5 Image Size Policy | 新增 `inference/size_policy.py` 四档决策（direct / adaptive_resize / tile / reject），接入运行时；adaptive 档推理后恢复原始尺寸，保证用户下载的图不变形 |
| 9 | §6 版本化权重目录 | `ModelManifest` 支持 `models/<name>/vN/` 优先解析，声明版本 `1.0.0` 可映射到 `v1/`，无目录时回退到声明路径；新增 `available_versions()` 供热更新前检查 |
| 10 | §3.4.2 日志字段 | 阶段日志补齐 `model_name` / `model_version` / `attempt` / `latency_ms`（与 `duration_ms` 同值双写） |

## 二、本轮测试抓到并修复的真实缺陷

| # | 缺陷 | 影响 |
|---|---|---|
| 1 | `XAUTOCLAIM` 返回值按二元组解包 | 真实 Redis 返回三元组 `(cursor, records, deleted)`，流为空时 `claim` 直接 `ValueError` —— Redis 后端从未被真正执行过 |
| 2 | `RedisQueueBackend.ack(task_id)` 是空实现 | 完成的任务其 stream 条目一直 pending，租约到期后被重新领取、重新校验、重新 ack，噪声且慢 |
| 3 | Redis 后端无任何降级 | `enqueue`/`claim` 在 Redis 抖动时直接把异常抛进 worker 循环；现改为记日志并降级（DB 仍是权威） |
| 4 | `requeue_for_manual_retry` 未清 `error_code` | 人工重试后任务仍带着上一次的失败码 |
| 5 | 同步 `run()` 也被延迟评估 | 返回的 `evaluation_text` 为 None（上一轮） |
| 6 | 配置档案名被当作环境名 | `FIXIMG_PROFILE=docker` 导致 `app_env="docker"` 校验失败（上一轮） |

## 三、第三批补齐（2026-09-26 下午）

基线 **`637 passed, 3 skipped`**，ruff 全绿，连续两次全量无 flake。

| 原项目 | 处理方式 |
|---|---|
| §3.9.2 对比仅两态 | 历史面板新增 Mask 与 Face crop 预览 + `original ◀ ▶ mask` 第二个滑块（有 mask 才显示）；检测 tab 的滑块改为 mask 语义 |
| §3.9.3 无缩略图与分页 | 新增 `gr.Gallery` 缩略图（按 mtime 缓存在 `storage/thumbs`）+ 每页条数下拉 + Prev/Next + `Page N of M`；行号按页相对解析 |
| §7.1 GPU peak memory per stage | `inference/gpu_memory.MemorySampler` 测量**每阶段增量**峰值，写入 `stage_metrics["gpu_peak_mb"]`；`/health/ready` 暴露每卡 free/used + 进程高水位 |
| 多 GPU 显存感知调度 | `select_device()` 按显存余量选卡（`FIXIMG_GPU_MEMORY_AWARE` / `_HEADROOM_MB`）；OOM 时自动迁移到同能力的另一张卡重试一次，无备选则报原始错误 |
| §3.14.1 外部插件发现 | 新增 `inference/plugins.py`：`ModelPlugin` / `StagePlugin` 协议 + `fiximg.models` / `fiximg.stages` entry-point 组；两个默认注册表启动时自动发现；`FIXIMG_PLUGINS=0` 可关闭 |
| §3.12 依赖分组锁定 | 新增 `scripts/export_locks.py`，按 pyproject extras 生成 `requirements/{runtime,gpu,test,benchmark,dev,redis,s3,postgres,otel}.txt`；`make lock` / `lock-check` / `lock-freeze` |
| §4.3 Step 2 数据库方言 | `engine.resolve_database_path()` 解析 `FIXIMG_DATABASE_URL`；**非 SQLite 显式抛 `UnsupportedDatabaseError`**，不再静默写错库；`validate_database_url()` 在 `init_db` 中无副作用校验 |
| §3.13 生产路径验证 | 新增 `scripts/smoke_services.py`（`make smoke-services`）：真实探测 SQLite/Redis/对象存储/队列，未配置的服务报 SKIP 而非 FAIL |

### 第三批测试抓到的真实缺陷

| # | 缺陷 | 影响 |
|---|---|---|
| 1 | SSE 在 `follow=false` 时跳过异步宽限期 | 只轮询一次的客户端永远拿不到 `task.evaluated`，异步指标形同不存在 |
| 2 | 阶段指标非空使异步评估永不调度 | `metrics == {}` 被当作「评估未运行」的判据，阶段上报指标后判定恒为假 —— 改用 `evaluation_text is None` |
| 3 | `apply_database_url` 无条件覆盖 `DB_PATH` | 测试/嵌入方直接设置 `DB_PATH` 会被 YAML 默认值覆盖，48 个用例挂掉；拆成「无副作用校验」与「仅 env 显式配置时重定位」 |

## 四、第四批补齐（2026-09-26 晚）

基线 **`669 passed, 3 skipped`**，ruff 全绿，连续两次全量无 flake。

| 原项目 | 处理方式 |
|---|---|
| §4.3 Step 2 SQLAlchemy + Alembic | **Alembic 已引入**：`alembic.ini` + `migrations/env.py` + 两个版本化修订（`0001` 基线建全表、`0002` 补后加列）。`runner.py` 重写为 Alembic 封装（`upgrade`/`downgrade`/`stamp`/`status`/`history`/`--sql`），旧 `MigrationRunner` 保留为兼容别名。URL 从 `FIXIMG_DATABASE_URL` 解析，与运行时同源 |
| §3.13 Redis 真实协议验证 | **手写 fake 换成 fakeredis**（实现真实 `XADD`/`XREADGROUP`/`XAUTOCLAIM`/`XACK` 与 consumer group 语义）。另设 `FIXIMG_TEST_REDIS_URL` 可对真实服务器跑同一套用例 |
| §3.12 锁定文件不完整 | 联网重建 `requirements.lock`（93 → 103 行），补入 pytest/fakeredis/SQLAlchemy/alembic/redis 等；`fakeredis`/`alembic`/`SQLAlchemy` 加入 `[test]` extra；CI 增加 `export_locks.py --check` 门禁 |
| §3.2 `ui/components/` 空目录、无 `ui/state.py` | `ui/components/` 承载 `result_column` / `comparison_slider` / `preview_row`；新增 `ui/state.py`（`UserState` / `resolve_user_state` / `visible_user_id` / `tab_visibility`），把「调用者是谁」收敛到一处 —— 此前 `apply_role` 与 `history_panel` 各写一遍 |
| §3.11 CI 门禁 | 新增「迁移 upgrade → downgrade base → 再 upgrade」的往返校验，以及锁定分组新鲜度校验 |

### 第四批测试抓到的真实缺陷

| # | 缺陷 | 影响 |
|---|---|---|
| 1 | `applied_revision()` 未应用 `FIXIMG_DATABASE_URL` | CLI 里 `upgrade` 写了正确的库，紧接着的 `current` 却读默认库并报告 `(none)` —— 迁移工具与运行时不同源 |
| 2 | `status().pending` 把已应用的修订也算作待处理 | `walk_revisions(base="base", head=applied)` 返回的是「到 applied 为止」的链，方向反了；应从 applied 走到 head |
| 3 | `0002` 在离线模式（`--sql`）下调用 inspector | `MockConnection` 没有 inspection 系统，`alembic upgrade --sql` 直接崩 |
| 4 | `alembic.ini` 含非 ASCII 字符 | ConfigParser 按平台默认编码（中文 Windows 为 GBK）读取，UTF-8 的 `§`/`—` 触发 `UnicodeDecodeError`，全部迁移失败 |
| 5 | `queue/redis.py` 的 `redis` 是函数内惰性导入 | 无法被测试替换，也无法断言「未安装 redis 包」的分支 |

## 五、第五批补齐（2026-09-26 深夜）

基线 **`704 passed, 3 skipped`**，ruff 全绿。

| 原项目 | 处理方式 |
|---|---|
| PostgreSQL 数据路径 | 新增 `infrastructure/db/dialect.py`：把四类 SQLite 专有构造抽象为方言（`?` 占位符、`json_group_array/object`、`PRAGMA table_info`、`BEGIN IMMEDIATE`）。仓储改为通过方言取片段，**SQLite 输出逐字不变**（全量套件即证据），PostgreSQL 输出用 SQLAlchemy 的真实 `postgresql` 方言**编译验证**（无需服务器） |
| §3.2 `normalize` / `analysis` / `evaluate` 不是 stage | 在 `stages/base.py` 写清了**为何刻意不做成 stage**：size policy 必须先于 plan（它会拒绝输入、会改变 stage 收到的图），analysis 是 `plan_auto_restore` 的输入 —— 做成 stage 就要算两次 plan 或让 stage 改 plan，正是 stage 契约要防的耦合；evaluation 则与 §3.16「不阻塞结果」冲突。二者仍是独立可测模块并记入报告 |
| `requirements.lock` 缺口 | 安装 boto3 / opentelemetry / psycopg 后重建锁定文件 |

### 方言抽象的边界（第六批更新）

`dialect.py` 解决的是「SQL 文本差异」。**连接层已在第六批补齐**：
`infrastructure/db/connection.py` 提供 dialect-aware 的 DB-API 包装（占位符适配、
`executescript` 拆分与 DDL 翻译、事务语义），`engine.get_conn()` 按配置的目标打开
连接，非 SQLite 驱动**每线程一条连接**（psycopg 连接不是线程安全的，而 API 走线程池）。
`INSERT OR IGNORE` / `PRAGMA table_info` / `sqlite3.IntegrityError` 这三处绕过方言的
调用点也已收口（分别改为 `dialect.insert_ignore()`、`dialect.table_columns_sql()`、
`connection.integrity_errors()`）。

仍然**没有**做的是真实实例上的验证——见 §六。

## 六、仍待处理

| 项目 | 说明 |
|---|---|
| Redis / PostgreSQL / MinIO **真实服务器**验证 | Redis 已由 fakeredis（真实协议实现）覆盖；S3 有 fake-client 契约测试；`make smoke-services` 可对真实服务执行。剩余是**环境**：需要一台跑着这些服务的机器 |
| PostgreSQL 真实实例验证 | 连接层已完成（见 §七），方言/事务/异常/每线程连接均由 fake driver 覆盖；缺的是在一台跑着 PG 的机器上跑一遍套件 |
| `requirements.lock` 完整性 | 本机装了 boto3/otel/psycopg 后已重建；未安装的 extra（如 moto）仍会如实报告缺项 |

## 七、第六批补齐（2026-09-27）

收尾报告 §4.3 Step 2（PostgreSQL）。这一批**先暴露了一个测试隔离回归**，它比 PG 本身
更严重：全量套件此前已在静默共用开发库。

### 根因：`settings.database_url` 的派生默认值压过了 `DB_PATH`

`config.py` 把 `database_url` 的默认值写成 `sqlite:///{db_path}`——**永远非空**；
而 `engine._configured_database_url()` 会读 `settings.database_url`。于是 15 个测试文件里
16 处 `monkeypatch.setattr(engine, "DB_PATH", tmp)` 全部失效，约 700 个用例写同一个
`admin_data/fixoldimg.db`。实测表现：**106 次 `database is locked` + 70 次
`UNIQUE constraint failed: tasks.id`**（不同用例复用同一个字面量任务 id）。

修法是让代码与 `apply_database_url()` 早已写下的规则一致：**只有环境变量能重定位数据库，
`DB_PATH` 是最终权威**。同时删掉 `configs/{local,docker}.yaml` 里两个与默认路径完全重复的
`database_url` 键——留着它们等于伪装成"部署覆盖了数据库"，正是这个覆盖把 fixture 架空了。

### PostgreSQL 数据路径收口

| 项目 | 处理方式 |
|---|---|
| 绕过方言的 SQL | `INSERT OR IGNORE` → 新增 `dialect.insert_ignore()`（PG 出 `ON CONFLICT (…) DO NOTHING`）；`PRAGMA table_info(users/tasks)` → `dialect.table_columns_sql()`；`cli/migrate_v1.py` 的 `datetime('now','localtime')` 改为参数传入 |
| 裸事务文本 | `claim_next_task` 的 `execute("COMMIT")` / `execute("ROLLBACK")`、`migrate_v1` 的 `execute("BEGIN")` 改为 `conn.commit()` / `conn.rollback()`——两引擎等价，SQLite 行为不变，而 PG 下 psycopg 自己管事务 |
| 驱动异常类 | `except sqlite3.IntegrityError`（`engine.add_user`、`task_service` 幂等竞争）→ `connection.integrity_errors()`；新增 `Connection.rollback_after_conflict()`：PG 上一条冲突语句废掉整个事务，幂等回查前必须回滚，SQLite 侧刻意不动 |
| 每线程连接 | `engine.get_conn()`：SQLite 保持一条共享连接（`check_same_thread=False` + WAL），非 SQLite 走 `threading.local`（psycopg 连接非线程安全，而 API 跑在线程池上）；新增 `engine.close_connections()`，`tests/conftest.py` 的 `isolated_db` 改用它 |
| `open_sqlite` 的方言 | 改为强制 `dialect_for(SQLITE)`。原先它读 `default_dialect()`，在 `FIXIMG_DATABASE_URL=postgresql://…` 的部署里会把 PG 方言（`%s`、无 PRAGMA）套到 sqlite3 连接上 |

### 新增测试

`connection.py` 此前只被 `engine.get_conn()` 引用过，从未用非 sqlite3 驱动执行过——整条
PG 路径是未验证代码。新增：

- `tests/unit/test_db_connection.py`（32 例）：fake DB-API driver 断言**到线上的语句文本**
  （占位符改写、`executemany`、DDL 翻译 + 语句拆分、注释剥离、dict 行访问）、事务与
  `with` 语义、缺 psycopg 时的可操作报错、每线程一条连接 / 同线程复用 / SQLite 仍共享。
- `tests/unit/test_postgres_data_path.py`（8 例）：用 PG 方言真实跑通 `init_db()` →
  `ensure_schema()` → `create_task()` → `claim_next_task()`，断言线上**没有** `PRAGMA`、
  `INSERT OR IGNORE`、`BEGIN IMMEDIATE`，DDL 出现 `BIGSERIAL`， introspection 走
  `information_schema`，claim 走 `FOR UPDATE SKIP LOCKED`。
- `test_sql_dialect.py` 补 4 例 `insert_ignore` 两侧拼法；`test_plugins_and_dialects.py`
  补 PG URL 解析与 `POSTGRES_SCHEMES` 断言。

### 陈旧断言（断言的是"PG 尚不支持"）

5 个用例改为新契约：`resolve_database("postgresql://…")` 返回 `("postgresql", url)`；
只有真正没有驱动的 scheme（`mysql://`）才 `UnsupportedDatabaseError`；
`apply_database_url()` 对 PG 返回 URL 且**不**移动 `DB_PATH`；smoke 脚本的探针名是
`database (postgresql)` 且连不上时报 FAIL 而非 SKIP。

### 基线

`pytest -m "not gpu"`：**728 collected / 全绿（退出码 0）**；`ruff check src tests benchmark
main.py worker.py run.py` 全绿。本机 `mypy 2.3.1 + Python 3.14` 下**本轮改动的文件零错误**；
存量 62 条分布在未触碰的模块（`face_detect`、`download_weights`、若干 `stages`、以及
vendored `basicsr`——它本不该被 mypy 跟随），以 CI 的依赖 pin 为准。

### 仍未完成

真实 PostgreSQL 实例上的套件验证。方言、连接、事务、异常、每线程连接都已就位并由 fake
driver 覆盖，`make smoke-services` 也能对真实实例探测；缺的是环境，不是代码。

## 八、第七批补齐（2026-09-27 下午）

基线：**`729 passed, 30 skipped, 4 deselected`**（`-m "not gpu"`，退出码 0），ruff 全绿，
mypy 在本轮改动文件上零新增错误（存量 62 条不变）。

| 项 | 报告出处 | 处理 |
|---|---|---|
| 模型 sha256 **从空转变为生效** | §3.5.2 | `models/manifest.yaml` 里 5 个模型 `sha256=None` → `verify_hashes()` 全部返回 True 却什么都没查。现在校验以 `config/weights_manifest.json`（26 条真校验和）为准：单文件精确匹配，checkpoint **目录**按前缀覆盖其下所有网络（global_restore 20 个文件、scratch_repair 21 个）。显式 YAML hash 优先于基线；基线没覆盖的模型进 `unverified_models()`，不再被当成"已验证"。热更新的 `_validate` 走同一套（原先只认 YAML sha256，等于对 legacy 链完全不校验）。启动自检已证明的文件记入 `weights_check.VERIFIED`，健康检查不再重复读 1.3 GB |
| `global_restore` / `scratch_repair` 权重路径是错的 | §3.5.2 | 声明照抄了 Face_Enhancement 的 `Setting_9_epoch_100`，该目录并不存在 → `weight_present()` 一直 False（而链路照常工作，因为 legacy CLI 用硬编码路径）。改为 `Global/checkpoints/restoration` 与 `Global/checkpoints` |
| Traces 从接口壳变成接线 | §2.10 | 之前 `grep opentelemetry src/fiximg` 零引用，而且**没有任何生产调用点**（只有测试 import 过 tracing）。现在 `FIXIMG_TRACING=off\|sdk\|otel\|internal`（config 拒绝非法值），`OtelTracer` 用进程级 OTel tracer（exporter 交给部署），缺包时**记 WARNING 而不是假装在追踪**；三个 span：`task.execute`（带 queue wait）、`inference.stage`（task_id/stage/order/device）、`model.load`（version/weight_uri） |
| `install_tracer()` 对生产调用点无效 | — | 模块级 `tracer = _tracer` 在 import 时就绑死了；改为 `_TracerProxy`，每次调用解析 `get_tracer()` |
| `_RecordingTracer` 丢掉失败的 span | — | 原先 `with super().span(...)` 正常返回才 append，异常路径不记录——而那恰恰是最需要看的。改为 finally 里 append |
| integration 层是空壳 | §3.10.1 | `tests/integration/` 只有 `__init__.py`、`pytest.mark.integration` 使用 0 次，而 testpaths 与 CI 步骤名都宣称有这一层。把真正跨层的 4 个模块（queue reliability 23、task queue 5、artifact store 19、worker loop 3）迁入并打 marker，另加 span 发射的跨层测试；marker 描述与 CI 步骤名同步 |

新增测试：`test_model_integrity.py`(18)、`test_tracing_spans.py`(3)、observability 扩充(10)。

### 仍未完成（第七批时的排序，第 1 项已由第八批完成）

1. **Global / Face 的原生 in-process backend**（§2.3、§4.2 Step1-2，报告 §9 认定的最高
   价值项）：~~目前只有 DDColor 常驻~~ → Global 质量路径见第八批；Face 链与 scratch
   分支仍走 `subprocess.Popen`。
2. batch inference / pinned memory / CUDA stream（§3.5.4 剩余三项；tiled、channels_last、
   autocast、inference_mode 已完成）。
3. 真实模型 golden 基线与 Global/Scratch/Face 的 gpu 用例（§3.10.1 / §3.10.2）——第八批
   补了 native↔CLI 等价对拍，但尚无**固化基线**。
4. 时间戳与 `metric_value` 仍是 TEXT、阶段 P50/P95 在 Python 侧算（§2.7）。
5. 发布流水线（§3.11 末端的 release）。
6. Redis / PostgreSQL / MinIO 的真实实例验证（§4.3；PG 覆盖见 §七）。

## 九、第八批补齐（2026-09-27 晚）

报告 §9 认定的最高价值项：**Global 质量修复从"每请求起子进程"改为常驻 in-process
backend**（§3.5.1 / §4.2 Step 1-2）。

基线：**`767 passed, 30 skipped, 9 deselected`**（`-m "not gpu"`），ruff 全绿，mypy 回到
本轮起点（62 条存量、改动文件零错误）；等价对拍 5 例在带权重的机器上实跑通过。

| 项 | 报告出处 | 处理 |
|---|---|---|
| 原生 Global backend | §3.5.1 | 新增 `inference/backends/global_restore.py`：一次构图三张网络（`GlobalGenerator_DCDCv2` ×2 + `Mapping_Model`）、strict 加载、跨请求常驻。它**复用 vendored 代码而不是照抄**——`parameter_set` / `data_transforms` 直接取自 `Global/test.py`，输出走同一个 `vutils.save_image(..., normalize=True)`（那是逐图对比度拉伸而非裁剪，手写就会改像素）。另提供 `run_folder`/`produced_dir` 桥接，folder 化的 stage 与 tiling 路径无需改写 |
| 实现选择与可见回退 | §3.5.4 | `FIXIMG_GLOBAL_NATIVE`（默认开）+ 能力探测（torch / 权重 / 同进程包名冲突）；`select_restore_backend()` 决定用哪个，stage 只认接口。任务元数据写 `"backend": native\|legacy-cli`，`GET /api/v1/models` 新增 `implementation`（`framework` 只说声明，说不清权重在哪） |
| 一棵树一个进程 | 约束 | `Global/` 与 `Face_Enhancement/` 都有顶层 `options/models/util/data`，**一个解释器只能容纳一棵**。所以 scratch 与 Face 链仍留在子进程适配器，并由 `legacy_tree_conflict()` 显式检测，而不是悄悄混用两套代码 |
| **静默随机权重**（缺陷） | §3.5.2 | `base_model.load_network` 找不到文件只打印 "not exists yet"，然后**用随机初始化的网络继续**：任务"成功"、输出是噪声。`--HR` scratch 正踩中——上游根本没提供 `mapping_Patch_Attention/`。现在原生侧加载前校验本分支三个权重；适配器在 `restoration` 目录**部分存在**时拒绝启动（完全没装则交由可用性上报，免得把 CI 与无权重签出变成报错） |
| **Python 3.14 上整条 legacy 链起不来**（缺陷） | — | `Global/options/base_options.py:282` 的 help 串含裸 `%`，3.14 的 argparse 在 `add_argument()` 阶段就抛 `ValueError: badly formed help string` → 子进程路径同样崩。改为 `%%`（旧版本渲染结果不变） |
| 等价性证据 | §3.10.2 | `tests/gpu/test_global_equivalence.py`(5，需权重)：同一张图分别走 native 与 CLI，断言几何/PSNR/SSIM/MAE，CPU 上要求逐像素一致；另钉住"绝不 spawn 子进程""权重常驻不重建""folder 桥接与 jpg 改名"。实测 PSNR `inf`、SSIM `1.0`、max abs diff `0`；单请求 4.5 s → 0.9 s |
| 无权重单元测试 | §3.10.1 | `tests/unit/test_global_native_backend.py`(32)：可用性原因串、包名冲突检测、四种选择与回退路径、进程内单例、绝对路径构造、分支拒绝、缺权重拒绝加载、设备映射表、卸载释放 |
| 顺带修正 | — | `test_legacy_chain.py` 的四阶段接力用例被 native 抢走 stage 1（本机有权重）→ 显式钉 `global_native_backend=false`，它测的本来就是子进程链 |

## 十、第九批补齐（2026-09-27 晚二）

基线：**`798 passed, 30 skipped, 16 deselected`**（`-m "not gpu"`），ruff 全绿；
mypy 由 62 条降到 44 条（第三方跟随策略修正，见下），**本轮改动文件零错误**。

| 项 | 报告出处 | 处理 |
|---|---|---|
| 真模型 golden 基线 | §3.10.2 | 原先只有 synthetic 桩基线。抽出共享度量模块 `tests/fixtures/golden_kit.py`（两套基线必须用同一段代码算 mean/std/PSNR/SSIM，否则不可比），新增 `tests/gpu/test_global_golden.py`(6) + `golden/global_quality.json`（真权重在 96×64 固定输入上的特征：mean 106.49 / std 49.49 / PSNR-vs-input 20.21 / SSIM 0.8745）。含"基线不能来自恒等模型"的反空转断言与"同实例重复推理逐像素一致"的可复现断言 |
| Face 检测原生化 | §3.5.1 / §4.2 Step 2 | 新增 `backends/face_detect_native.py`：dlib HOG + 68 点 predictor 常驻；几何**委托**给 vendored 的 `search` / `compute_transformation_matrix`（`target_face_scale=1.3`、`normalize=False`、256 画布），命名规则 `<stem>_<n>.png` 逐字复现（stage 4 靠它对位）。`Face_Detection/` 不含撞名顶层包，因此可与 Global 原生实现共存于同一 worker |
| 默认关闭并说明原因 | 诚实性 | `FIXIMG_FACE_DETECT_NATIVE=false`。本机 dlib 无法编译（缺 C++ 工具链），像素级等价测试跑不了 → 只有契约测试就默认打开是不诚实的。等价测试 `tests/gpu/test_face_detect_equivalence.py` 已写好并在无 dlib 时如实报告跳过原因；HR 是另一套变换（512 画布），因此 `hr=True` 一律回退适配器，原生侧遇到 HR 直接报错而不是近似 |
| 顺带修出的缺陷 | — | 原生 face bridge 的 `run_folder` 不触发加载（stage 只调 run_folder，不走 infer）→ 补按需 load，并加两例守卫（未 load 就 `_align` → ModelUnavailableError；未 load 的 warmup 必须无害） |
| mypy 会被第三方打断 | 工程 | 新代码 `import skimage` 使 mypy 跟随到 `tifffile`，其 PEP 695 `type` 语句在 `python_version=3.11` 下被判 syntax error，**整个 mypy run 直接停摆**（不是报一个文件，是全盘中断）。对 skimage/tifffile/imageio/matplotlib/dlib/cv2/torch 设 `follow_imports="skip"`：门禁恢复可用，噪声 62 → 44 |
| 清理 | §2.1 | `face_detection` 改走选择器后，适配器类 `FaceDetectionBackend` 已无引用（保留但未注册）；stage 不再硬编码 `"backend": "legacy-cli"`，改取实例的 `implementation` |

## 十一、第十批补齐（2026-09-27 晚三）

基线：**`826 passed, 30 skipped, 16 deselected`**，另加 gpu 标记里本机可跑的部分
（Global 等价 5 + Global golden 6 + face enhancement 等价 2 = 13 例实跑通过）。
ruff 全绿，mypy 44 条存量、改动文件零错误。

| 项 | 报告出处 | 处理 |
|---|---|---|
| 树归属显式化 | §2.9 / §3.13 / §4.2 Step 2 | 新增 `backends/legacy_tree.py`：`FIXIMG_NATIVE_TREE=global\|face\|none` 取代上一批的 `FIXIMG_GLOBAL_NATIVE` 布尔（两个旋钮表达同一件事，迟早互相矛盾）。默认 `global`=已验证那条。**为什么不做 import 隔离**：两棵树都在运行时用 `importlib.import_module()` 解析自身子模块（`Face_Enhancement/models/__init__.py:11`、`util/util.py:149`），换 `sys.modules` 会让动态导入绑到"那一刻"占有包名的树——静默串线。边界只能是进程，也正好对上报告"GPU0 restore worker / GPU1 face worker" |
| Face Enhancement 原生化 | §3.5.1 / §4.2 Step 2 最后一条链 | 新增 `backends/face_enhance_native.py`：92M SPADE 生成器常驻，预处理用 vendored `FaceTestDataset`/`create_dataloader`、模型用 `Pix2PixModel(mode="inference")`、落盘用同一句 `vutils.save_image((gen+1)/2)`。**实测逐字节一致**（CPU delta=0；测试里 GPU 允许 1 级，因 `cudnn.benchmark=True` 会换 kernel）。等价测试把原生侧放进**子进程**（`tests/gpu/test_face_enhance_equivalence.py` + harness），因为 pytest 进程可能已属于 Global 树——测试要尊重这条约束而不是假装它不存在 |
| vendored 解析器只读 `sys.argv` | 约束 | `BaseOptions.gather_options()` 用 `parse_known_args()`（base_options.py:192），没有传 argv 的入口。为免照抄其后的 `semantic_nc`/gpu_ids/batch 断言，改为**加锁临时换 argv** 再调 `parse()`，把后处理留在 vendored 代码里；锁是必需的，否则两个模型同时加载会互相读到对方的 flag |
| 共享设备映射 | §2.1 | `_device_index` 从 global_restore 提到 `backends/__init__.py` 的 `device_index()`，两条链共用同一 `gpu_ids` 约定 |
| stage 3 空裁剪语义 | §3.7 | 上游是"没有裁剪就早退、stage 4 透传原图"，原生桥接照做（返回 root 而非报错），否则人脸数为 0 的照片会从"跳过"变成"失败" |
| 我自己测试的隔离缺陷 | §2.12 | 一例 ownership 测试隐式假设 `models` 干净，但同批别的用例会真去 build Global 的 opt 从而导入 `options/models/...` → 单跑绿、全量红。改为显式用 clean-ownership fixture；并把 stage-3 选择器测试改成"只替换环境、不替换判断"（此前 `native_available` 桩让"关掉开关"这个用例变成空断言） |
| 门禁被第三方打断 | 工程 | mypy 跟随 `skimage → tifffile`，其 PEP 695 `type` 在 `python_version=3.11` 下报 syntax 并**中断整个 run**；对 skimage/tifffile/imageio/matplotlib/dlib/cv2/torch 设 `follow_imports="skip"`，噪声 62 → 44 |

新增测试：`test_native_tree_ownership.py`(26)、`tests/gpu/test_face_enhance_equivalence.py`(2)、
`tests/gpu/face_enhance_native_harness.py`（子进程执行桩）。

## 十二、第十一批补齐（2026-09-27 晚四）

基线：**`826 passed`（CPU）+ 本机可跑的 gpu 标记 18 例实跑通过**，ruff 全绿，mypy 44 条
存量、本轮改动文件零错误。

补完 §4.2 Step 2 的最后一个缺口：**scratch 分支原生化**（它与 quality 同属 Global 树，
所以一个 global worker 就能同时常驻两条分支）。

| 项 | 报告出处 | 处理 |
|---|---|---|
| 原生 scratch 后端 | §3.5.1 / §4.2 | `NativeScratchRepairBackend`（继承 quality 后端，复用 `_encode`/`_require_weights`/策略钩子）：常驻检测 UNet（`FT_Epoch_latest.pt`）+ `VAE_A_quality`/`VAE_B_scratch`/`mapping_scratch` 三元组。检测预处理复用 `Global/detection.py` 自己的 `data_transforms`/`scale_tensor`（以私有模块方式加载），mask 落盘用同一句 `save_image(..., normalize=True)`，恢复步骤**从这些文件读回**——文件中转正是逐位一致的原因，而且 mask 本来就是 UI 要展示的产物 |
| 等价性 | §3.10.2 | `tests/gpu/test_scratch_equivalence.py`(5)：真划痕样本上比对**恢复图与 mask 文件**，两者 max abs diff 都是 0（PSNR `inf`）；目录布局（`stage_1_restore_output/masks/{input,mask}`、`restored_image`）逐一对齐，后续阶段分辨不出；计时 6.6s/请求 → 一次性 2.2s + 1.7s/图。另钉住"HR 必须拒绝"（`mapping_Patch_Attention` 没提供，vendored loader 不会报错只会跑随机权重）与"检测器与三元组跨请求常驻" |
| 无权重单元测试 | §3.10.1 | `test_global_native_backend.py` 追加 14 例：scratch 需要比 quality 多一张网络（半装不算装）、选择与回退、注册表与实际服务者一致、分支守卫互斥（quality 后端拒绝 scratch opt、scratch 后端拒绝 quality opt）、未加载时 `_detect_mask`/`_restore_from_files` 报领域错误、health 报出 detector 路径 |
| 声明与陈旧措辞 | §2.1 | manifest 里 `scratch_repair`/`face_enhancement` 的 `framework` 改为如实（`pytorch`；`face_detection` 是 `dlib`，它与树归属无关所以开关独立）；registry 与 quality 后端里"scratch 仍走子进程"两处说明已过期，一并改掉；deployment 的归属表与等价表补上 scratch 行 |

顺带自纠：两条**上一批我自己写的**契约测试（"scratch 永远走适配器"、注册表一致性用例）在
scratch 转原生后变红——它们的断言把"当时尚未迁移"当成了"永远如此"。改成显式控制
`scratch_native_available` 的两条分支，语义从"它不行"变成"它缺权重时才行不通"。

## 十三、第十二批补齐（2026-09-27 晚五）

基线：**`914 passed`（CPU）+ 本机可跑的 gpu 标记 18 例实跑通过**，ruff 全绿，mypy 44 条
存量、本轮改动文件零错误。本轮把 §2.7 数据层剩下的三条全清了，并顺手补上 §3.11 末端的
release 流水线。

| 项 | 报告出处 | 处理 |
|---|---|---|
| 时间戳不再是"字符串就完了" | §2.7 | 新增 `infrastructure/db/timestamps.py`：统一成**定宽 UTC + 微秒 + `Z`**（`2026-09-27T04:53:04.246659Z`）。原来的 `datetime.now().strftime("%Y-%m-%d %H:%M:%S")` 一次踩三个坑——秒级精度让同秒提交的 FIFO 顺序变成运气、无时区让跨 DST 的租约长度不确定、格式在各调用点各写一遍。定宽的意义是 TEXT 上的字典序比较**就是**时间序比较，两个引擎都能用索引做。**为什么列类型仍写 TEXT**：SQLite 根本没有 datetime 类型，ISO-8601 TEXT 就是它的规范瞬时表示；PG 可以上 `timestamptz`，但仓储里十来个租约比较传的是 Python 字符串，而 psycopg3 把 `str` 绑定成 `text`，要改就得逐处加 `::timestamptz`——这台机器上一条都执行不了。改成 bug 的是值的精度和缺失的时区，不是声明类型（deployment.md 里把这条决定写下来了） |
| metrics 不再是字符串 | §2.7 | `metric_value` → `REAL`，另加 `metric_text` 装 REAL 装不下的（位相同图像的 `inf`、标签），读取侧 `COALESCE`。JSON 里数字终于是数字（`{"psnr": 22.5}` 而不是 `"22.5"`），API 文档同步。`split_metric()` 单独可测，10 组取值钉住边界 |
| 统计不再拉全表 | §2.7 | `task_stats()` 的 P50/P95 原来是"把窗口内每条 duration 读进 Python 再 sort"，换成**一条**用窗口函数的 SQL（`ROW_NUMBER() OVER` + 相关子查询取行）。两引擎同一句 SQL、同一个**最近秩**定义（`ceil(p*n)` 夹到 `[1,n]`）——没用 PG 的 `percentile_cont`，因为 SQLite 没有对应物，那样同一批数据两台机器会报不同数字。n∈{1,2,3,4,5,7,10,21} 与 stdlib `math.ceil` 参考实现逐一对齐 |
| 迁移 | §4.3 | 新增 `0003_typed_instants`：值转换（`datetime(x,'utc')` / `AT TIME ZONE`）+ `metric_value` 换类型，且**先把非数值搬进 metric_text 再 CAST**，否则 `inf` 会被静默变成 NULL。round trip 实测可逆（本地 08:00 → UTC 00:00Z → downgrade 回到本地 08:00，再 upgrade 仍是 00:00Z）。第一次写 downgrade 时漏了 `'localtime'`，探针跑出来 8 小时漂移才发现 |
| "代码已经前进、库还没迁" | §2.7 的落地风险 | 就地升级若用 `stamp` 跳过 `upgrade`，读写侧都容忍两种格式，于是**先是安静正常**、然后 `lease_until` 的字典序把一条还没到的本地 23:00 判成早于 canonical 02:00Z → 正在跑的任务被重新入队。加了 `legacy_instant_rows()` + `fiximg-db check`（`make db-check`，非零退出）把这件事变成运维可见的一步。**没有**做成启动扫描：那要按列全表 COUNT，而"这库迁过了吗"是运维的问题，不是每次启动的问题 |
| release 流水线 | §3.11 末端 | 新增 `.github/workflows/release.yml`：tag `vX.Y.Z` 必须等于 pyproject 版本，否则直接失败；wheel 装进干净 venv 后**从安装包**构建后端注册表（源码树永远"看起来完整"，只有装出来的轮子会暴露漏打的文件——本地 `python -m build` 已验证 9 个 backend 模块进包、删掉的 `face.py` 不在）；api/worker 双镜像推 GHCR 带 SLSA provenance + SBOM；GitHub Release 附 sha256 与自上个 tag 的 changelog。手动 dispatch 默认 dry-run，且不推送时**明写**"(not pushed — dry run)"而不是留个空字段配绿灯 |
| 顺手修掉的三个真缺陷 | — | ①`worker._queue_wait_seconds()` 用 `strptime(created_at, "%Y-%m-%d %H:%M:%S")`：值换成 canonical 后它会 `ValueError` → 排队等待指标静默消失，改为走 codec；②`ensure_schema()` 的 `_DDL_DONE` 是**进程级**闩锁，但库可以被 fixture/嵌入式改指向，于是"表根本没建"就返回了——改成按 `(kind, target)` 记账（`False`/`None` 仍表示"没建过"，11 处测试钩子照常工作）；③`get_conn()` 的缓存句柄不认目标，同理改成按 `(kind,target)` 自愈 + `Connection.closed`，12 处 `engine._conn = None` 变成 `close_connections()`，`migrate_v1` 里那处生产代码也一并改掉 |
| 测试隔离的又一例 | §2.12 | 装上 alembic/SQLAlchemy 后（本机此前缺）迁移测试第一次真跑，立刻暴露 `migrated_db` fixture 只清了 `_conn`、没清线程本地句柄 → `stamp` 写进了上一个用例的库。同一批还有一条**过期断言**：`test_a_non_sqlite_url_fails_at_migration_time` 断言的是"PG 不被支持"，而 PG 数据路径上一轮已经做完，改成断言未知 scheme 仍 fail-fast + PG URL 落到 `postgresql+psycopg://` |
| 死代码 | §3.6 | 删掉 `backends/face.py`（`FaceDetectionBackend` + `FaceEnhancementBackend` 两条 CLI 适配器）：两个选择器都直接返回 `LegacyCliBackend`，全仓零引用。留着一个同名实现正是报告说的"语义重复" |

顺带：`migrate_v1._ensure_v2_schema()` 原本自己抄了一份"后来加的列"清单（已经漏了
`metric_text`），改为直接调 `task_repository.ensure_schema()`——一处定义。



## 十四、第十三批补齐（2026-09-27 晚六）

基线：**`929 passed`（CPU）+ 本机可跑的 gpu 标记 18 例实跑通过**，ruff 全绿，
mypy **0 条**（并打开 `check_untyped_defs`）。本轮做了两件事：把"类型检查"从**报告**
变成**门禁**，以及对 §3.5.4 剩下的三项给出有依据的结论而不是硬凑实现。

| 项 | 报告出处 | 处理 |
|---|---|---|
| mypy 44 → 0 | 工程化 §3.11 | 44 条存量全部清掉。根因分四类：①**8 处 `os.path.join(context.run_dir, ...)`** 而 `run_dir: str \| None` —— 不在调用点各贴一句 `or ""`，而是在 `StageContext` 上加 `path_in_run()`，缺 run_dir 时抛领域错误（原来会在 os.path 里 TypeError，离原因好几层）；②`Image.LANCZOS/BILINEAR` 改成 `Image.Resampling.*`（Pillow 12 的类型桩里旧别名已删，运行时还在，属于"能用但已废弃"）；③`Optional` 却直接当值用的模块级状态（`_YUNET`/`_DLIB_DETECTOR`/`_model`/`_topology`/`FileLock`）逐个写明类型；④`Connection.database_target` 从"动态挂属性"改成正式字段（上一轮加的缓存自愈本来就依赖它）。**顺带修掉一个真缺陷**：`task_service.task_result_path()` 标着 `str \| None`，但两条"没有文件"的路径都会抛错、永不返回 None —— 那个假的 Optional 逼下载路由去判断一个不存在的分支 |
| 门禁本身 | 工程化 §3.11 | CI 的 typecheck job 去掉 `continue-on-error`：它之前**只出报告不拦人**，而 44 条发现里就有几条正躺在"忘了写注解的文件"里。同时打开 `check_untyped_defs = true`——未标注函数体也检查，否则"忘记写注解"恰好等于"忘记被检查"（`skimage` 的隐式再导出和 `Image.BILINEAR` 就是这么躲过一整个版本的）。vendored 的 `basicsr`/`ddcolor` 设 `follow_imports="silent"`：**上游复制来的代码不是我们的类型债**，改它还会让它和上游不再逐字一致 |
| 基准的 p95 一直是假的 | 验收 性能「p50/p95 基线」 | `benchmark/common.py` 的 `_percentile` 写作 `int(n*pct)-1`：n=2 时它返回 `ordered[0]`，也就是**把最小值当 p95 报** —— 而 CI 恰好就用 `--runs 2`。修成最近秩（与 `task_stats` 同一套定义），并补三类测试：n∈{1..100} 对齐参考实现、`min ≤ p50 ≤ p95 ≤ p99 ≤ max` 顺序不变量（旧实现必红）、以及**基准与数据库对同一批 duration 报出同一个数**。实测 `--runs 6` 从 `p50 .0543 / p95 .0514` 变成 `p50 .0513 / p95 .0565` |
| §3.5.4 剩余三项 | §3.5.4 | **结论是不实现，理由如下**（报告原文也是"在确认模型兼容后**逐项加入**"，且这三项不在附录 B 的 37 条里）。pinned memory / CPU→GPU prefetch / CUDA stream / batch inference 共享一个前提：**这段代码得自己拥有张量搬运与前向调用**。查下来的事实是本项目没有这样的位置：DDColor 走 modelscope 的 `pipeline.process()`（搬运在它内部），Global/Face/Scratch/质检四条原生链是**逐字节复刻** vendored `test.py`/`batch.py`/`detect_all_dlib.py` 的调用序列（包括它在第几步 `.to(gpu)`），face embedding 走 dlib 的 `compute_face_descriptor`。要"加 batch"就得重写这些内部序列，直接抵掉已证明的等价性——而等价性是这个 runtime 唯一的正确性凭据。另一条路（先写 `infer_batch`/`BatchPolicy`/`claim_many` 脚手架等着以后接）在这个环境里只能靠桩证明，正是本报告反复点名的"看着完成、从没跑过"。**因此这三项记为"有前提的待办"而非"已完成"**：前提是存在一个自己掌控张量路径的原生模型（新接入的模型天然满足），以及一台能量出增益的 GPU 机器 |
| 差点埋进去的坑 | §2.12 | 为消除一条 "skimage 未显式导出 img_as_ubyte"，把 `from skimage import img_as_ubyte` 换成官方位置 `from skimage.util import ...` —— **红了 6 个测试**：face detection 的测试一直在 patch `skimage.img_as_ubyte`，也就是生产代码"顺手"用的那个再导出点；换声明位置后桩打不通了。处理是保留正确的导入、把 patch 挪到同一点并注明"要在生产查找的名字上打桩，否则这是个假桩"。教训：**类型桩逼你选的导入路径，同时就是测试桩的位置**，改它必须连带看谁在 patch 它 |

---

## 十五、第十四批补齐（2026-09-27 晚七）

基线：**`981 passed`（CPU）+ 本机可跑的 gpu 标记 18 例实跑通过**，ruff 全绿，mypy 0。
本轮先用 agent 把附录 B 的 37 项按"生产调用点 + 测试"两条证据重审了一遍，再逐条补真缺陷。
重审本身也值得记一句：**`requirements/` 里 test 组的锁文件在我装完 fakeredis 之后就不一致了，
而这件事只有把依赖装上才会暴露**——和上一轮"迁移测试因缺 alembic 整文件 skip"是同一类。

| 项 | 报告出处 | 处理 |
|---|---|---|
| **Redis worker 会饿死** | 验收 架构5 / §3.13 | `QueueBackend.enqueue()` 在生产里**没有调用点**：`claim()` 只读 stream，DB 只决定"一次 claim 意味着什么"。提交只写行 → stream 永远空 → `FIXIMG_QUEUE_BACKEND=redis`（compose 里就写着）的 worker 一个任务也拿不到。SQLite 那边 `enqueue` 是文档化的 no-op，所以这个洞在默认拓扑上完全不可见。改为 `submit_queued` 成功后尽力投递（Redis 抖动不能把已入库的提交变成失败），并补两条测试：提交→stream→claim 端到端、投递失败时行仍在且仍可 claim。**用变异测试确认新测试有效**：注掉那一行，测试立刻红 |
| 基准不是门禁 | 验收 工程化37 | `benchmark.latency` 永远 `return 0`，所以 CI 那步"benchmark regression"是一张没人读的表。合成阶段睡的是已知时长，**睡意之上的部分就是平台开销**（领取、调度、artifact 行、日志），于是给它加了 `--max-overhead`（默认 1.0s，只对最小尺寸设限——图像准备按像素涨，拿 512 的天花板去量 4096 只会假红）。实测 overhead `+0.0012s` |
| 三个 p95 不是同一个数 | §7.1 / §3.10.3 | 上一轮我在 SQL 里实现最近秩，这轮发现 Python 侧有**两份各自为政的 percentile**：`benchmark.common`（`int(n*p)-1`，n=2 时把最小值当 p95，而 CI 正是 `--runs 2`）和 `observability.metrics._Histogram.percentile`（同一个公式，喂给 Prometheus）。统一成一个 `nearest_rank()`，基准改为委托它；补"min ≤ p50 ≤ p95 ≤ p99 ≤ max"顺序不变量与 SQL/基准同数对照测试 |
| TTL 只在同步路径生效 | 验收 可靠性12 / §2.8 | 文档里写着"Disk growth: FIXIMG_RESULT_TTL"，但 `maybe_purge()` 只挂在 `PipelineOrchestrator.run()`（同步路径）上 —— **API + 独立 GPU worker 拓扑下回收根本不会发生**，磁盘慢慢涨。改成 worker 的周期扫描里按 `FIXIMG_ARTIFACT_SWEEP_SECONDS`（默认 900s）固定扫；同步路径那处保留（无 worker 的部署靠它）。补 4 条测试：过期删除/新目录保留/**绝不碰 tasks_root 之外**、概率为 0 不扫、worker 按自己的节拍扫、`result_ttl` 真被用到，外加 API 侧 410(ARTIFACT_EXPIRED) 与 409(TASK_NOT_READY) 的区分 |
| UI 伸进 pipeline 内部 | 验收 架构2 | `ui/history_panel` 直接 `from fiximg.inference.backends.legacy_cli import list_images` 去数人脸裁剪目录。目录布局是 pipeline 知识 → 下沉成 `task_service.first_face_crop()`。**并且把三条验收分层规则做成静态门禁** `tests/unit/test_architecture_boundaries.py`（AST 扫 import）：UI 不得 import `fiximg.inference.*`；API 不得 import 具体后端/阶段；Stage 不得 import HTTP/Gradio/DB/queue。规则范围**只写报告要求的**——"UI 不得碰 DB"是我自己加的，而 `ui/state.py` 现在确实直读 `user_repository`，把代码不遵守的规则写成门禁只会得到一次永久豁免。同样用变异确认：给 UI 塞一行违规 import 即红 |
| 组件里两个"死导出" | §3.2 | `comparison_slider` / `preview_row` 写着"给历史面板用"，实际零调用，而历史面板自己内联建了同样的两个 slider 和四格预览。改成**用**这两个 helper（消除重复定义），而不是删掉 |
| 锁文件检查在非全量环境必红 | §3.12 | `export_locks.py --check` 用装好的环境算依赖闭包，CI 只装 `.[test]` → `dev`/`gpu` 组闭包算不全 → 与全量环境生成的文件必然不一致 → 那步门禁在任何干净 CI 上都是红的。改为**只校验根包都装了的组**，其余打印"not verified here"（能算的地方仍严格：CI 会校验 `runtime` 与 `test`） |
| Prometheus 少两个分位 | §3.4 | 直方图 summary 算得出 p50/p95/p99，文本暴露只写 avg 与 p95 —— JSON 里有、抓取端没有。补齐并加测试 |
| 3. 可替换性 | Redis 投递缺失、MinIO 未接进 pipeline | 队列半边当场补完（提交即投递 + 端到端测试）；存储半边在下一批补完，见第十七节 —— 当时它确实没接：| 顺手 | — | `_collect_stage_metrics` 平铺合并让任务级 `gpu_peak_mb` 变成"最后一个阶段的值"（两个各到 4GB 的阶段，任务峰值仍是 4GB，不是"谁最后报"）；`MemorySampler` 在无 CUDA 主机上曾报 `0.0 MB` 这种**看着可信的假读数**，改为"没开始测就不报" |

**两处文档比代码先行**：`docs/v3-gap-analysis.md` 早先就把"p50/p95 基线：CI 门禁"和
"queue wait 基线：`benchmark.throughput`"写成已完成——前者的命令跑不了红，后者压根没测排队
等待。本轮分别补上（`--max-overhead`；throughput 打印 worker 自己观测的
`queue_wait_seconds` p50/p95/avg，n=8 实测 p50 963ms），并说明为什么**只报不设限**：单消费者
下第 N 个任务的等待是 `tasks × stage-delay` 的函数，卡上限只会量到命令行参数。


**一条"看着严格、其实是掷硬币"的断言**：scratch 等价测试原来无条件要求
`mae == 0`。这一轮它开始间歇性变红（同一份代码 3 次跑挂 2 次），先按 import 改动去
二分，结果证明**跟本轮任何改动都无关**：`Image.BILINEAR` 与
`Image.Resampling.BILINEAR` 是同一个 IntEnum 值（实测 `==` 且 resize 结果逐位相同）。
真正的机制是 torch 在 CPU 上的卷积归约顺序会随线程调度变化，而检测器 sigmoid 恰好压在
0.4 阈值上的那几像素会翻面，mask 又会被合成进输出——**vendored 流水线自己跑两遍也会
差 1 LSB**。实测负载低时参照自己 3×3 全为 `(0, 0)`，负载高时才有抖动。于是契约改成：
每次运行先测参照的噪声地板，native 不得比"参照与自己"更远；参照逐位一致时，照旧要求
native 逐位一致（deployment 的等价表把这个限定写清楚了）。顺带这次排查也说明：**一个
偶尔红的断言如果比"改了什么"更好解释，先怀疑断言的假设**。

## 十六、第十五批补齐：把"门禁"这个词落实（2026-09-27 晚八）

按固定门禁清单重跑一遍（4 条 + CI-only job 本地复现 + 逐条 skip 归因 + 每条门禁的
"能不能变红"验证）。这一批基本没有新功能，全是**让检查真的能拦人**。

| 门禁 | 本机结果 | 它怎么被证明能红 |
|---|---|---|
| ① `pytest -m "not gpu"` | `984 passed, 2 skipped`（23 deselected = gpu 标记） | 上一批起每条新测试都做过变异：Redis 投递注掉→红、架构边界塞一行违规 import→红、基准阈值调紧→exit 1 |
| ② ruff（CI 的完整路径表，含入口脚本） | All checks passed | — |
| ③ `python -m mypy` | 0 发现 / 118 文件（`check_untyped_defs` 已开） | CI 的 `continue-on-error` 已删；现在**由测试** `tests/unit/test_ci_gates.py` 禁止任何 step 再挂它 |
| ④ gpu 标记里本机可跑的部分 | `18 passed, 5 skipped`（CUDA+DDColor 权重 4 例、dlib 1 例） | 见下一行 |
| scratch 等价 | 5 例通过 | 无条件 `mae == 0` 在负载高时自己会红 —— 参照流水线本身不是位稳定的，契约改成对着参照的噪声地板比 |

**两处"永远是绿灯"的 CI 步骤**（本轮新发现并修掉）：
`pip-audit ... --ignore-vuln GHSA-0000-0000-0000 || true`。占位符 ID + `|| true` 的含义是
"依赖审计这一步不可能失败，输出没人在读"。改成：审计默认失败即拦，豁免必须是
`config/audit_ignore.txt` 里一行可 grep 的 GHSA/CVE（写清理由、受影响 pin、复查日期），
且 `pip-audit` 对已失效的 ID 会直接报错，所以豁免不会烂成永久静默。同时新增
`tests/unit/test_ci_gates.py`（5 例）把这条规则钉在 CI 配置上：任何 step 不得设
`continue-on-error`；**脚本最后一条命令**不得 `|| true` / `&& true` / `exit 0`（中间行的
`|| true` 允许，release job 探上一个 tag 就是合法用法）。两条规则都做了变异验证。

**skip 全部归了因**：`needs a usable GPU`、`POSIX file modes are not enforced on Windows`
（CPU 侧 2 条）、gpu 标记里 `requires CUDA GPU + DDColor weights`(4) 与
`dlib is not installed`(1)。没有一条是"代码路径自己 skip 自己"。

**本轮顺带修的两个跨平台问题**（都是想本地复现 CI job 时才暴露的）：
`requirements.txt` 与 `requirements/*.txt` 的注释里带 em-dash，Windows 非 UTF-8 区域下
`pip-audit`（按平台编码打开文件）直接 `UnicodeDecodeError` → 所有依赖清单改为纯 ASCII；
`export_locks.py` 的 HEADER 同样处理。另外 `pip-audit` 在这台 Python 3.14 上装不了
`numpy==2.4.6`（没有 cp314 轮子），所以**审计结果本身只能由 CI 第一次运行给出**——
这点写进了 deployment，没有假装本地跑过。

**仍未完成**：只剩 `#23`（产物发布接进 `ArtifactStore` 协议：`FIXIMG_STORAGE_BACKEND=s3`
目前不改变字节落点）和 `#17`（batch/pinned/stream，前提是自己掌控张量路径 + 有 GPU 量增益），
以及依赖真机的那批验证。镜像与 compose 校验在本机没有 docker，属环境缺口，不是代码缺口。

## 十七、第十六批补齐：产物存储从"装饰"变成真的（plan §2.8）

基线：**`1013 collected / 全部通过`（CPU）+ 本机可跑 gpu 标记 18 例**，ruff 全绿，mypy 0。
上一节记下"存储半边仍未接"，这一节把它接上；接的过程里被自己的规则逼出了两个额外结论。

| 项 | 报告出处 | 处理 |
|---|---|---|
| 发布 | §2.8 | `PipelineOrchestrator._publish_output()`：结果图按 run 树自身的相对键（`2026/09/<task>/output/final.png`）上传，键记进新的 `artifacts.uri`（迁移 `0004`，可逆）。**本地 store 一律 NULL**：所有读法都是 `row["uri"] or row["path"]`，本地模式塞一个*相对*键等于让读者拿它当 CWD 下的路径 —— 那会把这个改动从"补功能"变成"给单机部署埋雷" |
| 下载 | §2.8 | `task_result_path()` 在本地文件不在时先走 store 回捞，捞不到才 410。**修复前**：API 节点与 worker 不同机时，任务明明成功、下载却回答 `410 Gone` —— 因为"文件在不在这儿"是下载路径问的唯一问题 |
| 回收 | §2.8 | 只发布不回收等于把 TTL 变成假话（对象存储只涨不消）。`purge_expired_published(cutoff)` 按 `artifacts.created_at` + `FIXIMG_RESULT_TTL` 删对象并清 `uri`（行留着当历史）；本地后端直接返回 0，**避免两个回收器抢同一批文件**。顺带用上上一批的 canonical 时间戳：cutoff 就是 `timestamps.cutoff(result_ttl)` 的字符串比较 |
| 门禁的副作用 | 工程化 | 装 `moto` 才能测 S3 那半边，而 `moto` 一装就把一个**一直在 skip 的老测试**变成了真跑，并且当场失败：它断言 `s3_store.client.objects[...]`，那是手写 fake 才有的字典 —— 说明那条测试从来没碰过 boto3。同时我给新 fixture 起的名字**遮蔽**了文件里原有的 `s3_store`，让老测试拿到了 moto store。两处都改了（新 fixture 更名 `moto_s3`，老断言改成协议方法）。教训：**"这条测试一直是绿的"不等于"它跑过"**，`importorskip` 尤其容易藏这种 |
| 可复现性 | 工程化 | `moto`/`boto3` 加进 `test` extra，否则 S3 这条腿"只在某台恰好装过 moto 的机器上跑过"——正是本项目反复点名的那类。CI 的 `.[test]` 因此会真的执行它 |
| 变异验证 | — | 三条都做了：去掉 `_recover_from_store` → 回捞测试红；去掉 `store.delete` → 回收测试红；本地模式那条同时守住"`uri` 不许在单机下被填" |

**仍然没接的那一半**：真实 MinIO/S3 端点。`moto` 走的是同一套 boto3 代码路径，但它不开 socket、不验签、不管 endpoint/bucket 策略；`make smoke-services` 才是那个能证伪的检查，而它需要一台实例。这条写在 `docs/deployment.md` 的 Scaling out 末尾，而不是藏在某个"已支持"的表格里。

## 十八、第十七批补齐：把服务真跑起来读数字（plan §3.5.5、§3.9、§3.5.4）

基线：**`1008 passed, 2 skipped, 24 deselected`**（CPU）+ 本机可跑 gpu 标记 19 例
（连续 5 次全绿），ruff 全绿，mypy 0（118 文件）。

这一批不按报告条目推进，而是**把 V3 平台真启起来跑一次 `restore`**，再逐个读它吐出来的
数字。七处里只有第一处表现为失败，其余六处都是"通过"的样子——状态 completed、HTTP 200、
字段齐全，只有数值本身是错的或自相矛盾的。

| 现象 | 根因 | 处理 |
|---|---|---|
| 任务失败（三次重试后 failed） | 这台机器编不出 dlib，而 face 链把"依赖不存在"当运行时错误抛 | preflight 降级：face_detection 发现 dlib 不可导入即记 `skipped`，后面阶段直接透传恢复结果。**修复前**一次已经成功的整图修复会连人带结果一起被丢掉 |
| 下载 296×448，上传 298×450 | vendored 修复链把边长对齐到 4 的倍数（检测器对 16），这条舍入不在任何尺寸策略里 | §3.5.5：`_execute_core` 出口无条件 `restore_original_size()`（几何本来就对时是 no-op）。顺带才暴露出指标一直在拿两种几何互比 |
| `ssim: -0.6657` | `_calculate_ssim` 把 `sigma2_sq` 写成 `E[y²] − μ_x μ_y`（下一行的交叉项抄到了上一行），应为 `E[y²] − μ_y²` | 改公式，并用 skimage 钉住：同一对图两实现给出 `0.661758` / `0.661758`。这个偏差**两个方向都会犯**——正常修复被压到 0.099（真值 0.652），无关噪声被抬到 0.0875（真值 0.0093），后者恰好是朝"看起来正常"的方向 |
| 报告写 `Face Count 1`，metrics 写 `face_count 0` | §17 身份指标把"自己的检测器看到几张脸"写进了 face 链的 `face_count` 键，`_execute_core` 合并时评估侧覆盖阶段侧；标签 "Enhanced Faces" 更在宣称增强过 1 张脸，而这次一张都没增强 | 键改名 `identity_faces_input/output/paired`，阶段键不再被触碰；标签改成检测器真正测的东西（Faces Detected (in)/(out) / Faces Compared） |
| untrained 描述子给出 0.98，读起来像生物识别命中 | `backend: gradient_fallback` 只写了名字没写含义 | 每种 backend 配一行说明；fallback 明说"0.98 只代表这块脸看起来还是这块脸" |
| `/models` 里 face_detection / face_enhancement 的 `implementation` 是 `null`，ddcolor 是 `"unknown"` | 一个 `LegacyCliBackend` 类服务三个阶段，而它的 `name` 是类属性 `global_restore`；`describe_all()` 拿**上报的名字**建索引，于是两个 face 阶段从字典里整个消失 | `describe_all` 以注册键为准命名每一条（键才是模型名）；DDColor 补 `implementation = "native"`。§3.5.4 的"可见回退"到这一步才真的可见 |
| face 阶段状态 `completed`，其实什么都没做 | `StageStatus.SKIPPED` 声明了若干批没有写入方 | §3.9：metadata 带 `skipped` 即写 `skipped`，事件 `stage.skipped`（每阶段仍恰好一个终止事件），UI 加 ⊘ 并显示原因，`docs/api.md` 记状态与事件词表 |

**跑第二遍才发现的那一个**：`tests/gpu -m gpu` 大约三次里红一次，红的总是 scratch 等价那两条。
把参照流水线连跑五次做逐对比较（10 对）：mask 全部逐位一致，图像 6 对一致、4 对差
`1 LSB / 3 像素`——**参照自己是双峰的**；native 在同一进程里跑三次逐位一致。原来那条"两次
参照一致就要求 native 逐位一致"的升级规则，本质是拿"这次抽到哪一支"当质量信号：它既能假绿也
能假红，而那 1/3 就是假红。改成三条各自站得住的断言：参照仍须落在实测带内（带变了是换机器，
不是回归，单独报出来）、mask 必须逐位一致（它本来就有这个性质）、图像允许一个支步，另加
**native 自身两次运行逐位一致**这条真正精确的要求。文档里"检测器 sigmoid 卡在 0.4 阈值"那句
也被这次测量推翻了：mask 从没动过，分支在 triplet 修复的卷积里。

**自己读错的那一处**（记下来免得再犯）：我一度以为 `GET /tasks/{id}` 的 `evaluation_text` 是空的。
它不在那个响应里，它在 `/report`。这不是缺陷，是我拿错了端点——和"数字看着正常但没人核对过"
是两类问题，处理方式也相反：后者要加对照，前者只要别猜。

变异验证：SSIM 公式改回去 → 两条红（skimage 对照、"噪声打分必须 <0.05"）；`face_count` 键改回去
→ 冲突测试红；`describe_all` 去掉注册键 → 两条红；DDColor 摘掉 `implementation` → 红；skipped
状态改回 completed → e2e 红；scratch 的图像/mask/自再现三条各用"把输出涂改 1 LSB"证明是活的
（`REFERENCE_LSB = 0` 那个变异**故意测不出红**——抽样运气好时它本就不该红，这正是把常数当地板
的理由，注释里写明了）。

## 十九、第十八批补齐：报告口径按任务类型对齐（plan §16、§3.9）

基线：**`1012 passed, 2 skipped, 24 deselected`**（CPU）+ gpu 标记本机 19 例，ruff 全绿，mypy 0。

上一批是"把服务启起来读数字"，这一批读的是**同一次运行里不同块之间的口径是否一致**。两条
缺陷都来自"同一件事在两个地方各判断一次"：

| 项 | 现象 | 根因 | 处理 |
|---|---|---|---|
| §16 差值指标 | `auto_restore` 的报告里没有 PSNR/SSIM/MAE，而 `restore` 有；但 §17 身份指标**一直**给 `auto_restore` 算 | 身份那行的判断写成 `_EVALUATED_TYPES \| {"auto_restore"}`，而差值那行只用 `_EVALUATED_TYPES`。同一物理流水线换个入口就少一块报告 | `auto_restore` 进 `_EVALUATED_TYPES`，身份那行的并集随之删掉。**当初排除它是有道理的**：auto_restore 对灰度图会追加 colorization，上色后与灰度原图的差值必然难看——但正确做法是照报并说明，而不是把所有非灰度 auto_restore 的数字一起扔掉，所以加了 `_colorization_note()`：报告里多一行"本次还做了上色，低值不等于修复退化"。说明**按实际跑过的阶段判断而不是按入口类型**，因为 `options={"auto_colorize": true}` 能让一条普通 `restore` 长出一模一样的形状——那条路径也顺带被覆盖了。`colorize` / `detect_scratch` 仍排除（拿灰度原图比色、把 mask 当照片，都不是有意义的差值） |
| §3.9 降级原因 | 阶段按 plan 顺序给 `degrade_note` 赋值，**最后一个匹配的覆盖前面的**，而 warp_back 永远是最后那个 | 循环列了三个候选阶段，却只有"谁排最后"真正起作用；warp_back 跳过时 metadata 里没有 report → 返回 None → 把 global_restore 已经给出的理由擦干净 | 改成"第一个有话说的人赢"（拿到非 None 即 break）。今天只有 warp_back 会产生 report，所以这是潜伏缺陷而不是线上缺陷——但 auto_restore 恰恰把阶段顺序变成 global/scratch → face → warp_back → colorization，是这种覆盖最容易咬人的位置 |

实机复核（灰度输入 `auto_restore`，走 HTTP）：

```
status: completed
stages: scratch_repair=completed, face_detection=skipped, face_enhancement=skipped,
        warp_back=skipped, colorization=completed
metrics: psnr 17.60 / ssim 0.6257 / mae 0.1082      ← 修复前这一块是空的
⚠ This run also colorized a grayscale input: ...    ← 数字难看，但说明也到了
```

同一趟运行也把上一批的 `skipped` 状态在真实 HTTP 路径上验了一遍：三个阶段是 `skipped`，任务
整体仍 `completed`，而分析器那边 `face_count: 1`——两组键各说各的，正是改名之后才能同时成立
的那种局面。

变异验证三条：把 `_EVALUATED_TYPES` 改回两种 → auto_restore 与 restore 的指标集合不一致，红；
把 `break` 去掉（恢复"最后写入者胜"）→ 早先阶段的降级理由被后来的沉默擦掉，红；摘掉
`_colorization_note()` → 上色说明那行不见，红。

**一次自己犯的错，记在这里**：清理实机 scratch 时写了 `rm -rf storage/tasks/2026/09/2026092*[0-9]`，
通配范围比本意宽（它匹配整个 9 月下旬）。被删的是本会话昨天与今天留下的 run 目录——
`admin_data/` 与 `storage/` 都在 `.gitignore` 里，开发库里最早的行是 09-26，没有用户历史，
165 条行记录指向已消失的文件也正好是 `410 Gone` 那条路径本来要处理的情况——但这不该用通配写。
教训：**删 scratch 也要按完整路径逐个删，不写 glob**；`git status` 干净不代表通配删除安全，
被 gitignore 的目录正是通配最容易伤到的地方。

## 二十、第十九批补齐：拿报告逐条重扫，专找"声明了但没人用"（plan §2.10、§2.5、§2.6、§3.3、§2.11）

基线见本节末尾的门禁数字。做法换了一次：把报告交给一个只读子代理，要求它**对每一条候选
缺口拿两条证据**——"生产里有人调用吗"+"删掉这行有测试会红吗"——它交回 6 条。逐条自己复核，
6 条全部成立（其中两条比它说的更严重），第 6 条我选择不做，理由写在下面。

| 项 | 报告出处 | 现象（复核后） | 处理 |
|---|---|---|---|
| 指标半边空转 | §2.10 | `MetricName` 声明 10 条，**只有 5 条有写入方**；`queue_depth` / `model_load_seconds` / `model_inference_seconds` / `gpu_memory_bytes` / `artifact_io_seconds` 从没被记过，因此 `/stats` 与 Prometheus 文本里根本不出现。唯一的"测试"是**字符串相等**——把所有写入点删光它照样绿 | 五条各接上真实调用点：load 只在 `model_manager`（一处一名一义）、inference 在 `BaseModelBackend.infer`（窄于 stage_duration，只量前向）、VRAM 由采样器的 peak_mb 换算、产物 I/O 记 publish/recover 两头、queue_depth 记"这次提交排进了多长的队"。加**静态门禁**：`MetricName` 里任何一条在 `src/` 别处找不到引用即红；再加行为测试（桩后端跑一次，断言序列出现且不含 load） |
| 进度事件是空头支票 | §2.5 点 4 | `EventType.TASK_PROGRESS` 有枚举、无发送方；阶段内的进度只写 DB 行。单阶段计划（整条 vendored 链都在 `global_restore` 里）的 SSE 客户端在阶段结束前什么都看不到 | `_on_stage_progress` 按 5 个百分点节流发 `task.progress`，`data` 带 `progress/stage/detail`；e2e 断言事件存在且百分比映射正确（四阶段计划的 50 % → 12 %） |
| 重试留下描述已消失字节的行 | §2.6 | run 目录按 task id 命名，重试**原地覆盖** `final.png`，而 `add_artifact` 每次 INSERT 一行。`GET /tasks/{id}/artifacts` 于是有两行 `output`：**老行仍带着第一次尝试的 sha256/尺寸**，指向的内容已经不是它——客户端按它校验刚下载的图必然对不上 | `add_artifact` 改成"同 kind 先删后插"。两条测试：覆盖后只剩一行且 sha 与磁盘当前字节一致；替换只按 kind 生效，submit 时登记的 `input` 不会被顺手清掉 |
| 生产启动自己铸密钥 | §3.3 | 四条 fail-fast 只做了三条的一半；`API secret 缺失` 时它**生成一个 token 写进 `admin_data/api_token.txt` 并打印**——服务"健康"启动，用的却不是运维配置的密钥，第二个副本会铸出**不同的**密钥。报告原文允许自动生成，但只限本地 demo | 生产 profile 缺 token 直接拒绝启动（运维放好的 token 文件算已配置）；本地保持原行为。四条里的另两条**明确不做**并写进文档：`DATABASE_URL 缺失` 在本项目是合法单机配置（SQLite 是随货生产拓扑），"库落后于代码"由 `make db-check` 表达；"S3 credential 无效"要联网探测，且 boto3 默认链走实例角色时本就没有环境密钥——那属于 `make smoke-services` |
| 面板上写着 PSNR 却没有 PSNR | §2.11 / §3.9.1 | 输出框标签是 "Difference metrics vs. degraded original (PSNR/SSIM/MAE)"，而默认排队路径往里塞的是进度条 + 阶段清单；指标文本只在**同步 fallback** 里出现。`get_task` 其实已经返回 `evaluation_text` | 完成帧改成 `evaluation_text` + 阶段清单 + planner 决策；决策读取从 API 路由里**搬到 `task_service.planner_decisions()`**，两处共用一个"决策写在哪儿"的定义（否则又会长出第二批不一致）。UI 测试同时钉住 PSNR、决策行与上一批的 `skipped` 图标 |
| golden 指纹缺 face count | §3.10.2 | 指纹字段里没有人脸数，且两份基线的输入本身不含人脸 | **不做**，并记下理由：在一个输入无脸的仓库里加一个恒为 0 的维度不产生信号，只会多一条"看着有覆盖"的字段；等有真脸基线（需要 dlib 那台机器）再加才有意义 |

**顺带修掉的一处我自己刚写的错**：加 `_PROGRESS_EVENT_STEP` 时先在 `_METRIC_MAXIMA` 上方插了一次、
又在下方留了一份定义——正是这两批一直在抓的"同一个名字两处定义"。是复核 `grep -c` 时发现的，
删掉第二份。教训：**新常量的第二处出现应当是 review 的红旗，而不是"再补一句注释"**。

## 二十一、第二十批补齐：把"声明了但没人产生"做成一类门禁（plan §2.5、§2.6、§2.10）

上一批我给 `MetricName` 手写了一条静态门禁，这一批把它推广到整个 domain 枚举层——同一类缺陷
不该每出现一次就重新发现一次。扫出来 3 个，顺带挖出队列层两个互相掩盖的缺陷。

**静态扫描结果（`src/` 全域，除 `enums.py` 外无任何引用的成员）**

| 成员 | 性质 | 处理 |
|---|---|---|
| `EventType.TASK_STARTED` | `docs/api.md` 的事件表里**写着这个事件**，枚举里也有，但从没人发。排队路径的事件流是 `task.enqueued` → 直接 `stage.started` | 在两个"任务真的开始跑"的位置各补一次发送：worker 领取成功后、同步路径 `start_task()` 后；e2e 把 `task.started` 钉进事件清单 |
| `ModelStatus.LOADING` / `UNLOADED` | 不可达：模型在装载完成前根本不进注册表，卸载后报的是 `registered`（`loaded` 字段区分两者）。API 发出去的词表里没有它们，客户端不可能写成处理分支 | 删。留着只会让人以为 `/models` 有五种状态 |

门禁写法有意宽松：`Class.MEMBER` 或它的字面值（`"skipped"` 这种，仓储层就是这么写的）都算用到。
它抓的是"接到任何东西上都没有"的词表，不是引用风格。

**队列层：`QueueBackend.ack()` 有实现、无调用方。** 于是每完成一个任务，它的 stream 条目一直
挂在 PEL 里，直到租约过期被 `XAUTOCLAIM` 重投给某个 worker，而那时 DB 已无可领取的行——白跑一次；
更实在的是 backend 内部那张 `task_id -> record_id` 映射**每完成一个任务长一条，进程不死就不回落**。
测试文件开头写着"手写 fake 曾抓到 a no-op `ack`"——当年修的是实现，调用方从来没补上。

**两个缺陷互相掩盖**：手工重试走 `requeue_for_manual_retry()`，绕开传输层、不重新投递（自动重试
走 `queue.retry()`，它是会重投的）。所以 Redis 上"手工重试"今天之所以偶尔有用，靠的正好是
"没人 ack、旧条目还在 PEL 里"。只补 ack 会把它变成**永远不被领取**。因此两边一起改：
`TaskService.retry_task` 成功后 `_notify_queue()`（与提交同一条规则），worker 只在**终态**ack
（跑成功、或次数耗尽），被重新排度的任务**不** ack——它的条目还要用来唤醒下一轮。

**测试里被真实行为打断的三处**（都不是被测代码的错，记下来防重犯）：

1. `from fiximg.inference import backends.base as m` 是语法错（`from` 后面不能带点号路径），编译期就拦下了；
2. worker 里 `task_repo` 是**函数内局部导入**，我当模块名用 → `NameError`。它让测试表现为
   **超时**而不是报错：任务停在 `running`，得去翻捕获日志才看到真因；
3. 断言写成"重试过的任务没被 ack"，但循环跑得很快，等我断言时它已经把 `max_attempts` 用尽并
   **正确地**ack 了。改成在 `retry()` 调用内部对 `acked` 取快照——钉决策点，而不是钉终态。

**顺着第 2 条挖出的一个真缺陷**：`_loop` 对 `self._execute(task)` **没有保护**，而 `_execute` 的
`try` 只包住流水线。领取之后、进流水线之前任何一处抛异常（事件写、队列等待统计、哪怕一个写错的
变量名），异常会穿出线程 target：**进程还活着，worker 不再轮询，任务停在 `running` 直到租约过期**。
运维看到的是"队列不消化"，不是"worker 崩了"。现在循环兜住每个任务：立刻按 `worker_error` 记失败、
走同样的重试判定，然后继续服务。变异验证就是把那层 try 去掉——测试从"good-1 跑到了"变成
"一个坏任务之后什么都没再执行"。

**§2.6 要求"说清是至少一次还是至多一次"**：全库此前 0 次提及。补进 `docs/architecture.md` 的
Delivery semantics：DB 行是权威、stream 只是唤醒；终态才 ack；run 目录按 task id 键控所以重执行
**覆盖**上一次字节、每 kind 一行；副作用按 `task_id` 幂等；不提供顺序与精确一次。

## 二十二、第二十一批：对照附录 B 逐条重验收（37 项）

做法：三个只读代理各带一块验收面（架构 6 + API 5 / 可靠性 6 + 工程化 8 / 性能 6 + UI 6），
每条要求两条证据——**生产调用点** + **删掉这行会红的测试**。收回来 11 条缺口，逐条自己复核，
全部成立；本轮关掉 6 条，其余如实列在下面。

基线：**`1046 passed, 2 skipped, 24 deselected`**（CPU）· ruff 全绿 · mypy 0（118 文件）·
gpu 标记本机 19 例 · 迁移往返 + `db-check` · 两条基准都跑过（见下）。

### 关掉的那几条

| 验收项 | 复核到的事实 | 处理 |
|---|---|---|
| 可靠性 1「状态机完整」 | `finish_task` / `fail_task` 都是 `WHERE id=?`，**没有任何栅栏**：租约过期被 w-2 接管后，慢掉的 w-1 回来照样能把 w-2 的行写成 `completed`（带 w-1 的结果路径）；被用户 cancel 的任务也能被后置的 worker 改写 | 加 `AND status='running'`，排队路径再加 `AND worker_id=?`，两个函数返回 bool。栅栏的两半分别被两条测试钉住（去掉 worker 栅栏 → "the stale write landed"；去掉 status 栅栏 → 取消用例红）。顺带发现**一批测试直接对 queued 行写终态**，那是生产到不了的形状，10 处改成先 `start_task` |
| 可靠性 3「retry」 | `FIXIMG_TASK_RETRY_BACKOFF` 由仓储层读取，但 worker 调 `queue.retry(task.id)` 时**一个参数都不传**，而传输层签名默认 `delay_seconds=0.0` —— 自动重试路径完全没有退避，热失败任务是立刻重领 | 传配置值；测试钉住传给传输层的实参（12.5 s）以及 `retry_at` 真的落到行上 |
| 可靠性 5/6「crash 恢复 / 过期回收」 | `_scan_stale()`（过期租约重排 + TTL 回收）**只写在空闲分支里**：任务不断的 worker 永远不扫盘 —— 恰好在"最快涨满磁盘"的那种部署上，TTL 又是一个没人读的数字 | 移到每轮都跑（内部本就有 60 s 时间闸）；测试用一个永远有活的假传输层证明"忙的时候也在扫" |
| 可靠性 2「lease」 | `_Heartbeat` 只有"自己调 `queue.heartbeat()`"的测试，**把心跳线程整个删掉，套件照绿** | 换成驱动真实循环的测试（编排器睡过心跳周期），断言运行中的任务确实刷新了租约 |
| 工程化 14「benchmark regression」 | `benchmark/throughput.py` **永远 return 0**；更糟的是它的 drain 行写的是 `"tasks": done`，也就是说"25 个任务只消化了 3 个"会打印成"3 个任务的速率"，缺口在数字里被抹平 | 行改成 `tasks`=提交数、新增 `executed`=完成数并打印；没消化完就 exit 1。两个方向都验：`--timeout 0.05` → exit 1，正常跑 → "drain gate: every submitted task was executed" |
| 工程化 12「CI」 | 镜像扫描那步写着 `exit-code: "0"`（trivy 文档里就是"永远成功退出"），而 CI 自检的正则只看 `run:` 脚本，**对 action 参数是瞎的** | 改成 `"1"` + 提交 `.trivyignore`（空文件，注释写明：要豁免就在里面写 CVE 和理由，不许回来改这个开关）；自检新增两条：action 参数不许把失败中和掉、引用的 ignore 文件必须在仓库里 |

API 侧还关掉一条：`test_task_is_attributed_to_the_api_principal` 名为验"任务归属认证主体"，
实际只断言 `settings.api_token_user == "api"`（一个配置默认值）——把路由里 `principal.user_id`
改回硬编码 "api"，它**仍然绿**。改成把主体名改掉再断言行上的 `user_id` 跟着变，然后确认
"改回硬编码"这个变异会让它红。

### 连续两次假验证（我自己的错，记下来）

心跳那条测试的变异验证，我连着"验"了两次都是假的：

1. 第一次 `-k heartbeat` 选中 **0 条**（改名后测试名里已经没有 heartbeat 了），pytest 什么都没跑，
   `tail -1` 是空 → 我读成"没输出＝没问题"；
2. 第二次更糟：上一条命令末尾已经把备份还原了，我在**干净代码**上跑变异检查，自然"通过"。

真正的做法是每条命令内部自带状态断言：变异脚本先 `assert s2 != s`，跑测试前 `grep -c` 证明调用点
确实为 0，跑完还原后再 `grep -c` 证明回到 1 —— 这次才看到 **mutant 红 / clean 绿**，中间值都是自欺。
这和前面几轮反复碰上的"skip 不是失败、空输出不是通过"是同一件事的另一面：**过滤条件命中 0 条、
被 deselect 掉的用例、还原后的重跑，全都不表现为失败。**

### 仍未完成（本轮复核确认，非新问题）

1. **§4.2 Step 4 / 验收 6 的另一半**：只有 `output` 走产物存储。输入图与阶段中间结果仍由
   `create_run_dir` / `save_input_image` 写本地路径、从不发布，所以远程存储下**换一台机器的 worker
   领到任务会在读输入时失败**。这是跨节点拓扑的真缺口，不是环境限制。
2. **§3.8 / P0 Step 3 的尾巴**：`/api/v1/stats` 与四个 health 端点仍返回手工 dict（无
   `response_model`），OpenAPI 里是自由对象。
3. **§3.5.3 warmup**：`models/manifest.yaml` 每个模型的 `warmup:` 键**没人读**；
   `ModelManager.load_all()` 无调用方（worker 启动预热没接）；测试传的 warmer 都是自己造的。
4. **§3.5.1 统一接口的实际形状**：阶段真正调用的是 `run_folder()` / `produced_dir()` /
   `pipeline.process()`，这些**都不在 `ModelBackend` 协议上**，协议也没有 `runtime_checkable`
   一致性检查——一个新后端不满足文件夹桥接契约，没有测试能发现。
5. **§3.9.1 UI**：`POST /tasks/{id}/retry` 在 UI 里**没有入口**（历史面板的 Re-run 是从输入图重排
   一个新任务，不重试语义）；失败结果旁边没有重试按钮；工作标签页上没有显式下载控件
   （下载目前落在历史面板选中行，本轮已补）。
6. **§3.10.3 基线数据**：p50/p95 只有散文里的数字，没有提交的机器可读基线文件；
   `benchmark.latency --real` 无条件 return 0（真模型延迟不在门禁内，属环境受限那一半）。
7. **#17 §3.5.4 batch / pinned memory / CUDA stream**：仍未做，理由不变。
8. 环境受限项：真实 GPU/CUDA、真实 PostgreSQL/Redis/MinIO、能编译 dlib 的机器。

## 二十三、第二十二批补齐：A-1 与 A-3（plan §2.8 / §4.2 Step 4、§3.5.3）

基线：**`1051 passed, 2 skipped, 24 deselected`** · ruff 全绿 · mypy 0（118 文件）· gpu 标记
本机 19 例 · 迁移往返 + `db-check` · 两条基准门禁。

上一节列出 6 条"需要写码"的缺口，这一批关掉其中两条，都属于"接口在、没人走"，但各带一个额外形状。

### A-1：只有产物会跨节点，输入不会

`ArtifactStore` 半边接通之后（第十七批）我以为 §2.8 完事了。实际是：**发布的是 `output`，
读取路径依赖的却是 `input`**。API 节点收下上传、写进自己的 run 目录；远程存储下另一台 worker 领到
任务，在阶段 1 打开输入文件时失败——错误长得像"文件丢了"，而不是"这个拓扑没通"。验收标准
"图片与中间结果迁成 object key"因此只满足一半。

- 键规则从 `_publish_output` 提出为 `object_key_for()`，发布/回捞收进 `publish_local_file()` /
  `fetch_to_local()`：结果与输入用**同一条**规则，否则两个节点对同一个字符串各自解释。
- 提交时发布输入并记 `artifacts.uri`（`op="upload"` 一并进 `artifact_io_seconds`）；
  `execute_queued` 打开文件前先 `_recover_input()`；本地存储不记键，所以缺文件仍是明确的
  `Input image missing`，不会被静默变成"再试一次"。
- 回收自动覆盖：`purge_expired_published()` 本来就按"有 uri 的行"清对象，输入对象从此也在里面。

四条测试；三条变异（去掉提交时发布 / 让 `_recover_input` 声称成功却不真取 / 让 `execute_queued`
退回 `os.path.exists`）各自对应红。

### A-3：`warmup` 是文档里的三档、代码里的一档、进程上的零档

`models/manifest.yaml` 表头写着"`warmup` 覆盖全局 `FIXIMG_WARMUP`"，五个模型**每个都写了**这个键，
而代码没有任何地方读它（它掉进 `ModelVersion.metadata` 之后再没人碰）。同时：

- 启动预热只存在于 `create_app()`，也就是**唯一不执行推理的进程**（纯 API 节点）负责抱权重，而真正
  执行的独立 worker 每次冷启动，第一个请求替所有人付权重加载 + 首次前向的代价；
- `ModelManager.load_all()` 的 docstring 自称用于"worker warmup"，调用方数量为零。

改法是把判断收敛到一个解析器：`warmup_strategy(name)` = manifest 覆盖 → 否则全局，认不出的值退回
全局而不是猜；`should_warm_on_load()` 让 `lazy` 真的不跑那次 dummy（`first-use` 跑）；
`startup_models()` 决定谁在启动预热；`warm_at_process_start()` 由 `PipelineWorker.start()` 与 API
启动**共用**（内联 worker 部署本就同一进程，第二次加载是 no-op）。`load_all()` 删掉——它想做的事
现在有一个有人调的版本。

11 条测试（`tests/unit/test_model_warmup.py`）+ 三条变异：warm 恒定传入 / 启动选择忽略每模型键 /
从 `worker.start()` 删掉那次调用。

### 这批的方法收获：**第一次全绿不证明测试在跑**

新写的 11 条一次通过，我第一反应是"不可能这么顺"。两个坑叠在一起：

- 我用 `pytest -q | grep -aoE '[0-9]+ (passed|failed)'` 提取结果，三次拿到**空字符串**——空输出既可能
  是"没跑"，也可能是提取式不匹配。改成看**退出码**之后，三条变异如实返回 1。
- 上一批已经因为 `-k` 命中 0 条而把"没有输出"读成"通过"一次，同一个坑换了个入口又踩了一遍。

规则化：**变异验证只认退出码；任何"绿"都要能说出它跑了几条用例。**

## 二十四、第二十三批补齐：把"契约"这个词做实（plan §2.5 P0 Step 3、§3.5.1）

基线：**`1062 passed, 2 skipped, 24 deselected`** · ruff 全绿 · mypy 0（119 文件）· gpu 标记本机
19 例 · 迁移往返 + `db-check` · 两条基准门禁。

剩下的四条"要写码"缺口关掉两条。它们的共同点不是"缺功能"，而是**声明与实跑各说各话**。

### 1. `/stats` 与 health 探针是最后两个手工 dict

P0 Step 3 说"手工 dict 返回全部收敛到 schemas"。任务/模型/用户端点早就做了，**偏偏运维真正会去
轮询的那两个没有**：OpenAPI 里它们是自由对象，生成的客户端只能拿到 `dict[str, Any]`。

补的时候有意思的地方在于，光加 `response_model` 不够，甚至可能有害——FastAPI 会**静默丢掉**模型里
没写的键。于是两头都钉：

- `StatsResponse` / `ReadinessResponse` 等按**实测形状**建模（先探真值再写模型：readiness 里的
  `models` 是 `ModelManager.health()` 的 `{cuda_available, loaded_models, lazy_models, versions}`，
  跟 `/stats` 里的 `models`（只有 `load_times_ms`）不是一回事——凭想象写就会写错）；
- 测试两头比对：payload 有而模型没有 → 红（那正是会被静默丢掉的情况）；模型有而 payload 没有 → 红；
- 再加一条**全量门禁**：遍历 `/openapi.json`，任何 2xx/4xx/5xx 的 `application/json` 响应若没有
  schema 引用即红。

这条门禁当场又抓出三个**文档说谎**的端点：`/stats/metrics`（其实 text/plain）、
`/tasks/{id}/events`（SSE）、`/tasks/{id}/result`（PNG）——三者都返回裸 Response，FastAPI 就替它们
默认写了 `application/json`。用 `response_class=` + 显式 `content` 补上媒体类型。门禁自己也被误判
教育过一次：`/api/v1/users` 返回 `list[UserView]`，schema 是 `{type: array, items: {$ref}}`，
只认顶层 `$ref` 的检查把它当成未声明——数组要往下看一层。

变异验证两条：摘掉 `response_model` → 门禁红；给 payload 加一个模型没声明的键 → 比对红。

### 2. 阶段调用的接口，不在"统一接口"上

§3.5.1 的 `ModelBackend` 是 load/infer/unload/health 那一套；但拆分后的 legacy 链（§3.6/§3.7）
真正依赖的是**文件夹桥接**：把输入目录交给后端、再从共享根目录读回上一阶段的产物。这一半此前只
存在于鸭子类型里——`run_folder` / `produced_dir` 不在任何协议上，也没有 `runtime_checkable` 一致性
检查，一个注册到 folder 阶段却缺方法的后端只会在**请求进来时**以 AttributeError 现形。

做法是把契约拆开写，而不是硬塞进一个协议：新增 `FolderBridgeBackend`（同样 `runtime_checkable`），
并**显式保留它的可选性**——DDColor 没有文件夹桥接，"人人都要满足但其实三分之二不满足"的协议不是
契约而是谎。于是：

- 被测阶段清单**从源码扫出来**（哪些 stage 模块里出现 `.run_folder(`），不是手维护的名单——一份
  会过期的名单本身就是同一类漂移；另有一条测试专门钉这份清单能扫到那四个阶段；
- 每个 folder 阶段对应的注册后端必须同时满足两个协议，且 `implementation ∈ {native, legacy-cli}`；
- 一条反向测试钉住 DDColor **不是** folder 桥接（协议退化成"人人都满足"时它会红）。

变异验证：把 `LegacyCliBackend.produced_dir` 改名 → 参数化用例立刻在 `face_detection` 上红。

### 现在还剩什么

写码类只剩 **UI 的 §3.8 重试入口**（历史面板 Re-run 是重排新任务，不重用尝试预算；失败结果旁无
重试按钮；提交标签页无显式下载控件）与**可提交的 p50/p95 基线数据**（`latency --real` 仍无门禁值，
那一半属环境受限）。其余是刻意不做（#17、golden face count）与环境受限（真 CUDA / PG / Redis /
MinIO / 能编译 dlib 的机器）。

## 二十五、第二十四批补齐：UI 的最后一格与基线数据的最后一格（plan §3.8、§3.9.1、§3.10.3）

### 1. 重试：端点有、UI 没有，而"Re-run"悄悄不等于重试

历史面板一直有个 Re-run selected，但它是**从存过的输入图重新提交一个新任务**——新 id、新尝试预算。
§3.8 的 `POST /tasks/{id}/retry`（同 id 重排、`attempt_count` 归零、产物与指标仍描述同一件工作）
在面板里**没有入口**。于是验收项"retry"看起来满足，实际用户能点到的那个是"复制一份"。

补一个 `retry_history_entry` + 独立按钮，并把它与 Re-run 的差别写进注释。四种结果各有测试：
同 id 重排成功、非 failed/cancelled 行**不调用服务**（否则一次误点就多跑一次 GPU）、服务抛异常
时回答在 caption 里而不是抛给 Gradio、`retry_task` 返回 False 时说明原因、没选中行时提示。

### 2. 下载：`gr.Image` 在 Gradio 6 没有另存出口

五个提交标签页的结果只有预览图。先确认组件本身没有 `show_download_button` 这类参数（有就不用
加），确认没有之后在 `result_column` 里加 `gr.DownloadButton`，于是**每个 handler 多一个输出**：
进度帧用 `gr.update()` 不碰它、完成帧给出 `value=result_path, visible=True`、取消帧收回可见性、
同步 fallback 同样带上。契约测试这次真的在管事：改完先红两处（submit 依赖 3→4 输出、clear 依赖
4→5），改到位后才绿。另加一条结构断言：五个 handler 的输出里必须各有一个 downloadbutton——新加的
标签页忘了放，它会红。

### 3. §3.10.3 的"p50/p95 基线"从散文变成数据

新增 `benchmark/baseline.json`：四档尺寸的 cold/p50/p95/overhead、测于哪台机器、何时测、为什么
只有 `small` 受门禁。`latency` 每次跑都打印对照（`small: p50 0.0513s vs recorded 0.0514s ->
x1.00`），测试**双向**钉住文件与代码常量一致（`max_overhead_seconds` == `DEFAULT_MAX_OVERHEAD_SECONDS`、
尺寸集合与 `INPUT_SIZES` 相同、被门禁那一档必须有数）。`--real` 那条改成明说的 report-only：
真模型延迟没有可提交的 CPU 基线，硬编一个阈值要么永不触发要么每次运行都响，真正的回归网是
`tests/gpu` 的 golden 比对。

写文件时被抓两次：第一版只写了三档尺寸、还按记忆把 `large` 的像素写成 4096（实际 `INPUT_SIZES`
里 `large=2048, xlarge=4096`）——一致性测试立刻红。**"凭记忆写基线"和"凭记忆写 schema"是同一件事**，
所以两处都是先测再写。另修一个默认参数陷阱：`def load_baseline(path=BASELINE_PATH)` 在 def 时就
绑定了路径，monkeypatch 模块常量无效；改成调用时解析。

变异验证：把 `download_1` 从某个标签页的 outputs 里摘掉 → 结构与 arity 测试红；完成帧不给文件
→ `KeyError: 'value'`；把基线文件的天花板改成 5.0 并删一档 → 一致性测试红。

### 验收清单现在的状态

37 项里，代码侧只剩**环境受限**与**刻意不做**两类；本轮之后不再有任何"接口在、没人调"或"标签与
实测不一致"的已知项。

## 二十六、第二十五批补齐：把整套门禁搬到真实部署环境里跑一遍（plan §3.5.4、§2.7、§3.12）

前 24 批全部在 CPU-only 的 Python 3.14 解释器上验证。第一次在项目**真正部署的那个环境**
（conda `fixoldimg-gpu`：3.11.15 / torch 2.7.1+cu128 / dlib 20.0.1 / cv2 5.0.0.93 /
gradio 6.22 / SQLAlchemy+psycopg+redis+boto3）里跑同一套门禁，出现 5 个红色，全部已修：

| 发现 | 为什么 3.14 上看不见 | 处理 |
|---|---|---|
| `device_index("none")` 返回 `0`，权重被搬上 GPU | `none` 被并进了 `auto` 分支；CPU 机器上两者都答 `-1`，缺陷不可观测而非未被发现 | `src/fiximg/inference/backends/__init__.py`：`cpu`/`none`/`-1` 一律 `-1`，只有 `auto` 才去问 CUDA |
| MemorySampler 契约测试断言 `peak_mb is None` | 只有 CUDA 机器才会返回 float | 测试按 `torch.cuda.is_available()` 分两路，两路各自断言类型 |
| `ModelStats` 少一个 `gpu_memory_peak_mb` 字段 | 该 key 只在有 GPU 时出现 | schema 补字段（顺带证明"先测再写 schema"这条规矩的价值） |
| created_at 唯一性断言 | Windows 3.11 的 `datetime.now()` 步进 ~0.5 ms，四次紧凑插入**同一个**时间戳；3.14 的时钟更细，于是看到 4 个不同值 | 只断言"格式规范 + 顺序确定"，并且用**冻结时钟**主动造同刻；去掉 `ORDER BY created_at DESC, id DESC` 的 `id DESC` 后，3.11 与 3.14 都变红（不再依赖某台机器的时钟精度） |
| 锁定审计报 `out of date: gpu, otel, postgres` | 答案取决于哪个解释器最后跑过生成器 | 见下 |

### §3.12 锁定审计的三个真问题

1. **`requirements/gpu.txt` 装不上**：文档里"安装 GPU 栈"就是这一条命令，而它把
   torch / torchvision / dlib 钉成 `file:///D:/Soft/fixoldimg_wheels/torch-2.7.1+cu128-...whl`
   和 `file:///C:/bld/dlib-split_.../work` —— 只存在于生成它的那台机器的路径。
   根因是 `pip freeze` 对本地 wheel / conda build 就输出 direct-url，而分组生成器
   把锁里的整行原样抄走。
2. **门禁自己造输入**：`test_export_reports_groups` 用无参方式调生成器，等于每跑一次
   测试就把仓库里的 `requirements/*.txt` 重写一遍；`requirements/` 当时甚至还没进版本库，
   谁最后跑测试谁定义"当前"。
3. **解释器耦合**：逐字节闭包对比依赖已安装元数据（CPU torch 与 cu128 torch 的依赖集
   不同），并且 gradio 的 `audioop-lts; python_version >= "3.13"`、
   `importlib-metadata; python_version < "3.8"` 这类**别的 Python 才需要**的条件依赖被无条件
   收进 runtime，生成出的文件在当前解释器上根本装不出来。

处理方式：`scripts/export_locks.py`

* 新增 `--freeze`：从当前环境写 `requirements.lock`，direct-url 规范化为 `name==version`
  并把原始来源留成注释（既可安装又可追溯），同时记录 `# Frozen-environment: python 3.11 / win32`；
  `make lock-freeze` 不再走 `pip freeze`。
* 新增 `--output` / `FIXIMG_LOCK_OUTPUT_DIR`：测试一律写临时目录，仓库不再是生成器的工作目录。
* 闭包**按 marker 求值**（`packaging.markers`），别的 Python 才用的条件依赖不再进分组文件。
* `check()` 分两层：任何机器都跑的红（锁可移植、逐条 pin 与锁一致、每行都是 `name==version`），
  以及**只在锁记录的那个环境**跑的闭包重导；换台机器就把两个环境名都打出来作为跳过理由，
  而不是把差异说成"锁过期"。

测量：runtime 61→58、gpu 39→36、redis 2→1 个包（marker 修正）；`requirements.lock`
现在 115 个包、0 条本机路径。变异证明（每项都先红、复原后绿）：改一个 pin → 红；
把 `torch @ file:///...` 放回锁里 → 红（门禁 + 两个测试）；删环境守卫 → 红；
让 `--output` 失效 → 红；`strict = True` → 红；去掉 `id DESC` → 两个解释器都红。

顺带修掉一处声明互相矛盾：`pyproject.toml` 依赖 `opencv-python-headless`，而
`requirements.txt` 钉的是 `opencv-python`（同一份依赖的两个"权威来源"给了两个 cv2 提供者）。
代码里没有任何 GUI 调用（全仓 `cv2.imshow` 零命中），所以统一到 headless；
本机这个 conda 环境装的仍是 `opencv-python`，因此 `runtime` 组在本机的闭包重导会如实
打印"root not installed"——这是环境不匹配，不是代码缺陷。

### 一条新的静态门禁：目标解释器的语法

`f"numpy=={importlib.metadata.version("numpy")}"` 是 PEP 701（Python 3.12+）语法，
它在 3.14 上写出来、在 3.11 上收集期直接 `SyntaxError`。ruff 本机实测能报（`target-version =
"py311"`），但这次是**新测试文件在被执行之前没先过 ruff**——顺序问题就用门禁保证：
CI 的 3.11 job 增加 `python -m compileall -q $(LINT_PATHS)`，本地 `make syntax`，
并由 `tests/unit/test_ci_gates.py::test_the_target_interpreters_grammar_is_checked_by_a_gate`
断言该步骤存在、路径集合与 `make lint` 一致、且所在 job 钉的是 3.11
（把该步骤从 ci.yml 删掉 → 测试红）。实测：同一个探针文件 3.14 `compileall` 退出 0、
3.11 退出 1；全仓在 3.11 下 `compileall` 退出 0。



## 二十七、第二十六批补齐：在真实 GPU 与真实 PostgreSQL 上跑（plan §3.5.1、§3.5.4、§4.3 Step 2、§3.10.1）

前两批把门禁搬进了部署解释器，这一批把**依赖**也搬进去了：真 CUDA（RTX 5060）与
真 PostgreSQL 18.6（conda `pgserver` 的 initdb/pg_ctl，127.0.0.1:55432，仓库内数据目录）。

### GPU 层（`pytest tests/gpu -m gpu`，fixoldimg-gpu）

第一次跑 24 项：**19 通过 / 5 失败 / 0 跳过**，五个失败全是 scratch 原生分支：

```
RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor)
              should be the same
```

根因在 vendored 侧：`Global/detection_models/networks.py:85` 在 `UNet.__init__` 里做
`self = DataParallelWithCallback(self)`（`sync_bn=True`），这个包装会把刚建好的权重搬到
GPU。于是 `load("-1")`（CPU worker）得到 `opt.gpu_ids=[]`（CPU 输入）却拿着一个
`cuda:0` 的分割网络。CPU-only 机器上两边都在 CPU，缺陷不可见。
修法是在构造**之后**显式落位：`model.to("cpu" if index < 0 else index)`。
新增 `tests/gpu/test_scratch_device_placement.py`（3 项：CPU 请求不得占 VRAM 且 mask 仍跑通；
GPU 请求确在 cuda:0；`auto` 与主模型同设备）。把修复退回原写法 → CPU 项变红。
修复后（加上三门新的落位测试）`pytest tests/gpu -m gpu` = **27 passed, 0 failed,
0 skipped**，全套 `-m "not gpu"` = **1079 passed, 19 skipped, 27 deselected**
（退出码判定；19 项跳过 = 8 项 PG 服务器层未给 URL + 8 项缺 `moto` + 2 项"本机有可用
GPU 所以只测报错路径" + 1 项 Windows 不强制 POSIX 文件位）。

同一批里 `tests/gpu/test_face_detect_equivalence.py` 在 dlib 20.0.1 +
`shape_predictor_68_face_landmarks.dat` 上**通过**（不是跳过：`-rs` 无 SKIPPED 记录），
即 §4.2 Step 2 一直等的"那台机器"就是本机。据此把
`FIXIMG_FACE_DETECT_NATIVE` 默认从 `false` 改成 `true`（config、registry docstring、
README 中英、deployment 三处同步），并把测试改成读**出厂默认**：
`monkeypatch.delenv` 后 `Settings().face_detect_native is True`，
再断言 `=0` 仍回退适配器（缺 dlib/模型时 `native_available()` 自己兜底）。

### PostgreSQL 层：第一次连真服务器

四个缺陷全部只在真服务器上出现——fakedriver 按代码问什么就答什么，结构上看不见
"服务器拒绝什么"：

| 缺陷 | 服务器报的错 | 修法 |
|---|---|---|
| 迁移 0001 原样执行 SQLite DDL | `syntax error at or near "AUTOINCREMENT"` —— 全新 PG 库根本没有表 | 0001 逐条走 `dialect.adapt_ddl`（与 `Connection.executescript` 同一规则） |
| `history.user` 是 PG 保留字 | `syntax error at or near "user"` | DDL 与两条 INSERT 里写成 `"user"`（SQLite 同样接受） |
| `SELECT COUNT(*)` 用 `row[0]` 取 | `KeyError`（psycopg `dict_row` 没有位置访问） | 语句加 `AS total` 按列名读；**同时删掉 fake Row 的位置访问**，让替身不比真驱动宽容 |
| `COALESCE(metric_value, metric_text)`；子查询用了 SQLite 变参 `json_object` | `DatatypeMismatch` / `UndefinedFunction: json_object(unknown, text, ...)` | 方言新增 `row_object_fn`（PG `json_build_object`）与 `json_metric_value()`（PG 用 `to_json` 分支保数值类型）；`get_metrics` 改在 Python 侧合并 |
| `downgrade base` 在 PG 上失败 | `HINT: You will need to rewrite or cast the expression.` | 0003 回滚先把 `metric_value` 放宽成 TEXT 再搬值（SQLite 两种顺序都过，所以只有服务器能发现） |

验收方式（都是先红后绿，退出码判定）：全新库 `upgrade` → `status` →
`downgrade base` → `upgrade` → `check` 全 0；`tests/integration/test_postgres_server.py`
8 项在真服务器上全过（含 4 worker × 20 任务的 `SKIP LOCKED` 无重复领取、worker 围栏、
SQL 侧 p50/p95 与独立最近秩实现相等）；把 `{row_object}` 退回字面量 → 3 项红；
把 `"user"` 退回裸词并换新库 → `upgrade` 红。
CI 新增 `postgres` job（`postgres:16` service container + 上述往返 + 该层），
并由 `test_ci_gates.py::test_the_postgres_tier_runs_against_a_real_server` 断言这个 job
**必须**带数据库服务、必须含 `downgrade base`、必须含 `runner check`、不得吞失败
（分别删掉任一项 → 该测试红）。

### 顺带：一个能让整套测试挂死的真缺陷

`psycopg.connect` 不带 `connect_timeout` 时，对着**拒绝**的端口也永不返回
（本机实测：裸 socket 2.0s 拿到 `ConnectionRefusedError`，psycopg 默认 70s 未返回，
`connect_timeout=3` 3.01s 抛 `ConnectionTimeout`）。`test_smoke_services_fails_on_an_unreachable_postgres`
因此在 3.11 上把整套测试挂住，而 3.14 因未装 psycopg 走"缺驱动"快路径永远看不见。
修：`connection.DEFAULT_CONNECT_TIMEOUT = 10` 传入 psycopg，smoke 脚本用自己的 5s 预算，
smoke 测试改为**限时**并断言"在预算内给出答复"；`make lock-check` 与门禁本身不受影响。



## 二十八、第二十七批补齐：声明过的优化必须真生效；只会走运的清扫规则（plan §3.5.4、§2.6、§2.7）

### 一、manifest 的优化键只有一个人读（§3.5.4）

`models/manifest.yaml` 声明 `precision` / `channels_last` / `compile`，但真正调
`apply_model_optimisations()` 的只有 `ddcolor.py:67`；`global_restore.py`、
`face_enhance_native.py` 只用了 `inference_context()`。也就是说给这两个模型写
`channels_last: true`，什么都不会发生、日志里没有一行、`/api/v1/models` 还照样把
这个模型显示成"已配置"。这正是本项目反复踩的"接口在、没人调用"那一类，只不过这次
是"配置在、没人读"。

做法：

* `precision.apply_declared_policy(model, policy, device)` 返回
  `(model, applied)`——**逐个键**报告有没有落地；`apply_model_optimisations()`
  变成它的薄封装（签名不变，原有断言仍然成立）。
* `BaseModelBackend.apply_policy(model)`：任何持有 `policy` 的后端都要把加载好的网络
  过一遍；记录写进 `optimisations_applied`，`unload()` 时清空（换设备重载后不能引用
  上一次的答案）。一个后端多个网络时（scratch = 检测网 + mapping 网），**任一个没接受
  就记 False**，不能拿"大网络接受了"当结论。
* 三个原生后端的 `_do_load` 全部接上：`global_restore.py`（含 scratch 检测网）、
  `face_enhance_native.py`、`ddcolor.py`。
* 落地情况进 API：`ModelView.optimisations`（`ModelOptimisationsView`：声明值 +
  `applied`），`docs/api.md` 与 `docs/deployment.md` 各写一段。没有 torch 策略的后端
  （子进程适配器、dlib）为 `null`。
* `models/manifest.yaml` 里 global_restore 那句"Subprocess-backed: precision is chosen
  by the legacy script"已过期（质量链现在是常驻的），改成说明这行由谁读。

门禁三条，都能红：

| 门禁 | 变异验证 |
|---|---|
| `test_every_native_torch_backend_applies_its_declared_policy`（静态扫 `_do_load` 及其调用的私有辅助） | 把 `face_enhance_native` 的 `self.apply_policy(model)` 改回 `self._model = model` → 该参数项红 |
| `test_a_declaration_reaches_the_backend_that_serves_it`（读 manifest：谁声明了优化，服务它的后端就必须有策略） | 给 `face_detection` 加 `channels_last: true` → 红，报 `LegacyCliBackend has no policy` |
| `tests/gpu/test_ddcolor_gpu.py::test_the_declared_optimisations_really_are_applied`（真硬件：取一个 Conv2d 权重断言 channels-last 连续，并断言后端记录为 True） | manifest 里 `channels_last: false` → 红 |

顺带记录一个**做变异测试时自己犯的错**：第一次我把
`"    channels_last: true"` 当唯一串替换，结果命中的是第 19 行的**图例注释**，第 43
行的真声明没动，于是"变异后仍然通过"差点被当成"测试无效"。按行定位重做后，变异确实红。
教训：改文件要用行/结构定位，不能用可能出现在注释里的短字符串。

### 二、过期租约回退比较会把正在跑的任务抢走（§2.6 × §2.7）

最后一轮全量回归里 `test_reset_stale_running` 在**两个解释器上同时变红**——这不是环境
差异，而是这个用例一直在靠巧合通过。事实（实测）：

```
本机 UTC+8。legacy 写法存下 '2026-09-28 00:14:21'（localtime，空格分隔）
timestamps.now()  = '2026-09-27T18:14:21.419406Z'
cutoff(3600)      = '2026-09-27T17:14:21.419418Z'
```

`reset_stale_running()` 在没有 `lease_until` 时用 `started_at < cutoff` 兜底，而这个
比较是**字符串比较**。定宽 UTC 下它等价于时间比较；对 §2.7 之前的值是纯碰运气：
空格(0x20) < `T`(0x54)，于是"一秒前刚启动"的 legacy 行会被判成比 cutoff 更旧而被
重新入队——把任务从正在跑的 worker 手里抢走；反过来跨过午夜后同一个用例又判成更新，
于是永远不被清扫。原测试用 `datetime('now','localtime','-2 hours')` 造数据，只在
"本地日期 == UTC 日期"那几个小时里成立；今天本机过了零点，它才终于失败。

最终规则不是"拒绝比较"，而是**各按各的写法比**：

* `lease_until`（永远是 canonical）与 canonical `started_at` → 对 `timestamps.cutoff()`；
* pre-codec `started_at`（19 字符、本地钟面）→ 对新增的 `timestamps.legacy_cutoff()`，
  与 `timestamps.parse()` 对 legacy 值的既有假设一致（"拒绝比较"是中途方案，它会让
  2000 年那种老行永远收不回来，`test_reset_stale_falls_back_to_started_at_for_legacy_rows`
  当场把它顶红——那条测试的意图是对的）；
* 既无租约又无 `started_at` 的行才是"无法定时"：`running_rows_without_a_clock()`
  计数 + WARNING，不去猜。

新测试 `test_each_clock_is_aged_against_a_bound_of_its_own_shape` **注入同一个虚构时刻的
两种写法**（`2026-03-14T10:00:00.000000Z` 与 `2026-03-14 17:59:00`），因此不受宿主机时区
和当前时刻影响，四个方向全钉：一分钟前的 legacy 行不许动、两天前的 legacy 行必须回收、
过期租约必须回收、无时刻的行必须被计数。把两条分支并回"一个 bound 两种写法"→ 两个
解释器都红；只删 legacy 分支 → "两天前的行没被回收"红。原测试则改成写它本来就想表达的
`timestamps.deadline(-2*3600)`。

顺带用**真实开发库**验了一遍 §2.7 的自检命令（只读打开前后计数一致，确认它不改数据）：

```
applied: (none)      head: 0004      pending: 0004, 0003, 0002, 0001
legacy-format rows (run `fiximg-db upgrade`):
  metrics.created_at: 2081   system_events.created_at: 1867   task_stages.created_at: 675
  tasks.started_at: 223      tasks.created_at: 236      artifacts.created_at: 366 ...
exit=1
```

这台机器上的 `admin_data/fixoldimg.db` 有 236 条任务、时刻全是 pre-codec，且从没打过
alembic 修订——正是 `make db-check` 该在部署脚本里挡住的那类库。测试套件并不写这个文件
（跑完一轮后行数与 `max(created_at)` 不变，mtime 变化只是 WAL checkpoint）。

这一条是本轮唯一"测试绿着而生产行为错"的发现，也是 `verify-refactor-gates` 说的
"green is not evidence"的又一个实例：巧合通过的用例比红着的用例更贵。



## 二十九、第二十八批：把 V3 真的启动起来，读它报出来的每一个数（plan §2.5、§2.6、§2.7、§3.8、§7.1）

前二十七批都在"测试绿"的坐标系里工作。这一批换了个坐标系：在真实 Anaconda GPU 环境
（torch 2.7.1+cu128 / RTX 5060 / dlib 20.0.1）里用 `python -m fiximg.cli.api` 与
`python -m fiximg.cli.worker` 起两个进程，走 `POST /api/v1/tasks` → SSE
`/events` → `GET /result`，然后把响应里每个数字拿另一条路算一遍。测出来的东西和
读代码看到的确实不是一套。

| 项 | 现象（实测） | 处理 |
|---|---|---|
| **独立 worker 不读数据库 URL** | API 进程经 `create_app()` 调 `apply_database_url()`，`cli/worker.py` 只调 `init_db()`。设了 `FIXIMG_DATABASE_URL` 之后，worker 自己打印 `db=E:\...\admin_data\fixoldimg.db` —— 它开的是默认 SQLite，而 API 写的是另一个库。compose 与所有文档里的分体拓扑都会踩：任务永远 queued，两个进程都"健康" | `prepare_database()`：`apply_database_url()` → `init_db()` → `ensure_schema()`，并把 `(kind, target)` 打进就绪日志。用变异确认：删掉那一行，测试报 "the engine itself was never relocated" |
| GPU 峰值三处对不上 | 同一次运行里 stage 报 334.8 MB，`GET /stats` 报 146.3 MB。根因是 `MemorySampler` 每 stage `reset_peak_memory_stats()`，而 stats/model_manager 直接读那个"自上次 reset 以来"的计数器 | `gpu_memory` 维护一份不会被 reset 清掉的进程级 high-water（进 window 前、出 window 后各折一次）；`/stats` 再折入**其它进程**持久化下来的 stage 值（分体拓扑下 API 进程自己量到 0.0）。4 条测试 + 3 处变异各自变红 |
| `artifacts.model_version` 全是 NULL | 列存在、API 也透出，但没有任何写入者 | `_run_plan` 记录每个 stage 写了哪些 kind，写行时填 `<stage>@<version>`（取后端 load 后报告的版本，不是 manifest 声明值）。e2e 断言 + 变异（把参数改回 `None`）变红 |
| `stage_order` 在 HTTP 响应里消失 | 数据库有 0..3，聚合 SQL 的 `json_object` 列表没带上它，`StageView` 也就没有 | 聚合与视图都补上；e2e 断言 `[0,1,2,3]`。顺带发现 `list_tasks` 与 `get_task` 两份聚合必须同时改，否则分页视图又少一列 |
| `/artifacts` 把服务器绝对路径发到线上 | `uri` = `C:\Users\...\storage\tasks\2026\09\<id>\output\final.png`，而 `ArtifactView` 的 docstring 写着"绝不含裸路径" | 视图层改为 store key / 相对 tasks_root 的 key 形状（正斜杠），远端 store 时仍是对象键；docstring 改为真话。变异（直接返回 path）变红 |
| ArtifactKind 两个方向都漏 | 真实运行落库 5 种 kind，其中 `faces_dir`/`final`/`result`/`each_img_dir` **未声明**；而 `colorized`/`report` 声明了却无人写——原来那条"枚举成员必须可达"的门禁靠 metadata 键名和进度文案把它们判成了活的 | 补齐 4 个成员、删掉 2 个装饰成员，并加一条 AST 双向门禁（`add_artifact` 第二参 + 任意 `artifacts={...}` 字面量键）。两个方向各用一次变异证明会红 |
| 事件词汇表只单向门禁 | `task.rejected`、`stage.retried` 由生产代码写库，`EventType` 里没有，文档里也没有；反过来文档曾列过没人发的事件名 | 补 2 个成员；文档按"是否带 task_id"区分可流式与仅审计（`task.rejected` 落在无 task 行的拒绝路径，客户端看到的是 HTTP 503 `QUEUE_FULL`）；新增双向门禁，4 次变异全红 |
| `worker_max_queue` 零覆盖 | 队列饱和这条路径没有任何测试，"拒绝"到底拒到什么程度无人知晓 | 新增集成测试：拒绝后不得留下 tasks 行、不得留下产物目录、不得占队列槽位、`enqueue` 必须抛 503。**其中一条断言最初是假的**：我按 `tasks_root/<id>` 手写路径，而 `create_run_dir` 实际用 `tasks_root/<y>/<m>/<id>`，变异（把目录创建挪到检查之前）测不红；改成布局无关的扫描后才红 |
| `tabulate` 是个幽灵依赖 | `benchmark` extra 声明它，`benchmark/common.py` 注释写着"避免 tabulate 依赖"，自己打表 | extra 清空并注明原因；`export_locks.py --check` 不再把 benchmark 组列为"未安装" |
| running 任务不可取消 | 分体拓扑下对正在跑的任务 `POST /cancel` 得到 409 `TASK_NOT_CANCELLABLE`（实现只允许 `status='queued'`）。这不是 bug（消息与实现一致），但对 5–60 秒的 GPU 任务来说这个端点几乎用不上 | 记为待办：阶段边界协作式取消（worker 在下一 stage 前检查行状态），本轮未做 |

顺带用真实启动核对了几个数：一次 `restore`（450×630 老照片）在冷进程 6.2 s、热进程 1.6 s；
`psnr=23.175036017000377` 与现场用 numpy 独立复算**逐位相同**（SSIM 0.8258 是平台的高斯窗定义，
我用均匀窗复算得 0.85083，差异可解释）；下载字节的 sha256 与 artifacts 行一致；
SQLite 与 PostgreSQL 两条拓扑跑同一输入得到**同一结果摘要** `b3ffff54ea65…`；
`degraded_count=1` 的真实含义是"1 张图回退到未增强结果（原因 no_face_detected）"，
文档原来写成"人脸链处理了几张脸"，已改。

本轮跑到的门禁（都在真实环境，不是跳过）：

| 门禁 | 结果 |
|---|---|
| `pytest -m "not gpu"`（dev 3.14） | 1130 passed, 10 skipped, exit 0 |
| `pytest -m "not gpu"`（Anaconda GPU 3.11） | 1124 passed, 19 skipped, exit 0 |
| `pytest tests/gpu -m gpu`（CUDA + 全量权重 + dlib） | 28 passed, 0 skipped, exit 0 |
| `pytest tests/integration/test_postgres_server.py`（真实 PG 18.6） | 8 passed, exit 0 |
| 迁移往返（真实 PG） | upgrade → 8 表；`downgrade base` → 只剩 `alembic_version`；再 upgrade → 8 表 @0004；`runner check` 干净 |
| `ruff check` / `mypy` / `compileall` | 干净；mypy 抓到 `gpu_stats` 的 dict 值类型问题并修掉（119 文件 0 error） |
| `benchmark.latency --runs 2 --sizes small` | exit 0，overhead +0.0018 s，p50 0.0518 vs 基线 0.0514（×1.01） |
| `benchmark.throughput --tasks 10` | exit 0，10/10 排空；enqueue 1093 tasks/s |
| `scripts/export_locks.py --check` | exit 0，`requirements/` 与 `requirements.lock` 一致 |

## 三十、第二十九批：跑通之后接着做完 —— 取消语义，以及对方案再核一遍（plan §3.8、§2.6、§3.3）

上一批把 V3 真的启动起来，读到的第一个"做不到"就是取消：`POST /tasks/{id}/cancel`
只对 `queued` 有效，而一次 GPU restore 的 queued 窗口只有几百毫秒。这一批把它做完，
然后按同一套证据标准把方案再核一遍（新读只代理，双向证据，主线程逐条复验）。

### 取消：从 409 到"下一段边界停下"

| 项 | 现象 | 处理 |
|---|---|---|
| running 任务不可取消 | 实测 409 `TASK_NOT_CANCELLABLE`，消息 "missing or already running" | `cancel_task` 谓词放宽到 `status IN ('queued','running')`；行立即翻到 cancelled，状态视图/SSE/后续 fence 一次一致 |
| 谁来停 | runtime 每段开始前查一次 | `task_repo.interruption(task_id, owner_worker_id)`：`cancelled`（用户决定）或 `worker_id` 已不属于本进程（租约被 sweep 转交）才停；**行不存在/无主状态**一律不停——那既不是一个决定，又会让 harness 与内嵌用法在读取自己账本时停下来 |
| 停在边界而非中途 | stage 一旦开跑就跑完 | 明确写进文档：没有任何一段代码被写成可中途打断，打断电式的停在 CUDA 上下文里不可定义。因此单段计划仍会把这段跑完 |
| 取消不是失败 | `execute_queued` 的兜底 except 会写 `failed` + `task.failed` | 单加 `except TaskCancelledError`：只记一条 INFO 后重新抛出；行的 `error_code` 保持 NULL |
| 取消不能被重试 | worker 的失败路径按 `attempt_count < max` 重投 | 单加 `except TaskCancelledError`：ack 投递、不 retry、不花 attempt |
| 进度条继续爬 | `update_progress` 无 fence | 与 heartbeat/terminal write 一样按 `status='running'` 过滤，取消后停在原地 |
| Redis 拓扑 | 取消只改行，stream 里的唤醒条目不动 | 复核发现 `claim()` 早已对被丢弃的条目 `xack`（redis.py:130），且取消后的行不再是可领取状态 → 该条目自然被释放。补一条测试钉住"取消的任务永不会被交到 worker 手上、pending 归零"，而不是再写一遍代码 |

实测（真实 GPU worker，640×960 `restore`，领取后 0.35 s 取消）：
`POST /cancel → 200 cancelled`；在线流收到 `task.cancelled` 后正常关闭；
`global_restore` 记录 `completed`，其后三段从未开始；`error_code` 为 NULL、
`attempt_count=1`、未重投；`GET /result → 409`，artifacts 只剩 `input`；worker 日志
`run stopped by a cancellation` + `task cancelled; stopped at the stage boundary`。
测试面：仓库/运行时/worker/HTTP 四层共 7 条新断言，5 处变异分别变红（把谓词改回
`queued`、去掉进度 fence、去掉边界检查、去掉 execute_queued 分支、去掉 worker 分支），
对照全绿。三条旧断言（`test_cancel_only_queued`、Redis 状态机测试、
`test_cancel_only_affects_queued_tasks`）写的是旧语义，按新语义重写而不是删掉。

### 再核一遍方案：新发现（全部经主线程复验）

| 项 | 方案出处 | 复验结论 |
|---|---|---|
| `FIXIMG_TASKS_ROOT` 写在运维文档里但无人读 | §3.3 | **本轮已修**：`config` 之前把 `storage_root`/`tasks_root` 从 `base_dir` 硬推导，照文档设了环境变量的人其实仍在往系统盘写任务图，而且 TTL 清扫器读的还是同一个属性——于是"文档说换个卷"变成一件没人执行的事。现在两个根都可设，相对值以项目根为基准解析（不同工作目录的两个进程必须看到同一批字节），并补 2 条测试 |
| 上色链不走路由设备 | §4.3 Step 4、§7.1 | **确认待办（本批最高优先）**：其它三段都把 `context.gpu` 传给后端，`ColorizationStage` 不接这个参数，模型由 `model_manager` 按 `settings.device` 装载；于是 `MemorySampler(stage_device)` 与 `GPU_MEMORY_BYTES{device=…}` 会把并不在这张卡上的显存记到该 stage 名下 |
| 能力提示不看 options | §4.3 Step 4 | 待办：`required_capabilities(task_type)` 不读 `auto_colorize`，所以 `restore + auto_colorize` 会声称自己不需要 `colorize`，被一台没有 DDColor 的 worker 领走后照样上色（与 `deployment.md` 的承诺相反） |
| `ModelRequest.device` 在上色链里是死的 | §5.4、附录 B "Model Backend 统一接口" | 待办：生产代码里只有 global_restore 走 `.infer()`，ddcolor 的 `_do_infer` 仅被测试调用 |
| §7.1 "GPU 利用率" 无任何测量 | §7.1 | **已在第三十二批补齐**（见第三十三节）：当时只有显存与调度槽位，`utilization/nvml` 零命中 |
| §6 `ModelVersion` 未落库 | §6 | **已在第三十三批补齐**（见第三十五节）：当时无表无写入者，`/models/{name}/versions` 只报本进程驻留情况，`loaded_at`/`status` 只从 YAML 解析 |
| `RATE_LIMITED` 对 JSON 路由不可达 | §2.5、附录 B 统一错误格式 | **已在第三十四批补齐**（见第三十六节）：当时限流只保护 HTML `/register`，`tests` 内对 429/RATE_LIMITED 零断言 |
| §3.9.1 "处理中 ▸ ETA" 未产出 | §2.11、§3.9.1 | **已在第三十五批补齐**（见第三十七节）：当时进度条一行只有 bar/percent/stage，ETA 与 Queue 没人算，输入区三个勾选框一个都没画；同一批的测试还暴露出共享 SQLite 连接的线程安全问题 |

附录 B：36/37 完全满足，#16（GPU 显存指标）因上色链设备归属错乱只算部分满足；#4 形式上
满足（四个后端都注册了），但上色链的推理路径绕过统一接口。

### 本批门禁

| 门禁 | 结果 |
|---|---|
| `pytest -m "not gpu"`（dev 3.14） | 1143 passed, 10 skipped，0 failed |
| `pytest -m "not gpu"`（Anaconda GPU 3.11，部署解释器） | 1134 passed, 19 skipped，0 failed |
| `ruff check`（含入口脚本）/ `compileall` / `mypy` | 全清（mypy 119 文件 0 error；ruff 抓到一处因谓词收窄而失效的 import，已删） |
| 真实 GPU worker 取消实测 | 见上文 |

## 三十一、第三十批：让路由与显存数字在上色链上真的成立（plan §4.3 Step 4、§7.1、§3.5.4、§5.4）

上一批末尾排的第一项就是"上色链不接路由设备"。这一批做它的时候，用真实 GPU 环境把
一次 `colorize` 任务跑起来读它的数，结果顺出第二个更严重的问题：**显存峰值在每个进程的
第一个 stage 窗口里根本不是 0，而是"没有"**。

| 项 | 现象（实测） | 处理 |
|---|---|---|
| 上色链绕过 §5.4 后端契约 | `ColorizationStage` 自己从 model manager 取 pipeline、自己做 RGB→BGR→model→RGB；生产代码里唯一走 `ModelBackend.infer()` 的是 global_restore | 改为 `DDColorBackend.infer(ModelRequest(image, device=str(context.gpu)))`；`ModelRequest.device` 不再是死字段（审计第 3 项同时关掉） |
| 声明的精度策略没包住真实调用 | `inference_mode`/autocast 只在 `DDColorBackend._do_infer` 里进入，而上色生产路径根本不进那个函数 —— manifest 的声明对模型 API 为真、对用户每张上色的照片为假 | 走契约之后自动生效；测试用 spy 断言 `inference_context` 确实以"实际运行的设备"被调用（`tests/unit/test_ui_roles_and_progress.py`） |
| 路由设备到不了权重 | `context.gpu` 被调度、被 sampler 记帐，但 DDColor 的 vendored builder 默认"只要 torch 看得见 GPU 就用 cuda:0"：`FIXIMG_DEVICE=cpu` 的 worker 仍把约 1 GB 权重压到 0 号卡上，而报表把显存记在被调度的那张卡上 | `torch_device_for()` 统一一处翻译；`_build_ddcolor` 按配置建；`reside_on()` 在调用前把驻留的 pipeline 移到被调度的设备，并**返回它真正跑在哪**；报告里 `metadata.device` 是观测值，做不到时另给 `device_requested` |
| **第一个 stage 窗口的显存数永久缺失** | 真实 RTX 5060 + 部署环境实测：`colorization` stage 的 `metrics` 是 `{}`，而任务级/`/stats` 也拿不到任何数。根因是 `torch.cuda.reset_peak_memory_stats()` 在 CUDA 上下文建立之前抛 `Invalid device argument`，`MemorySampler` 把它当"诊断失败无所谓"咽掉 —— 于是**每个进程里最重的那个 stage（它正是第一个装载权重的）永远不上报显存** | `__enter__` 先 `torch.cuda.init()` 再 reset；仍失败时 `reset_peak_memory_stats`/`max_memory_allocated` 任一抛错都记一条一次性 WARNING（"measurement unavailable"），不再静默。修好后同一任务的实测值：stage 与任务级与 `/stats` 三处一致 `gpu_peak_mb=1768.0, gpu_device=0`（修前为 `{}`） |
| "设备描述符"被当成"设备存在" | `_is_cuda("1")` 在无 CUDA 的机器上返回 True → 每张 CPU 机上的上色调用都进 autocast，torch 打印"CUDA is not available, disabling autocast"然后按 fp32 跑：声明与执行不一致，而且看起来一致 | `_is_cuda` 的任何 CUDA 形式都要求 `torch.cuda.is_available()`；测试写成 `_is_cuda("1") is torch.cuda.is_available()`，两个解释器上都成立 |

### 这一批的证据链

* 变异各测一次，全红后复原：`reside_on` 不移权重；忽略可用性照样搬；策略用 `request.device`
  而不是实测设备；stage 回显请求设备；`_is_cuda` 退回 `isdigit()`；去掉 `torch.cuda.init()`；
  把失败重新咽掉。7 处逐个变红，对照全绿。
* 真 CUDA 环境（`tests/gpu/test_ddcolor_gpu.py`，7 passed）新增两条：`FIXIMG_DEVICE=cpu`
  必须真的建在 CPU 上（否则就是"配了不用"）；`reside_on` 搬动**真实** pipeline 并在两个
  方向上报告正确的设备。
* 启动实测（inline worker，`examples/color/o2.jpg`，640×960 灰度）：4.8 s 完成，
  `stage_order=0`、`stage_version="1.0"`、stage metrics `{gpu_peak_mb: 1768.0,
  gpu_device: 0}`、`/stats` 的 `models` 同时给出 `load_times_ms.ddcolor=3263` 与同一
  峰值 —— 三处口径第一次对上。
* 一个测试替身暴露了两处"替身比协议宽容"：上色 double 缺 `load`/`acquire`、`to()` 不接受
  `memory_format=`、fake CUDA 没有 `init()`。都补齐（让替身像真东西，而不是让产品迁就替身）。

附录 B 的 #16 与 #4 由此恢复为**满足**：上色链的显存数既存在又归到正确的设备，四条链
也都经统一后端接口进入生产路径（其余三条本来就是契约调用或 folder-bridge 契约）。
仍然只有 #37 的一半受环境限制（多卡路由与真实 Redis/S3 需要那种机器）。

## 三十二、第三十一批：路由提示必须描述真正会跑的那份计划（plan §4.3 Step 4）

上一批列的第一项。`FIXIMG_WORKER_CAPABILITIES` 的承诺写在 `docs/deployment.md` 里：
"worker 只领取 `required_capabilities` 是自己子集的任务"。承诺成立的前提，是那组能力
确实等于任务将跑的计划 —— 而它不等于。

| 项 | 现象 | 处理 |
|---|---|---|
| 提示只看任务类型 | `required_capabilities(task_type)` 调 `plan(task_type)` 不带 options，而 §12 的开关是在 `plan()` 里加段落的：实测 `restore + {"auto_colorize": true}` 声明的能力集是 `{restore, deblur, denoise, face_*, warp_back}` —— 有脸链、**没有 colorize** | 签名改为 `required_capabilities(task_type, options)`，把同一份 options 交给 `plan()`；入队处一并传下去 |
| 后果不只是"多跑一步" | 一台专门只做修复、没装 DDColor 的 worker 会领走这张图，然后在它不该服务的那张卡上色 —— §4.3 Step 4 的意义被反着用 | 端到端测试：入队后行里的 `required_capabilities` 含 `colorize`；只服务修复链的 worker `claim_next_task` 返回 None；补上 `colorize` 后才能领取 |
| 同一个瞎处在反方向更阴 | `{"face_enhance": false}` 的任务仍声明需要脸链能力 → 没有 dlib 的 worker 判定自己干不了，任务在"所有 worker 都健康"的队列里永远不领 | 关掉开关后提示收缩为 `{restore, deblur, denoise}`（按计划里的段落实算），测试用字面量断言而不是拿被测值自证 |
| 一致性没有门禁 | 以后再有任何"按开关加段落"，同一个洞会重新出现 | 参数化门禁：4 种任务类型 × 7 组 options，把 `plan()` 的段落名对着测试内手写的 `STAGE_CAPABILITIES` 表算出应有的能力集，断言提示 ⊇ 之（28 个用例） |

证据：两处变异分别变红（planner 内不把 options 传给 `plan()`；入队处不传 options），
对照全绿；`tests/inference/test_planner.py` 28 例 + `tests/integration/test_queue_reliability.py`
2 例新断言。本批之前的 81 个相关测试全绿却没发现这个洞 —— 因为它不在任何一条断言里：
所有旧测试都用默认 options 提问。

分机顶上的真实验证（API + 两个 worker，同一 SQLite）：提交
`{"face_enhance": false, "auto_colorize": true}` 的 `restore`，只跑一个
`FIXIMG_WORKER_CAPABILITIES="restore,deblur,denoise"` 的 worker —— 任务 12 s 内 23 次
轮询始终是 `queued`，就绪探针同时报 `queued: 1 / worker_liveness: starting /
oldest_queued_seconds: 12.1`；再启一个不设限的 worker，同一任务被领取并完成，
`stages = [global_restore 3158 ms (gpu_peak_mb 4356.0), colorization 8433 ms
(gpu_peak_mb 5967.4)]`，两段的显存峰值各归各的、且第一个 stage 也有数（上一批的
init 修复）。

顺带修掉一处会让人误判的日志：worker 启动行原来只打印 GPU 拓扑的能力，于是设了
`FIXIMG_WORKER_CAPABILITIES` 的 worker 报的是 `served_capabilities: null` —— 我自己在
真机上读到它就以为限制没生效，白找一轮。现在拆成 `claim_serves`（领取过滤真正用的）
与 `gpu_capabilities`（拓扑声明的），并用结构化字段断言钉住；把 `claim_serves` 改回
None 的变异会让这条测试变红。

### 一个"单跑通过、全量失败"的测试，根子在 alembic

上面那条断言日志字段的测试单独跑绿、放进整套里就红。查下来不是时序问题（我先按"等一等
再断言"改过一版，仍然红），而是 **`logger.disabled=True`**：`migrations/env.py` 用默认
参数调 `fileConfig(...)`，而 alembic 模板的默认是 `disable_existing_loggers=True` ——
它会禁用调用时刻**已存在的所有 logger**。于是任何以编程方式跑过一次迁移的进程（嵌入式
用法、`fiximg-db upgrade`、迁移测试）之后的应用日志全部静默：服务照常响应、健康检查
照常绿，只是不再产出任何日志。测试套里它还会毒化后面的每一条"断言某条日志确实发出了"
的用例，且只在迁移测试先跑的时候。

改成 `fileConfig(config.config_file_name, disable_existing_loggers=False)`，并加一条
门禁测试：跑完 `runner.upgrade()` 之后，一个迁移前就存在的 logger 不得是 disabled。
把参数改回默认，该测试立刻变红；去掉参数之后整套的顺序依赖也一起消失（1182 passed,
10 skipped, exit 0）。

这类洞只有"把真实服务/真实测试套跑起来读现场"才会现形：它不在任何一条断言里，也不在
任何一个错误码里，症状是"什么都能跑，就是没日志"。

### 顺带：部署用 `FIXIMG_DATABASE_URL` 会击穿测试隔离

同一批里，把部署环境的整套测试跑第二遍时 `test_bootstrap.py` 报了三条"用户已存在"。
不是产品的错，也不是测试写错：引擎是在**调用时**读 `FIXIMG_DATABASE_URL` 的
（`apply_database_url()`，被 `create_app()`、迁移 runner、以及本批新加的
`cli.worker.prepare_database()` 调用），所以 shell 里导出过这个变量，就能把 fixture
刚设好的 `DB_PATH` 再搬回同一个文件 —— 于是整套共用一个库，先跑的测试给用户建了
admin，后跑的看到"已有用户"。`conftest.py` 里原来的注释写着"直接改 `_conn` 会让
部署级 URL 赢过 DB_PATH"，但它只防了自己赋值，没防环境。

修法是会话级 autouse fixture 把变量摘掉（谁真要用谁 `monkeypatch.setenv`，例如
"独立 worker 必须打开配置里那个库"那条测试）。验证方式就是把毒条件重新摆上：
带 `FIXIMG_DATABASE_URL=...gate_leak.db` 跑两套解释器的全量测试 —— 全绿，而且那个
文件**没有被创建**（说明测试真的没往它写）。

这类"CI 与本机行为不一致"的根因基本都是这种全局读取：配置在调用点读环境，而测试只
在赋值点做隔离。看到"单独跑绿、全量红"时先查环境，再去查时序 —— 我第一版就是按
"日志没来得及写"改的等待循环，白改一轮。


## 三十三、第三十二批：GPU 利用率从"没测"变成"每阶段有数"（plan §7.1、§2.4）

§7.1 的 KPI 表把"GPU 利用率"写成**实测**，§2.4 整节（把全局锁换成 scheduler +
concurrency policy）存在的理由就是"GPU 利用率未必高"。这两句在代码里没有对应的
测量：`utilization` / `nvml` 在 `src/` 零命中，能报的只有显存与调度槽位数。
于是"槽位放开了两张卡是不是真的更忙"这个问题，改完也仍然无法回答。

| 项 | 现象 | 处理 |
|---|---|---|
| 利用率没有任何生产者 | torch 不提供利用率 API；`MemorySampler` 量的是分配，`scheduler.stats()` 量的是槽位计数 | 新增 `inference/gpu_utilization.py`：驱动查询（`nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits`，实测本机输出 `0, 0, 496, 8151`）+ 每进程一个采样线程 + 带时间戳的环形缓冲 + 窗口聚合 |
| "stage 结束时读一次瞬时值"是个假测量 | 四条链里三条走 vendored 子进程，返回时卡已经空了，瞬时值恒为 0；写出来就是"这条 pipeline 让 GPU 全程空闲"这种自证其罪的数 | 利用率是**窗口统计**：runtime 在 stage 前后取探针自己的时钟（`probe.now()`），聚合落在区间内的读数（mean / max / 样本数），与显存那条 `gpu_peak_mb` 同一个归因方向 |
| 每次查询是一个子进程 | 常驻 0.5 s 轮询 = 每秒 2 次 fork，空闲 worker 也在付；而 API 节点被抓取一次就永久轮询下去 | 只在"有 stage 窗口开着"或"刚有读者要数"（2 个间隔）时轮询；停止轮询后读数会变旧，所以负载里带 `age_seconds`，让"几分钟前的 0 %"和"刚才的 0 %"可区分 |
| 没有驱动的机器必须报"没有"，不能报 0 | 伪造的 0 % 在 CPU 主机上读起来像"调度器让卡闲着"，正是这个指标存在的目的的反面 | 查询失败**或查询返回空行**都计入失败，连续 3 次即整进程 `unavailable` 并停止线程（不再重启、只 WARNING 一次）；`window()` 与 `describe()` 返回空，`status` 说原因 |
| 任务级合并会重演"最后一个赢" | `_collect_stage_metrics` 原来对未登记的名字一律后写覆盖：4 个采样的 100 % 会被 1 个采样的 20 % 顶掉 | 三条策略各归各的：`gpu_util_pct` 按 `gpu_samples` **加权平均**，`gpu_util_max_pct` 进 `_METRIC_MAXIMA`，`gpu_samples` 进新的 `_METRIC_SUMS`（任务级报的是它平均过的样本总数，读者能自验） |
| 消费点 | 声明了却没地方读 | `/health/ready` 的 `gpu.topology.utilization`（status / interval / age / devices / busy_devices）、每 stage 的 `metrics_json` 与 `metrics` 表、`gpu_utilization_percent{stage,device}` 序列进 `/metrics` 与 `/stats` |
| 一张卡上的两个 worker | 驱动给的是**整卡**读数，A 窗里会算进 B 的活 | 文档写清（`deployment.md`），并把"按 `gpu_util_max_pct` 定并发、按整任务的多个 stage 横向读均值"作为用法 |

`busy_devices` 只在真问过卡的时候出现：`describe([-1])`（CPU-only 拓扑）返回
`{"status", "interval_seconds", "devices": []}`，**没有** `busy_devices` ——
"我问了 0 张卡"和"我问了卡，它们都空闲"是两件事，合并成一个 0 就是假数据。

证据分三块：

1. **变异**：14 处各自变红、恢复按字节一致（去掉就绪块的 utilization 行、去掉窗口
   折叠、不预热探针、不开窗口、不关窗口、`_should_poll` 恒真、`close_window` 允许
   负数、去掉 `age_seconds`、CPU stage 也开探针、不记 `gpu_utilization_percent`、
   均值退回后写覆盖、max 退回后写覆盖、空行当读数、CPU-only 拓扑漏出 busy_devices）。
   "没去 fork 子进程"这种否定断言不能用等待证明，所以它测在 `_should_poll()` 的
   决策逻辑上（计数时钟，2 个间隔后必须过期）。
2. **门禁**：见下表。
3. **真机实测**（`fixoldimg-gpu`，RTX 5060，`auto_restore` `examples/old/f.png`
   372×524，整任务 14.5 s）：

```text
task-level   gpu_util_pct 13.4（26 个采样的加权均值）, max 87
scratch_repair    5358 ms   mean 32.4 %   max 87 %   10 采样
face_detection     807 ms   mean  0.0 %   max  0 %    1 采样
face_enhancement  4216 ms   mean  3.0 %   max 24 %    8 采样
warp_back         4098 ms   mean  0.0 %   max  0 %    7 采样
/health/ready → {"status": "measuring", "interval_seconds": 0.5,
                 "age_seconds": 1.0, "devices": [{device 0, utilization_pct 0,
                 memory_used_mb 1123, memory_total_mb 8151}], "busy_devices": 0}
/metrics → gpu_utilization_percent{device=0,stage=scratch_repair} avg 32.4 …
```

这批真正的产出是这个数本身：**一条 face 流水线整任务只让卡忙 13 %**，最重的
`face_enhancement`（4.2 s）均值 3 %、`warp_back`（4.1 s）0 %。 §2.4 担心的"利用率
未必高"被证实，而且定位了它不在卡上——三段里两段根本不把活交给 GPU，
剩下的 `scratch_repair` 也只有三分之一时间在路上。要提利用率，先动的是子进程桥接与
CPU 侧段落，不是把 `FIXIMG_CONCURRENCY` 调大；这也把第十四节"有前提的性能项"的
前提说清了：batching 值得做的地方是 `scratch_repair`，不是整条 pipeline。

### HTTP 现场（同一环境，API 单进程 + 真 GPU）

`age_seconds` 为什么必须在负载里，这一次读到了实例：静止 17 s 后的第一次数是

```text
{"status": "measuring", "age_seconds": 17.5, "devices": [{utilization_pct: 0, …}],
 "busy_devices": 0}      ← 数是真的，但它是 17.5 s 前的
{"status": "measuring", "age_seconds": 0.5,  "busy_devices": 0}          ← 问过之后的刷新
提交 colorize 任务，运行中：
{"status": "measuring", "age_seconds": 0.1,  "devices": [{utilization_pct: 100,
 memory_used_mb: 2585, memory_total_mb: 8151}], "busy_devices": 1}
任务结束后： utilization_pct 0 / busy_devices 0
stage 行： colorization 239 ms {"gpu_peak_mb": 1749.0, "gpu_device": 0,
         "gpu_util_pct": 100.0, "gpu_util_max_pct": 100, "gpu_samples": 1}
序列：  gpu_utilization_percent{device=0,stage=colorization} count 2, avg 52.9,
         p50 5.8, p95 100.0（冷启动那次 5.8 %，权重已常驻那次 100 %）
结果：  GET /result 200，329071 字节，sha256[0:16] 962785b1690f2f96
```

一个只有 1 个采样的 239 ms stage 也能报出 100 % —— 短 stage 的均值薄，这正是
`gpu_samples` 要一起报出来的原因；跨 stage 的任务级均值按它加权。

### 本批门禁

| 门禁 | 解释器 | 结果 |
|---|---|---|
| `pytest -m "not gpu"` | 部署环境 `fixoldimg-gpu`（3.11） | 1196 passed, 19 skipped，exit 0 |
| `pytest -m "not gpu"` | dev 3.14 | 1205 passed, 10 skipped，exit 0 |
| `pytest tests/gpu -m gpu` | 部署环境（真 CUDA + dlib） | 30 passed, 0 skipped，exit 0 |
| `ruff check src tests benchmark scripts migrations main.py worker.py run.py` | 3.11 | All checks passed |
| `compileall`（3.11 语法） | 3.11 | exit 0 |
| `mypy` | 3.14（`fixoldimg-gpu` 未装 mypy） | Success: no issues found in 120 source files |
| 变异 | 3.11 | 15 处各自变红、按字节恢复 |


## 三十五、第三十三批：§6 的第五个数据模型终于有了表（plan §6、§3.15）

§6 列了五个数据模型，前四个（Task / TaskStage / Artifact / Metric）都有表、有写入者，
第五个 **ModelVersion 只有 `domain/models.py` 里的 dataclass 和 manifest 解析**。后果不是
"少一张表"而是两件可观察的事：`status` / `loaded_at` 报的是 YAML 里**声明**的值而不是
**发生**过的事；`GET /api/v1/models/{name}/versions` 只能描述发起请求的那个进程 ——
纯 API 节点（`FIXIMG_INLINE_WORKER=0`）对它自己的 worker 正在服务的模型报"没有版本"。

| 决定 | 理由 |
|---|---|
| 按 `(name, version)` **upsert**，不是每次装载 append | 一个版本重载十次还是那一个版本；把重载变成"多出十个可用版本"会让 `/versions` 不能用来做发布判断。`load_count` 才是"跑过几次"的表达 |
| 声明字段用 `COALESCE(excluded.x, 旧值)`，观测字段一律取新值 | 写 NULL 的含义是"这个进程没读到 manifest"，不是"部署不再声明它"。把 `sha256` 抹掉会让 §3.5.2 的校验从此有个看起来正常的空值。反过来 `last_error` 必须被成功清掉，`loaded_at` 只在成功时盖章 —— 于是"试过但没起来"和"昨天起来过"可区分 |
| **卸载不写行** | 一个进程丢掉句柄，说明不了另一个进程是否还在服务它。原来 `ModelStatus.REGISTERED` 恰好同时表示"没装载过"和"从没装载成功"，写进去就是替全部署下一个没人能证实的结论 |
| 只有装载经过版本注册表的模型才有行 | 今天只有 `ddcolor`。四条 legacy-cli 链的权重在子进程里装载，这张表里给它们造一行"ready"就是假观测；它们的出处按 plan §2.6 走 `artifacts.model_version`（第三十一批已接上生产者） |
| 写入是尽力而为，且 `ensure_core_schema()` 在 try 内部 | 审计轨不能让模型装载失败。也是在这里翻出一个**会被静默咽掉的 NameError**：新函数引用了 `settings` 而 engine 模块没有这个名字 —— 若没有"全新进程从没跑过 init_db"这条测试，它会表现为"catalog 永远是空的"，日志里一句 WARNING |

### 顺带修掉的两处"测试自己不管用"

1. **头和链路是手抄的**：加了 0005 之后，`test_migrations_alembic.py` 里六处写死
   `"0004"` 的断言同时变红 —— 这跟迁移要解决的漂移是同一件事。现在从
   `migrations/versions/*.py` 解析出链条、头部以及"这一版拥有哪个表/列"，
   `EXPECTED_TABLES` 也从运行期真正用的两份 DDL 常量里正则取。再加一条
   `test_a_database_created_before_the_tip_revision_gains_its_object`：新建的库由 0001
   的 shipped DDL 直接带上表，**其余测试全绿也证明不了 0005 真会建东西**；这条测试
   模拟运维实际经历的那次升级（昨天的库 + 今天的修订），把 tip 的 DDL 变成可证的。
2. **"双向"schema 比对其实恒真**：第一版把 HTTP 响应体和 `RecordedVersionView` 的字段
   对来对去。`response_model` 会给缺失的声明字段填默认值，所以查询少选一列时body 里
   照样什么键都有 —— 两条变异（改名 / 漏字段）全绿。改成拿**仓储返回的行**跟 schema 比，
   响应体只用来验证值能穿过序列化，两条变异立刻变红。

### 证据

14 处变异分别变红、按字节恢复（端点不再读 catalog、sha256 可被抹掉、失败覆盖
`loaded_at`、失败也盖章、计数不再累加、成功不清错误、装载/失败不再写行、审计异常
会打断装载、tip 修订不建表、懒建表守卫失效、shipped DDL 少表、行的字段与 schema
两种错位）。真机两进程 + 真 PostgreSQL 18.6：

```text
catalog34（干净库，runner upgrade → head=0005，12 列齐）
API 节点（FIXIMG_INLINE_WORKER=0）装载前：  residents [] · catalog []
worker-gpu1 跑完一个真 colorize 任务之后，同一个 API 节点：
  residents [] · active_version null                    ← 仍然正确：这台没装过
  catalog [{name: ddcolor, version: 1.0.0, status: ready, framework: pytorch,
            weight_uri: weights/ddcolor/pytorch_model.pt, sha256: null,
            loaded_at: 2026-09-28T00:09:07.318036Z, load_ms: 3418, load_count: 1,
            writer: worker-gpu1,
            metadata: {channels_last: true, precision: fp16, compile: false,
                       warmup: first-use}}]
任务： colorization 5207 ms {"gpu_peak_mb": 1768.0, "gpu_device": 0,
       "gpu_util_pct": 4.6, "gpu_util_max_pct": 23, "gpu_samples": 9}
结果： GET /result 200，329071 字节，sha256[0:16] 962785b1690f2f96（与上一批同图同 digest）
```

`sha256: null` 是"这个部署没声明摘要"，不是校验结果 —— 文档按这个口径写。
同一份 upsert 语义在真 PostgreSQL 上单独跑过（`ON CONFLICT` + `COALESCE` +
`NULLIF(metadata_json,'{}')` 与 SQLite 行为一致），不是只在 SQLite 里成立。

### 本批门禁

| 门禁 | 解释器 | 结果 |
|---|---|---|
| `pytest -m "not gpu"` | 部署环境 `fixoldimg-gpu`（3.11） | 1213 passed, 19 skipped，exit 0 |
| `pytest -m "not gpu"` | dev 3.14 | 1222 passed, 10 skipped，exit 0 |
| `pytest tests/gpu -m gpu` | 部署环境（真 CUDA + dlib） | 30 passed, 0 skipped，exit 0 |
| `ruff check`（含入口脚本）/ `compileall` | 3.11 | All checks passed / exit 0 |
| `mypy` | 3.14 | Success: no issues found in 121 source files |
| 迁移往返（真 PG） | 3.11 | `upgrade → check(head=0005) → 表与 12 列齐`，卸载 PG 服务 |
| 变异 | 3.11 | 14 处各自变红 |

新增测试 15 条（`tests/unit/test_model_catalog.py`）+ 迁移测试 2 条。
§6 的五个数据模型至此都有表、有写入者、有跨进程读法。


## 三十六、第三十四批：把 `RATE_LIMITED` 从"文档里有"变成"路由发得出"（plan §2.5、附录 B 统一错误格式）

方案只要"统一错误体系"（AppError / ErrorCode / handler），没要求限流。这一项是**声明层
审计**抓出来的：`docs/api.md` 把 `RATE_LIMITED` 列进"客户端应当处理的通用码"，
`domain/errors.py` 里 `RateLimitedError` 也在，但**全代码无人构造它** —— 唯一的 429
来自 HTML `/register`，它返回的是自己渲染的页面，压根不走信封。测试目录里 `429` 与
`RATE_LIMITED` 当时是**零命中**。同类问题：`infrastructure/security/rate_limit.py` 里
还挂着一个自己写明"semantic placeholder"的 `RateLimitExceeded`，以及一个从来没人构造的
`AuthError`（401 其实由 status→code 表给）。

| 项 | 决定 |
|---|---|
| 限什么 | `POST /api/v1/tasks`。队列**容量**是另一件事，已有 `QUEUE_FULL` 在入队处回答；这里只防一个调用方把端点刷爆 |
| 按什么键 | 认证主体 `user:<id>`，没有身份时退回客户地址 `ip:<addr>`；一个 NAT 出口后的一百个账号不该互相抢配额。`details.scope` 把用了哪种键说出来 |
| 在哪拦 | **解码图片之前**。测试专门用"超限 + 坏图片"这一组合断言它拿的是 429 而不是 400 —— 限流的意义就是洪水时不再付解图的代价 |
| 报什么数 | `limit` / `window_seconds` / `remaining` 全部取**刚刚做出裁决的那个 limiter 实例**，不取模块常量：两者会不一致（重建 limiter、`FIXIMG_SUBMIT_MAX=0` 被夹到 1），而一个跟实际行为对不上的 `details.limit` 比不给数更糟。这一条是变异逼出来的：第一版写了模块常量，`limit == 2` 的断言直接变红 |
| 卸载/装载别的 | `AuthError`、`RateLimitExceeded` 删除：401 由 status 表映射，异常式调用风格今天不存在 |
| 多副本 | 配了 `FIXIMG_REDIS_URL` 就用 Redis 计数（各副本共享配额），Redis 挂了退回内存（fail-open）—— 缓存故障不能把服务打死 |

### 把这一类关成门禁

`tests/unit/test_domain_model.py` 新增两条：

1. **每个声明的 error 类都必须被产出**（在 `src/` 里被构造或被 raise），否则就是
   客户端永远收不到的码；基类除外。`ErrorCode` 的每个取值要么由"被产出的类"携带，
   要么出现在 `api/errors.py` 的 status 兜底表里。
2. **文档里列的每个码都必须可达**：期望从 `docs/api.md` 的 "Common codes:" 那一段
   现读，而不是在测试里再抄一份清单 —— 抄来的两份清单会一起漂。

写门禁的过程本身抓到两处：

- 第一版用文本正则从类体里取 `code = ErrorCode.X`，于是
  `ArtifactNotFoundError`（它自己不声明，从 `ArtifactError` 继承
  `ARTIFACT_NOT_FOUND`）被判成"没人能发的码"。改成**用真实类做内省**（`cls.code`
  自然沿 MRO 解析）。
- 第二版从 "Common codes:" 取到下一个 `## ` 标题为止，把我新加的说明段落里的
  `FIXIMG_SUBMIT_MAX` / `FIXIMG_SUBMIT_WINDOW` 也当成错误码扫进来了 —— 门禁自己
  变红。改成只取那一段（到空行为止）。

### 证据

变异 8 处各自变红、按字节恢复（去掉守卫；回报模块常量；把守卫挪到解图之后；键改成
按地址；去掉 `remaining`；auth 退回 import 期绑定 limiter；塞回一个没人产出的 error
类；文档里加一个发不出的码）。真机（`FIXIMG_SUBMIT_MAX=3`、`FIXIMG_SUBMIT_WINDOW=120`，
API 单进程，SQLite scratch）：

```text
连续四次 POST /api/v1/tasks： 202, 202, 202, 429
429 信封：{"error": {"code": "RATE_LIMITED",
                     "message": "Too many task submissions (3 per 120s); try again later",
                     "request_id": "eeb0db8ed80f",
                     "details": {"limit": 3, "window_seconds": 120,
                                 "scope": "principal", "remaining": 0}}}
GET /api/v1/tasks： 队列里就 3 条，全是 queued     ← 被拒的那次没有入队
```

另外给测试套加了一条会话级 autouse fixture，把 submission limiter 换成每测试一个新
实例：limiter 是**进程级滑动窗口按 principal 计数**，整套一分钟里从同一个账号提交几十
次，不设隔离就会出现"跟执行顺序有关的真 429"—— 与 `_no_deployment_database_url`
同一类污染。

### 本批门禁

| 门禁 | 解释器 | 结果 |
|---|---|---|
| `pytest -m "not gpu"` | 部署环境 `fixoldimg-gpu`（3.11） | 1220 passed, 19 skipped，exit 0 |
| `pytest -m "not gpu"` | dev 3.14 | 1229 passed, 10 skipped，exit 0 |
| `pytest tests/gpu -m gpu` | 部署环境（真 CUDA + dlib） | 30 passed, 0 skipped，exit 0 |
| `ruff check`（含入口脚本）/ `compileall` | 3.11 | All checks passed / exit 0 |
| `mypy` | 3.14 | Success: no issues found in 121 source files |
| `benchmark.latency --runs 2 --sizes small` | 3.14 | exit 0；p50 0.052 s 对基线 0.0514 s = x1.01 |
| 变异 | 3.11 | 8 处各自变红、按字节恢复 |

新增测试 5 条（`tests/api/test_rate_limit.py`）+ 2 条门禁（`test_domain_model.py`）。
统一错误码这一类从此有静态门禁兜着：文档写"客户端应当处理这些码"，代码就必须真发得出来。


## 三十七、第三十五批：进度条终于有 ETA 与队列，输入区终于有开关（plan §3.9.1、§2.11）

§3.9.1 的布局图在进度条那一行要四个东西（bar + percent、当前阶段、ETA、Queue），在输入
区要三个勾选框（□ Face Enhance / □ Auto Colorize / □ High Resolution）。代码里有前两
个，**ETA 与 Queue 从来没产出**，三个勾选框**一个都没画** —— `make_submit_handler(mode,
options=...)` 收 options 参数，但 5 个调用点全部只传 mode，`options` 永远是 `None`。

| 项 | 决定 |
|---|---|
| ETA 怎么算 | 用**运行时自己写的 percent** 外推：`elapsed × (100 − progress) / progress`。不引入"按阶段名查历史 p50"那套，因为那条数据描述的是**别的图**：同一阶段在本流水线里随图片内容的变化比随阶段名大得多 |
| elapsed 从哪算起 | **这次执行的 started_at**，不是提交时刻。排队十分钟的任务在第一个 25 % 刻度上会报 "ETA ~10 min"，而那十分钟里模型一秒都没跑 |
| 什么时候不报 | progress < 20 % 或已跑不足 1 s 时报 `ETA --`。四段计划第一个有效刻度就是 25 %，而一个 200 ms 的采样外推出来的分钟级承诺一定是错的 —— `--` 是关于估计的陈述，不是一个小估计 |
| Queue 显示什么 | 队列**深度**（整个部署的积压，与 `/health/ready` 同一个数），不是本进程视图。`Queue 0` 与 `Queue ?` 分开：空队列是"问过、答案是零"，读不到才是未知 |
| 勾选框放哪些 | 只放**planner 真会读的**键。`face_enhance` / `auto_colorize` / `hr` 走 `plan(task_type, options)`，所以两个修复 tab 给；`colorize` / `detect` 没有脸链可关，不给 |
| 取消勾选的含义 | 发 `false`，不是"用任务类型默认值"。用户看得见、能点的控件不能有第二层隐藏含义 |
| 控件接错怎么办 | handler 里加arity 守卫：checkbox 数量与声明的开关数量不符就明确报错。原来 `zip` 会静默截断 —— 少接一个框就等于把用户最后动过的那个开关丢掉 |

### 顺带查出来的：`auto_restore` 根本不读 options（**第三十六批已补齐，见 §三十八**）

`plan_auto_restore(analysis)` 只看分析结果，`face_enhance` / `auto_colorize` 传进去也没人
读。所以 **Auto tab 也不放勾选框** —— 放了就是装饰（勾了没反应）。要让"一键修复"也
能显式拒绝上色/脸链，得先让 `plan_auto_restore` 接受 options，这属于新增能力，记在下面
的剩余清单里。

### 一条自己就是装饰的测试，被变异抓出来

第一版 `test_build_switches_...` 写的是
`[box.value ...] == list(PIPELINE_MODES[mode]["switches"].values())` —— 拿被测值当期望
值，把默认值从 `True` 改成 `False` 它照样绿。改成两条：断言字面量 `[True, False, False]`
与真实 caption；再加一条**行为**断言 —— "按画出来的默认值提交"与"什么都不传提交"必须得
到同一份计划（独立来源：`planner.plan(task_type, {})`）。默认值被改动会立刻改变计划，
两条同时变红。

### 顺带：全量测试里"每次换一个测试红"的那个失败

第三十五批的门禁在 dev 解释器（3.14）上两次全量各红一条，**且不是同一个测试**：
一次是 `test_face_enhance_can_be_opted_out` 拿到 `ValueError: None is not a valid
TaskStatus`，一次是取消用例在 `commit()` 里抛
`SystemError: error return without exception set`。单独跑都绿，3.11 全量也绿。

写个并发小脚本压同一个连接就复现了，根因与 §3.2 的拓扑直接冲突：inline worker 与
HTTP 请求线程**共用一条 SQLite 连接**，而 `Connection.execute()` 返回的是
`sqlite3` 的**共享游标** —— 取数发生在锁外，于是

* 线程 B 在 A `execute` 与 A `fetchone` 之间 `execute`，A 取到的是 B 的语句结果
  （`SELECT COUNT(*) AS n` 回答 `None` → `int(row["n"])` 崩）；
* 两个 `commit()` 交错，3.14 的驱动直接给 `SystemError`/`InterfaceError`。

`check_same_thread=False` 只关掉"这条连接归谁"的检查，**不会**把一次事务变成线程
安全 —— 一次事务在这里是若干次 C 调用。修法在包装层：`Connection` 每个方法持
实例级 `RLock`，且 `execute()` **在锁内把结果取空**，返回一个 `_Result`
（`fetchone`/`fetchall`/`rowcount`/`__iter__`，行对象仍是 `row_factory` 产出的那
些）。仓库层一行未改。

回归是 `tests/integration/test_shared_sqlite_connection.py`：4 写线程 × 40 次
create/start/progress，加 2 个"读 + commit"线程，屏障同步后起跑。断言的是行为而不是
"没抛异常"（丢写也能过后者）：每条任务都读得回来、状态与进度是写进去的值。
把 `execute()` 改回修复前的形状（直接返回驱动游标），它在 3.14 上立刻变红；恢复按
字节一致。

**这一类值得记住**：`Connection` 原来那句"一个共享连接在 SQLite 上是正确的，因为
sqlite3 自己会串行化写者"是**未经验证的假设**，而它写在文档字符串里、被后来的每一批
当作前提。跨线程共享的从来不是"连接"，是"语句 + 取数 + 提交"这三步。

### 证据

12 处变异各自变红、按字节恢复（ETA 在无任何信号时也报；ETA 用已过时间而不是剩余时间；
ETA 按提交时刻起算；队列不再问、写死 0；空队列读成未知；开关被静默截断；arity 守卫去掉；
无开关 tab 发出空 options；给出一个 schema 不认的开关；tab 漏接 checkbox；默认值翻转）。
连接层修复另有一条独立变异（`execute` 退回返回驱动游标 → 并发回归变红）。

真机（`fixoldimg-gpu` + 真 GPU，`restore` 全链 17.0 s，inline worker，34 帧）：

```text
[0]  [░░░…░] 0%  · queued            · ETA --     · Queue 1     ← 排队时就有数
[1..8] 0%  · global_restore         · ETA --     · Queue 0     ← 首段不猜
[9]  [█…░] 25% · face_detection      · ETA ~14 s  · Queue 0
[11] [██…░] 50% · face_enhancement   · ETA ~6 s   · Queue 0
[21] [███…░] 75% · warp_back         · ETA ~4 s   · Queue 0
[31] [█████░] 99% · warp_back        · ETA ~0 s   · Queue 0
[33] ✅ Restore (no scratches) completed in 17.0 s
```

即 25 % 时报 "~14 s"，剩下三段实跑约 13 s —— 估计的尺度是对的，而 `ETA --` 出现在连一个
阶段都没完成的时候。

### 本批门禁

| 门禁 | 解释器 | 结果 |
|---|---|---|
| `pytest -m "not gpu"` | 部署环境 `fixoldimg-gpu`（3.11） | 1245 passed, 19 skipped，exit 0 |
| `pytest -m "not gpu"` ×2 | dev 3.14 | 1254 passed, 10 skipped，连续两次 exit 0（连接层修好后不再有"每次换一个测试红"） |
| `pytest tests/gpu -m gpu` | 部署环境（真 CUDA + dlib） | 30 passed, 0 skipped，exit 0 |
| `ruff check`（含入口脚本）/ `compileall` | 3.11 | All checks passed / exit 0 |
| `mypy` | 3.14 | Success: no issues found in 121 source files |
| `benchmark.latency --runs 2 --sizes small` | 3.14 | exit 0（最小尺寸对 1.000 s 门禁） |
| `scripts/export_locks.py --check` | 3.11 | exit 0 |
| 真 PostgreSQL 18.6 | 3.11 | `test_postgres_server.py` 全绿；`upgrade → downgrade base → upgrade → check`（head=0005）往返干净 —— 连接包装器改过，这一层必须重跑 |
| 变异 | 3.11 / 3.14 | 12（ETA/开关）+ 1（连接退回旧形状）+ 1（游标双件缺 `description`）各自变红、按字节恢复 |

新增测试 25 条：ETA/队列/开关 22（`tests/unit/test_ui_progress_eta.py`）、连接并发 2
（`tests/integration/test_shared_sqlite_connection.py`）、游标双件门禁 1。


## 三十八、第三十六批：`auto_restore` 终于读 options，顺带把"两种形状的 JSON 列"做成一类门禁（plan §12、§3.9.1、§4.3 Step 4）

第三十五批留下一条明确的账：`plan_auto_restore(analysis)` 只看分析结果，所以 Auto tab
不能放开关 —— 放了就是装饰。这一批把那句话反过来做：**先让 planner 读 options，再给
Auto tab 放开关**。开工前照例先要两条证据。

| 证据 | 内容 |
|---|---|
| 生产调用点 | `runtime._execute_core`（`src/fiximg/inference/runtime.py:483`）里 `context.options` 就在手边，静态计划那一支已经在用它（`plan(task_type, options=…)`），动态那一支调用 `plan_auto_restore(analysis)` 时不传。走这一行的是 **worker 进程**（队列路径）与 web 进程（同步回退路径） |
| 客户端早就在发 | `POST /api/v1/tasks` 的 options 白名单不看任务类型（`_ALLOWED_OPTIONS` 是全局的），所以 `type=auto_restore` + `{"face_enhance": false}` 一直是**收得下、存得进、没人读** |
| 变异判定 | 注掉 `plan_auto_restore` 里 `if analysis.get("face_count")` 那一段，测试会红 —— 被分析驱动的分支有覆盖；但**没有任何一条测试能给它传 options**，因为签名里没有这个参数。"开关在 Auto 链上被忽略"这件事，当时不可能让任何测试变红 |

### 决定表

| 问题 | 决定 |
|---|---|
| 三态怎么定 | 键缺席 → 由分析决定（保持 §10 原行为）；`true` → **强行加这一级**，哪怕分析说不用；`false` → **永不加**，哪怕分析说要用。与 `plan()` 里"显式值就是指令"完全同构 |
| 为什么 `true` 必须能强行 | 分析器在**缩到 512 px 的副本**上跑（`_ANALYSIS_MAX_SIDE`），大图扫描件里的小脸它根本看不见；泛黄/手工上色的黑白照又会被判成"有颜色"。不给 force，用户唯一的办法是离开 Auto tab 用固定恢复链 —— 那就连 scratch 判定一起丢了 |
| Auto tab 的勾选框发什么 | 勾上 = **不发这个键**（分析仍然是权威），不勾 = 发 `false`。复选框只有两个布尔态，而 Auto 的正确默认是"由图片决定"，把 `true` 当成"允许"发出去就等于替用户强行加级 —— 那不是我勾框的意思 |
| 为什么 caption 必须按 tab 改写 | 同一个键在两个 tab 上含义不同（恢复 tab 勾上=一定跑脸链；Auto 勾上=分析说跑才跑）。`SWITCH_LABELS_BY_MODE` 存在就是为了这个，caption 复用会承诺 checkbox 不提供的行为 |
| `hr` 为什么不是 plan switch | 它由 stage 自己从 `context.options` 读（`face_enhancement`/`warp_back`/`global_restore` 的 `_hr()`），改的是"这一级怎么做"，不是"要不要这一级"。把它当 plan switch 的话，`{"hr": false}` 会变成"不要脸链" —— 用户没要求的计划变更 |
| scratch 分支 | 没有 checkbox，也不该有：它是 Auto tab 存在的理由，`test_the_scratch_branch_is_nobody_elses_decision` 用四种 options 组合钉住"分支只由 `scratch_score` 决定" |
| 路由提示怎么办 | 仍然对 `auto_restore` 返回空集 = 不加限制。两个开关即便都写明，`with_scratch` 那一支还是看图决定，任何"子集"承诺都可能是夸大 —— 而**夸大能力要求会把任务 stranded 在一个人人都健康的队列上**（第三十一批写下的失败模式）。`deployment.md` 里加了这段解释 |
| 报告要能分辨"被拒绝"和"没检测到" | `decisions["options"]` 与 `decisions["analysis"]` 并排写进 report.json， keys 与 `plan()` 完全一致（一条门禁比两处的 key 集合） |

### 查出来两处比原描述更糟的

**1. `false` 和"没提"在序列化那一层就是同一个值。**
路由用 `task_options.model_dump(exclude_defaults=True)` 只转发客户端真正写过的键 ——
这个设计是对的，但 `TaskOptionsSchema.auto_colorize` 的默认值原本是 `False`。于是
`{"auto_colorize": false}` 与"什么都没发"序列化结果**相同**，拒绝在入库前就被丢掉。
对 `restore` 无所谓（那里默认就是不上色），对 `auto_restore` 则是：图片被判成黑白、
客户明说不要上色，结果照样上色。修法是让两个 plan switch 都默认 `None`（`face_enhance`
本来已经是），并把这条通道用测试钉住：schema 层 `test_a_refusal_survives_the_serializer_that_strips_defaults`，
API 层直接读回 `tasks.options_json`（`test_an_option_the_client_never_mentioned_is_not_stored_as_a_refusal`
—— 故意读库里那一行，因为缺陷存在于路由**发出去**的东西里，不在 parser 里）。

**2. 进度面板从来没有画过 stage 清单。**
在真机上跑 Auto 链是为了验开关，结果最后一帧只有评估文本和 decisions，`✓/▶/○` 那几行
一条都没有。根因是列形状：`get_task` 的 `stages` 在 SQLite 上是 **JSON 文本**，在
PostgreSQL 上 psycopg 已经解码成 list；面板写的是

```python
stages = row.get("stages") or []
if isinstance(stages, str):
    stages = []          # ← "是字符串就当还没有"
```

默认引擎（SQLite）下这条件永远成立，于是 §24 那份清单对用户不可见；而面板的测试全部
自备 `stages` 为 list 的行 —— 双件比生产环境更慷慨，所以整套测试一直是绿的。
`docs/architecture.md` 其实**早就写了**这条契约（"psycopg 解码、SQLite 返回文本"），
只是一直没有代码层面强制。

同一类的第二处更隐蔽：`_decode_capabilities` 把 `json.loads` 包在 `except (TypeError,
ValueError)` 里。PG 上那一列已经是 list，`json.loads(list)` 抛 `TypeError` 被吞掉，返回
**空集**；空集是任何 served 集合的子集 —— 于是"只服务上色的 worker 只领上色任务"这条
§4.3 Step 4 的承诺在 PostgreSQL 上整体失效，而且没有任何报错。这一条只有在真 PG 上
才看得见，所以补进了 live 层：`test_the_capability_filter_holds_on_the_shape_this_engine_returns`
（两条任务 priority 相同、被拒绝的那条先入库：过滤器若失效，claim 回来的就是 `restore`）。

修法是**一个解码器 + 一类门禁**：`fiximg.domain.tasks.decode_json_column` 与
`JSON_COLUMNS`（`stages`/`metrics`/`metrics_json`/`options_json`/`required_capabilities`/
`data_json`/`metadata_json`）。原来散着七处各自手写的解码 —— API 路由的
`_parse_json_field`、history 面板的 `_decode`、`Task.from_row`、`TaskEvent.from_row`、
两个 repository（`_decode_capabilities`、`model_repository._row_to_dict`）以及 worker 侧
读 options 的那段 `try/except` —— 全部改成走同一个函数，进度面板则补上原本完全没有的那
一次；`tests/unit/test_json_columns.py` 扫生产树：**任何读了这些列而没走共享解码器的模块
直接失败**，并且 `JSON_COLUMNS` 里的名字必须真的出现在 repository 源码里（防止列表烂掉
变成"永久豁免"）。

变异 M15 专门证明这道门禁有效：把面板改成用一个**自己写的、行为正确的**解码器
（`_private_decode`，能渲染、面板测试全过），门禁仍然变红 —— 它抓的是"私有解码器"这一类，
不是"这一处坏了"。

### 顺手删掉的两处死声明

| 项 | 证据 |
|---|---|
| `TaskService.plan_preview` | 全仓（src/tests/benchmark/scripts/docs）只有一个命中：它自己的 `def`。注释写着"Not used by the hot path"，实际是谁都不用 |
| `TaskOptionsSchema.to_dict` | 无调用点（唯一 `to_dict` 测试打的是 domain 的 `TaskOptions.to_dict`）。而且它用 `exclude_defaults=False`，在三态语义下会把"沉默"序列化成"拒绝" —— 一个还没人踩、但踩了就说不清的坑 |

### 证据：真机（`fixoldimg-gpu`，真 CUDA + dlib，inline worker，`examples/old/f.png`）

分析结果：`faces=1 grayscale=no scratch=1.00 blur=0.60`，所以"上色"本来就不该出现，
"脸链"本该出现 —— 两个方向都有地方落地。

| 提交 | 计划里真正跑的 stage | 用时 |
|---|---|---|
| API，不带 options | `scratch_repair, face_detection, face_enhancement, warp_back` | 18.9 s |
| API，`{"face_enhance": false, "auto_colorize": false}` | `scratch_repair` | **1.37 s** |
| API，`{"face_enhance": true}` | 与默认相同（这张图本来就检测到脸） | 10.6 s |
| UI Auto tab 全勾（True,True,False） | 全链 | 10.6 s |
| UI Auto tab 全不勾（False,False,False） | `scratch_repair` | **1.1 s** |

report 的 `planner_decisions`：默认那行是 `options={'face_enhance': True, 'auto_colorize': False}`
配 `analysis.face_count=1`；拒绝那行是 `options={'face_enhance': False, …}` 而
`analysis.face_count` **仍然是 1** —— "被拒绝"和"没检测到"从此在数据上可区分。
UI 两帧的 decisions 分别为 `face_enhancement=True` / `face_enhancement=False`。

### 本批门禁

| 门禁 | 解释器 | 结果 |
|---|---|---|
| `pytest -m "not gpu"` | 部署环境 `fixoldimg-gpu`（3.11） | 1293 passed, 20 skipped, 0 failed，exit 0（1313 collected，比上一批 +49） |
| `pytest -m "not gpu"` | dev 3.14 | 1302 passed, 11 skipped, 0 failed，exit 0 |
| `pytest tests/gpu -m gpu` | 部署环境（真 CUDA + dlib） | 30 passed, 0 skipped，exit 0 |
| `ruff check`（CI 的完整路径表） | 3.11 | All checks passed |
| `compileall`（3.11 语法） | 3.11 | exit 0 |
| `mypy` | 3.14 | Success: no issues found in 121 source files |
| `benchmark.latency --runs 2 --sizes small` | 3.14 | p50 0.0516 s / 记录 0.0514 s → x1.00，exit 0 |
| `benchmark.throughput --tasks 10` | 3.14 | 10/10 排空，queue wait p50 1202 ms · p95 1258 ms，exit 0 |
| `scripts/export_locks.py --check` | 3.11 | exit 0（未安装的组只报告不重算） |
| 真 PostgreSQL 18.6（scratch，端口 55446） | 3.11 | server 层 9 passed + fake-driver 层 8 passed，0 skipped（`-rs` 连跑两次） |
| 迁移往返（同一台真 PG，全新库） | 3.11 | `upgrade → status(0005) → downgrade base → upgrade → check` 全部 exit 0；head 处 9 张表（含 `model_versions`） |
| 变异 | 3.11 | 14 处全部变红并按字节恢复：options 三态 M1–M9、序列化通道 M10–M11、JSON 列 M12/M13/M15 |

跳过原因逐条：真 PG 层 9 条（只在设了 `FIXIMG_TEST_POSTGRES_URL` 时跑，本批已经在真服务
器上跑过）、`moto` 未装 8 条、"这台机器有可用 GPU"所以 CPU 回退用例让路 2 条、Windows 不
实施 POSIX 文件模式 1 条。第一次跑出现过 "9 skipped"，是那次调用的环境变量没进到 pytest
进程；带 `-rs` 复跑两次都是 0 skip，所以按复跑结果记录。

新增测试 49 条（collected 计数）：`test_analyzer_and_options.py` 34（原 22）、
`test_auto_restore_options.py` 7（新文件）、`test_ui_progress_eta.py` 30（原 27）、
`test_json_columns.py` 14（新文件）、`test_api_tasks.py` 13（原 10）、
`test_postgres_server.py` 9（原 8）。

### 一类值得记住的

这一批的两个"更糟"都不是功能没实现，而是**声明与形状没对上**：一个把客户端的显式值
在入库前当成默认值丢掉，一个把引擎返回的形状差异当成"没有数据"。它们的表现恰好相反
（一个多跑一级，一个什么都不显示），根因相同：**跨层传递的值没有唯一的解码/转发点，
于是每个读者自己猜形状**。`decode_json_column` + 扫生产树的门禁把"猜"变成编译期就得回答
的问题；而 Auto 开关那条链上，同一件事的做法是给两个 plan switch 定下 `None` 默认，
让"沉默"和"拒绝"在类型上就是两个值。

## 三十九、第三十七批：`priority` 终于有生产者，`strength` 被请出词汇表（plan §2.6、§6、§2.5 点 2、§3.8）

第三十六批结束时剩余清单的第一条是：**`tasks.priority` 这一列没有任何生产者**。列在、
索引 `idx_tasks_claim(status, priority DESC, created_at)` 在、claim 的
`ORDER BY priority DESC, created_at, id` 也在，`docs/architecture.md` 还把"任务之间的
先后顺序（只有 priority）"写成消费者可以依赖的性质。

| 证据 | 内容 |
|---|---|
| 生产调用点 | `submit_queued(priority=0)` 形参在，但唯一能到它的 `enqueue()` 压根没有这个参数；三个生产入口（`api/routes/tasks.py`、`ui/task_progress.py`、`ui/history_panel.py`）全部不传 → 任何进程提交的任何一行都是 0 |
| 变异判定 | 把 claim 里的 `priority DESC` 删掉，全套只剩 `test_claim_honours_priority_then_age` 一条变红，`test_claim_filter_respects_priority` **照样绿** —— 见下文"更糟的一处" |

### 做出来的事

| 项 | 决定 |
|---|---|
| 谁能设 | `POST /api/v1/tasks` 的 `priority` 表单字段，任何已认证主体；上界 `FIXIMG_PRIORITY_MAX`（默认 10）。UI 不给控件：§3.9.1 的输入区没有这一项，而"可见即可用"的规矩不允许放一个没人解释得清的旋钮 |
| 越界怎么办 | **拒绝，不夹取**。静默把 20 夹成 10 会让客户端相信自己拿到了一个从未申请的队列位置；`details` 报的是**当时真正决定的那个上限**（读 `settings.priority_max` 而不是模块常量，沿用第三十四批限流器那条教训） |
| 客户端怎么核对 | `202` 回复与 `GET /tasks/{id}` 都回 `priority`，值**从行里读回来**而不是回显请求：幂等重放返回的是先建的那条任务，也就必须先建那条任务的位置 |
| 域对象 | `claim()` 返回的 `Task.priority` 一并断言，否则"为什么是它先跑"在下游不可回答 |
| 历史面板重跑 | 继承原行的 priority，与 `POST /tasks/{id}/retry`（同一行重新入队，天然保留）保持一致：一个已经花掉队列预算的任务不该因为"再跑一次"掉回 FIFO |
| Redis 拓扑 | 无需改动即生效：`RedisQueueBackend.claim()` 最终还是走 `claim_next_task`，"stream 只是叫醒，数据库才是队列"这条既有设计把顺序也一起接住了 |

### 更糟的一处：那条"优先级"测试根本测不到优先级

变异 `ORDER BY priority DESC, created_at, id` → `ORDER BY created_at, id` 之后，
`test_claim_filter_respects_priority` 仍然绿。查出来是这台机器的时钟粒度：
`timestamps.now()` 名义上是微秒精度，但三次连续插入拿到的是**同一个**
`2026-09-28T04:10:52.628106Z`，于是 `created_at` 打平，顺序由 `id` 决定 —— 而那三条测试
用的 id 是 `t-low` / `t-high` / `t-other`，**按字母序恰好等于按优先级序**。测试是在靠
id 撞运气过的。

修法不是加断言，是把歧义从数据里拿掉：

* 顺序类测试的 id 改成"`id` 字典序与期望顺序**相反**"的形状（`a-low` / `z-high`、
  `a-first` / `z-second`），让 `ORDER BY id` 不可能伪造结果；
* 服务层那条用显式 `task_id` 走 `submit_queued`，提交顺序也刻意与期望顺序相反，
  因此 FIFO 与 id 两种误解都必然变红；
* `docs` 里那句"created_at 精度不等于 created_at 唯一"不是新写的：`test_task_repository.py`
  第三十五批已经记过这条，这次是它第一次真的影响门禁判断。

顺带把同一类的第二个手抄发现干掉：`test_sql_dialect.py::test_the_claim_statement_compiles_for_postgres`
**在测试里重写了一份 claim SQL**（含 `ORDER BY priority DESC`），所以它编译的是那句
可能被改掉的语句的副本，而不是仓库真正执行的语句。改成由 repository 暴露
`claim_statement(dialect)`（`_select_claimable` 用它），测试编译真实语句并断言
`priority DESC` 在两种方言里都在；那条原本靠 `inspect.getsource(_select_claimable)` 找
`"claim_row_lock"` 字样的"源码文本测试"也改成断言构造出来的语句 —— 断言函数怎么写而不是
它产出什么，正是这类门禁的常见失效方式。现在删掉 `priority DESC` 会让三层同时变红
（集成、路由单测、方言层），此前只有一层。

### `strength`：一个被接受、被存、被文档化、被所有人忽略的开关

`_ALLOWED_OPTIONS`、`KNOWN_OPTIONS`、`TaskOptionsSchema.strength`（`ge=0, le=2`）、
`docs/api.md` 的字段表里都写着它，`tasks.options_json` 里存着它；
`grep` 整个 `src/fiximg/inference`：**没有任何读取点**。客户端发
`{"strength": 0.3}` 会拿到 202、一行"我要求过"的记录，以及与不发完全一样的图。
变异判定：删掉那个字段只会让 `test_task_options_tolerates_bad_strength` 变红 ——
它断言的是字段自己的解析宽容度，不是任何行为。

处置：从词汇表里删掉，发送它现在得到 `INVALID_OPTIONS`（而不是被忽略）。删除是
非破坏性的，因为 `TaskOptions.from_dict` 本来就把不认识的键留在 `extra` 里、
`to_dict()` 原样写回，旧行仍然完整可序列化（新增一条测试专门钉这一点）。
同批删掉 `TaskOptions.planner_switches()`：无调用点，而且它把 `hr` 说成 planner 开关
（planner 从不读它）—— 第二份"同一份词汇"一旦存在就会这样漂走。

### 把这一类做成门禁（`tests/unit/test_option_vocabulary.py`）

1. **一份词汇表**：`_ALLOWED_OPTIONS`（API）、`KNOWN_OPTIONS`（domain）、
   `TaskOptionsSchema.model_fields` 三者必须相等（domain 不能 import API，所以只能这样钉）；
2. **每个声明的开关都得被"跑东西的那层"读**：planner 声明的开关用
   `options.get("key")` 形态在 `src/fiximg/inference/**` 里找读取点，`hr` 由 stage/backend
   读；只有声明没有读取点就失败（`strength` 当年就是这条会拦下）；
3. **反方向**：`context.options` 里被读到、API 却不接受的键也失败（内部键
   `INTERNAL_OPTION_KEYS` 豁免；backend 层的 `request.options` 是 stage kwargs + 设备
   合并出来的袋子，不能要求它等于客户端词汇表，所以那一侧只查方向不查全等）；
4. **声明必须可观测**：把每个 `PLAN_SWITCHES` 成员在两条计划构建器上 true/false 对切，
   stage 列表必须不同 —— 只看"有没有读取点"分不出"`hr` 被读"与"`hr` 影响计划"，
   而 UI 的 decline-only 映射是按"影响计划"写的；
5. **扫描本身不能是空的**：`hr` 至少两个读取点、每个 plan switch 都有读取点、
   声明集合 ≥3。

另外补了一条同类的配置门禁（`tests/unit/test_settings.py`）：**deployment.md 里写出来的
每一个 `FIXIMG_*` 名字必须在源码里被读**。豁免有两条且都从代码推导，不手写：运行时按
前缀拼出来的名字（`FIXIMG_CONCURRENCY_<CAPABILITY>`，前缀由 scheduler 以字面量声明）和
`FIXIMG_TEST_*`（测试层用的）。这条正是第三十五批那处"文档写了没人读"的类固化：
门禁还要求前缀豁免**真的救下某个名字**，否则它就是一条没人需要的永久豁免。

最后一条是 multipart 的形状：create 端点的字段被声明了两次（端点参数 +
`TaskCreateForm`，因为 multipart 不能是 Pydantic body，只能通过 `openapi_extra` 注入文档）。
新加的 `priority` 恰好是"只改一边"的典型字段，所以测试现在用
`inspect.signature` 抓出端点自己的字段集合，与文档模型、与 OpenAPI 里的
`multipart/form-data` schema 三方对等。

### 证据：真机（`fixoldimg-gpu`，真 CUDA + dlib，同一进程先提交后启 worker）

三次 `POST /api/v1/tasks`（同一张 `examples/old/f.png`，`restore` 全链）：
`ordinary-first`(0) → `urgent-second`(9) → `ordinary-third`(0)，都停在 `queued`
（`FIXIMG_INLINE_WORKER=false`），然后才启一个 worker：

```text
started-event 序号: {'ordinary-first': 15, 'urgent-second': 4, 'ordinary-third': 26}
实际领取顺序      : ['urgent-second', 'ordinary-first', 'ordinary-third']
用时              : urgent 13.7 s(含模型冷启动) / ordinary 9.27 s / 9.15 s
```

8 条检查全 PASS，连跑两次 exit 0、顺序一致。这里判断依据是 `system_events` 的自增 id，
不是 `created_at`：上面已经说过，同一时钟刻度里的三次插入共享同一个 instant，
用 `created_at` 判"谁先开始"会测到 id 撞运气。另外两次越界提交（`priority=999`）返回
`400 INVALID_REQUEST`，`details={"priority":999,"min":0,"max":10}`。

### 本批门禁

| 门禁 | 解释器 | 结果 |
|---|---|---|
| `pytest -m "not gpu"` | 部署环境 `fixoldimg-gpu`（3.11） | 1314 passed, 20 skipped, 0 failed，exit 0（1334 collected，比上一批 +21） |
| `pytest -m "not gpu"` | dev 3.14 | 1323 passed, 11 skipped, 0 failed，exit 0 |
| `pytest tests/gpu -m gpu` | 部署环境（真 CUDA + dlib） | 30 passed, 0 skipped，76.7 s，exit 0 |
| `ruff check`（CI 的完整路径表） | 3.11 | All checks passed |
| `compileall`（3.11 语法） | 3.11 | exit 0 |
| `mypy` | 3.14 | Success: no issues found in 121 source files |
| `benchmark.latency --runs 2 --sizes small` | 3.14 | p50 0.0512 s / 记录 0.0514 s → x1.00，exit 0 |
| `benchmark.throughput --tasks 10` | 3.14 | 10/10 排空；queue wait p50 1077 ms · p95 1131 ms · avg 1081 ms，exit 0 |
| `scripts/export_locks.py --check` | 3.11 | exit 0 |
| 真 PostgreSQL 18.6（scratch，端口 55448，两个全新库） | 3.11 | server 层 9 passed + fake-driver 层 8 passed，`-rs` 报 0 skip；priority 那条新测试（解码形状的能力过滤）在真服务器上跑过 |
| 迁移往返（同一服务器、全新库） | 3.11 | `upgrade → status → downgrade base → upgrade → check` 五步全 exit 0，head=0005，head 处 9 张表 |
| 真机队列顺序 | 3.11 | **两种引擎各跑一遍**：SQLite 两次 + PostgreSQL 一次，8 条检查全 PASS、exit 0；领取顺序恒为 `urgent-second → ordinary-first → ordinary-third` |
| 变异 | 3.11 | 13 处（N1–N13）全部变红并按字节恢复；其中 N7（删掉 `priority DESC`）现在让**三层**同时变红，此前只有一层 |

跳过原因逐条：真 PG 层 9 条（只在设了 `FIXIMG_TEST_POSTGRES_URL` 时跑 —— 本批已在真服务器
上单独跑过）、`moto` 未装 8 条、"这台机器有可用 GPU"所以 CPU 回退用例让路 2 条、Windows 不
实施 POSIX 文件模式 1 条。

新增测试 21 条（collected 1313 → 1334）：`test_option_vocabulary.py` 7（新文件）、
`test_api_tasks.py` 13 → 23、`test_settings.py` 5 → 7、`test_queue_reliability.py` +1、
`test_postgres_server.py` 8 → 9；另有 3 个文件里的 4 条测试被**重写**（两条顺序测试改成
歧义自由的数据形状、方言层两条改成断言真实语句）。

### 一类值得记住的

这一批的三处问题都不是"功能没实现"，而是**声明的强度超过实现的强度**：一列被索引、
被排序、被文档化成消费者可依赖的性质，却没有任何入口能写它；一个开关被 schema、白名单、
领域模型和三处文档共同担保，却没有任何运行时读它；两条测试断言的是语句的**副本**而不是
语句本身。它们不会表现为失败，只会表现为"通过"。所以修法都是把声明与实现绑到同一个
来源上：表单字段由端点签名推导、开关语义由 planner 的声明与一次真实的对切推导、
claim 语句由仓库自己暴露、文档里的旋钮由扫描推导。

## 四十、第三十八批：Auto tab 的输入区终于有"自动分析"这一行（plan §3.9.1、§2.11、§3.9.2）

第三十七批之后，needs-code 的第一条是 §3.9.1 输入区三件套里还缺的那一件：**自动分析**。
方案的布局图把它画在图片与勾选框之间：

```text
│  [Upload Image]   │   [Before / After]      │
│  Auto Analysis     │   [Download] [Retry]    │
│  □ Face Enhance   │                          │
```

| 证据 | 内容 |
|---|---|
| 生产调用点 | `analyze_image` 全仓只有一个生产调用点：`runtime._execute_core`（`src/fiximg/inference/runtime.py:476`），也就是**任务已经在跑**之后。`format_analysis_summary` 的唯一读者是同一行旁边的日志字段 |
| 用户其实也看不到 | 报告里 `planner_decisions.analysis` 带着这组数字，但进度面板完成帧拼 decisions 时过滤掉了 dict 值（`if not isinstance(v, dict)`），所以那串数字提交前看不到、提交后也看不到 |
| 变异判定 | 没有任何测试断言"提交前存在一份分析"，也没有任何测试断言 caption 里那句"only when the analysis finds a face"有对应的可见依据 —— 这一整块缺失不可能让任何测试变红 |

### 决定表

| 问题 | 决定 |
|---|---|
| 谁来算 | `TaskService.auto_preview(image, options)` —— UI 不能 import `fiximg.inference`（分层门禁 `test_architecture_boundaries` 就写着），所以预览必须走应用层；顺带这也让预览与提交用的是同两个调用 |
| 显示什么 | `format_analysis_summary` 的原样文本 + 将要跑的 stage 链 + 只在真的发生时才出现的 forced/declined 说明。第三行是上一批开关工作的闭环：caption 说"只在分析找到脸时"，这一行给出它到底找没找到 |
| 什么时候重算 | 换图**和**改任一开关都重算（4 个 trigger：image + 3 checkbox）。开关是答案的一部分，漏接就等于给用户看上一刻的计划 |
| stage 名字从哪来 | 从**构建出来的 stage 实例**取 `stage.name`，不是从计划的注册键取：`global_restore(with_scratch=True)` 注册键是 `global_restore`，它在任务行里记的名字是 `scratch_repair`。预览若印注册键，每张有划痕的图都会跟历史面板对不上 |
| 哪些 tab 有这一行 | 只有 Auto。恢复类 tab 的阶段由任务类型决定，摆一串测量值在那儿描述不了任何决定 —— 这正是本项目一直在删的装饰 |
| 失败怎么办 | 分析器对陌生人上传的字节报错时，行里写 `Analysis unavailable: <类型>: <原因>`，而不是把异常抛给浏览器；提交按钮不受影响 |
| 是否加开关 | 不加。它只是既有路径上的两次函数调用，做一个 `FIXIMG_*` 开关只会多出一个"关掉之后提交时又照样算一遍"的第二配置面 |

### 顺带抓到的一类：三个门禁各自猜测"哪些依赖是提交回调"

接线之后，两套 UI 布局门禁立刻变红（`test_every_submit_handler_declares_matching_outputs`、
`test_every_result_tab_offers_a_download`）：它们都用
`str(dep["api_name"]).startswith("handler")` 挑提交回调，而 Gradio 的 `api_name` 是
**所有**回调共用一个计数器 —— 我新加的 4 个分析行回调拿到了 `handler_1…handler_4`，于是
"5 个 tab"数成了 9 个。第三处（第三十五批写的 `test_ui_progress_eta`）为同一件事写了另一
份按 `api_name` 前缀的挑选逻辑。

修法不是在第三处再补一个形状判断，而是把选择器收进 `tests/fixtures`：
`submit_deps(cfg)` 按**形状**挑（第一个入参是 image、第二个是 state、第一个出参是
result image），外加 `deps_writing_to(cfg, label)` / `components_by_id(cfg)` /
`demo_config()`。三个门禁现在共用它，而 `api_name` 那种"看起来稳定其实由计数器顺序
决定"的挑法在这一层从此消失。为什么不用"含 state 且出参含 image"这种宽松形状：历史
面板的行处理器也满足，会误收 —— 这条写在 `submit_deps` 的注释里。

### 证据：真机（`fixoldimg-gpu`，真 CUDA + dlib，inline worker，`examples/old_w_scratch/a.png` 496×624）

四种开关组合各跑一次"预览 + 提交"，22 条检查全 PASS、exit 0：

| 勾选 | 预览给出的链 | 任务行实际记录的链 | 用时 |
|---|---|---|---|
| 全勾 | `scratch_repair → face_detection → face_enhancement → warp_back` | 同左（逐个字符串相等） | 16.4 s |
| 关脸链 | `scratch_repair` | 同左 | 1.4 s |
| 关上色 | `scratch_repair → face_detection → face_enhancement → warp_back` | 同左 | 9.8 s |
| 全关 | `scratch_repair` | 同左 | 1.3 s |

* 预览第一行与报告里 `planner_decisions.analysis` 经同一 formatter 得到的文本**逐字符相等**：
  `Analysis: grayscale=no scratch=0.79 blur=0.00 faces=1 size=496x624`
* 预览代价：冷 29.7 ms、热 7.0–7.3 ms（首次含人脸后端解析+加载 12 ms，后端 `yunet`，
  ONNX 232 KB）。没有 GPU 张量、没有 CUDA 上下文、不碰恢复权重
* 这张图不是灰度图，所以"关上色"看不出差别；于是另外用 API 式选项跑了一次强制方向：
  `{"auto_colorize": true, "face_enhance": true}` →
  `Planned pipeline: scratch_repair → face_detection → face_enhancement → warp_back → colorization`
  加一行 `Forced by your switches, not by the image: colorization`（上色仍在最后，
  不会把没上色的脸贴回原图）

### 本批门禁

| 门禁 | 解释器 | 结果 |
|---|---|---|
| `pytest -m "not gpu"` | 部署环境 `fixoldimg-gpu`（3.11） | 1331 passed, 20 skipped, 0 failed，exit 0（1351 collected，比上一批 +17） |
| `pytest -m "not gpu"` | dev 3.14 | 1340 passed, 11 skipped, 0 failed，exit 0 |
| `pytest tests/gpu -m gpu` | 部署环境（真 CUDA + dlib） | 30 passed, 0 skipped，68.2 s，exit 0 |
| `ruff check`（CI 的完整路径表） | 3.11 | All checks passed |
| `compileall`（3.11 语法） | 3.11 | exit 0 |
| `mypy` | 3.14 | Success: no issues found in 121 source files |
| `benchmark.latency --runs 2 --sizes small` | 3.14 | p50 0.0510 s / 记录 0.0514 s → x0.99，exit 0 |
| `benchmark.throughput --tasks 10` | 3.14 | 10/10 排空；queue wait p50 1028 ms · p95 1083 ms · avg 1034 ms，exit 0 |
| `scripts/export_locks.py --check` | 3.11 | exit 0 |
| 真 PostgreSQL 18.6（scratch 端口 55450，全新库） | 3.11 | server 层 9 passed + fake-driver 层 8 passed，`-rs` 报 0 skip |
| 迁移往返（同服务器、另一座全新库） | 3.11 | `upgrade → status → downgrade base → upgrade → check` 五步全 exit 0，head=0005 |
| 真机 UI 预览 | 3.11 | 4 组开关各"预览 + 提交"一趟，22 条检查全 PASS、exit 0 |
| 变异 | 3.11 | 8 处（P1–P8）全部在**拥有该性质的那一层**变红并按字节恢复；两处（P2 命名、P7 只在 Auto 有行）在另一层看不到，原因写进了对应测试的注释里 |

跳过原因逐条：真 PG 层 9 条（只在设了 `FIXIMG_TEST_POSTGRES_URL` 时跑，本批已在真服务器
上单独跑过）、`moto` 未装 8 条、"这台机器有可用 GPU"让路 2 条、Windows 不实施 POSIX 文件
模式 1 条。

新增测试 17 条（collected 1334 → 1351）：`test_analysis_preview.py` 15（新文件）、
`test_ui_progress_eta.py` 30 → 31（布局门禁 +1）、`test_auto_restore_options.py` 7 → 8
（预览与执行对账）。另有 3 处门禁改成用共享选择器读布局（不是新增断言，是换掉一份
会随回调数量漂移的猜测）。

### 一条经验

这一批的东西全是"看得见"层面的，但它抓出的两个问题都在最底层：一个是**命名的两个视图**
（注册键 vs stage 自报名字），一个是**测试用错了标识**（按自增计数器给回调分类）。
两者的共同点是：符号在同一系统里有两个名字，而每处读者各挑一个 —— 于是"看起来一致"
一直成立，直到有人新增一类回调或走一次划痕分支。所以修的方式也一样：把它收到一个
来源（`tests/fixtures` 的选择器、`_stage_labels()` 的名称解析），让第二份猜测没有地方
可以活下来。

## 四十二、第三十九批：把"声明了但没人用"做成一类门禁（plan §2.5、§2.7、§3.5.4、§3.11）

前三十八批都是按报告条目推进的。这一批换了个入口：先让三个只读子代理按
"生产调用点 + 删掉这行有没有测试会红"两条证据去扫方案，然后把每条候选自己复核一遍。
子代理的通道这一轮不通（连续三次 `Step interrupted`），于是改成自己扫——扫的是一个
772 个公开名字的树，工具写在 `.gates/audit_classify.py`。

**这一批的产出不是"又补了几个缺口"，而是一类门禁**：全部 27 条候选里有 10 条是路由
处理器（按装饰器推导，豁免），1 条是兼容别名，剩下的逐条处理。而其中**三条是"注释写了
它被谁读，它其实没人读"**——写注释这个动作本身在制造假证据。

### 我自己造成的一处回归，值得先记下来

处理 `analyzer.py` 的死阈值时我用了一条 PowerShell 批量重写：

```powershell
Get-ChildItem tests\**\*.py -Recurse | %{ ... [System.IO.File]::WriteAllText(...) }
```

`Get-Content` 在这台机器上按 ANSI（GBK）解码，写回时却是 UTF-8——于是**所有**测试文件
的双字节字符被转成了 cp936 的对应物。GBK 是双字节编码，所以 3 字节的 UTF-8 序列通常会
产生一个**合法**的 GBK 汉字加一个落单的字节：文件仍然可解析，没有 U+FFFD，两次
"替换字符扫描"都是干净的。第一个信号是 3 个文件 `SyntaxError`，真正的信号是
**11 个用例消失了**（`diff_junit.py` 对比两次 junit 才看出来）。

修的过程本身又暴露了两次同一类错误，值得写下来：

| 现象 | 原因 |
|---|---|
| `U+FFFD` 扫描报 0 处，套件还是红 | GBK 双字节：损坏产生的是合法汉字 + 落单 `?`，不是替换字符。判据要改成"这个字符是不是某个真实字形的 cp936 误读"（`find_mojibake.py`） |
| 8 个文件"修好了"之后仍然红 | 通用修复把 U+2713 当成分隔符填进了状态字形的位置。分隔符和状态字形是两个字符，且都要跟 `describe_preview` / `_stage_lines` 的真实输出**比对**得出，不能推断（`fix_arrow2.py` 取箭头、`fix_spaces.py` 补被吞掉的空格） |
| `git checkout --` 一个文件，丢 11 个用例 | 那 5 个文件在索引里是 **V2 版本**（`from app.services import ...`），工作区里是 V3 版本。checkout 取回的是旧版。11 条用例里包含 SSIM 对照、face_count 冲突、auto_restore 指标、skipped 状态——正是历史上抓出真实缺陷的那些。全部按生产契约重写，并加 `diff_junit.py` 作为常驻检查 |

跑回真机时又发现第二个后果：`run.py` 也被 checkout 成了 V2 的自包含四路径实现，
**不认 `--stages`**，于是 `LegacyCliBackend` 每一次 spawn 都死在
`unrecognized arguments: --stages 1`。**整套 CPU 测试是绿的**——只有
`pytest tests/gpu -m gpu` 会真的 spawn 它。这条补成了 `tests/unit/test_run_py_entry.py`。

### 复核后成立的缺口（按处置分类）

| # | 项 | 复核到的事实 | 处置 |
|---|---|---|---|
| 1 | `analyzer.py` 四个阈值常量 | `GRAYSCALE_SAT_THRESHOLD` / `SHARP_LAPLACIAN_VARIANCE` / `SCRATCH_STRUCTURE_THRESHOLD` / `SCRATCH_FRACTION_FULL` 全是 `settings.*` 的副本，全仓零引用；`estimate_scratch` 的 docstring 还把其中一个写成"实际使用的归一化常数" | 删掉，docstring 指向 settings。**一个校准 pass 改这些常量不会有任何效果，而且日志里不会有任何一行** |
| 2 | `paths.py` 的 `MODELS_DIR` | 零引用，而 `manifest.py` 硬编码 `os.path.join(PROJECT_ROOT, "models", …)` 五处——正是那个 docstring 写着"所有路径相关的模块都必须从这里导入"的模块 | 改为 `models_root()` 函数。**为什么必须是函数**：`test_error_code_and_versions.py` 用 `monkeypatch.setattr(manifest, "PROJECT_ROOT", tmp)` 搬根目录，而 `MODELS_DIR` 是 import 期捕获的，于是**断言两边都从同一个过期根算出答案，套件保持绿** |
| 3 | `passwords.needs_rehash()` | 有 3 个测试、零生产调用。抬高 `DEFAULT_ROUNDS` 永远不会升级既有用户的口令 | 接进 `authenticate()`：登录成功且参数偏旧时重算一次哈希（失败只记 WARNING，不影响这次登录） |
| 4 | `passwords.is_plaintext()` / `write_yaml_atomic()` | 零引用零测试，而模块 docstring 宣称"Writes use a temp-file + os.replace atomic commit"——**全仓 `src/` 没有任何 `yaml.dump`**。连同 `filelock` / `tempfile` / `_WRITE_LOCK` 一起删，docstring 改成实话 |
| 5 | `engine.append_history()` | 唯一调用方是一个测试。它是 `history` 表**唯一**的写入方，所以 V2/V3 永远写不进一行，admin 的 History 标签页在一台全新安装上恒为空 | 删掉，注释说明该表现在是只读的 V1 遗留（`migrate_v1` 从它读出，admin 面板读它） |
| 6 | `engine._enforce_history_cap()` | 删掉 `append_history` 之后它**被我孤儿化**了。连带 `FIXIMG_HISTORY_MAX` / `history_max` 在给一张不会增长的表配限额 | 一并删。`HISTORY_MAX` 作为 `list_history` 的默认行数上限保留——那是读侧限制，仍然有效 |
| 7 | `task_repository.get_metrics()` | 有 5 个测试、零生产调用。生产读 metrics 走 `get_task`/`list_tasks` 的 SQL 聚合，**这是同一套"折叠 metric_value/metric_text"规则的第二份实现** | 改名 `read_metrics`（诚实：它不是生产路径），并补 `test_metric_readers_agree.py`：两套读法 + `list_tasks` 的第三份拷贝，对含 `inf`（REAL 装不下、JSON 发不出）的用例必须逐字相等。**先量过再写断言**：SQLite 与真 PostgreSQL 上都相等，所以这不是缺陷，但"没有测试比对两份实现"是 |
| 8 | `task_repository.list_events()` | 有 3 个测试、零生产调用。`system_events` 里**不带 `task_id` 的行无法通过任何 API 读到**——提交被拒（队列满，没有任务行）、worker 起停、任务行从未创建 | 接进新端点 `GET /api/v1/stats/events`（`RecentEventsResponse` / `RecentEventView`），6 条测试 + 2 处变异。回答的是"我提交的任务不在队列里，为什么" |
| 9 | `gradio_app.ADMIN_ONLY_TABS` | 零引用。`ui/state.py::tab_visibility` 才是唯一规则 | 保留为**断言夹具**并写明它不是规则；测试把两个集合通过 demo 真实构建的 id→elem_id 映射对起来 |
| 10 | `make_limiter()` | 零生产调用，且它直接构造 `SharedSlidingWindowLimiter`，**绕过 `_make_limiter` 的 Redis 判定**——多副本部署下每个副本各算一份配额 | 保持"永远返回 shared 类"（该类每次调用才解析 backend），并补一条测试断言它在配了 `FIXIMG_REDIS_URL` 时落到的仍是 shared |
| 11 | `api/security.reset_cache` 的 docstring | 写着"used by tests and by settings overrides"——**生产侧零调用** | 改成实话：只有测试用，并说明为什么（`db_path` 是进程级设置） |
| 12 | `scheduler.from_env()` 重复读 `FIXIMG_CONCURRENCY` | `config.py:341` 已经解析过这个变量（env → YAML → 校验），scheduler 又读了一遍，于是 `configs/*.yaml` 的 `concurrency_default` **看起来可配、实际无效** | 改为读 `settings.concurrency_default`；补一条"改 setting 而不是改 env"的测试 |
| 13 | `engine.ARCHIVE_INPUT_DIR` / `ARCHIVE_OUTPUT_DIR` / `ARCHIVE_TTL_SECONDS` | 与 `settings.archive_input_dir` / `archive_output_dir` / `archive_ttl` **同一组东西的第二份声明**，而只有 engine 那一对被读 | engine 改为经 `_settings()` 读取（`archive_input_dir()` / `archive_output_dir()` / `purge_stale_archives(ttl=None)`） |
| 14 | `paths.STORAGE_ROOT` | 零引用（只在自己的 docstring 里出现过），与 `settings.storage_root` 重复 | 删掉，并在 `paths.py` 的 docstring 里写清**为什么**可写根归 `config`（它们是部署配置，且相对 `base_dir` 解析，两个不同工作目录的进程必须看到同一批字节） |
| 15 | `FIXIMG_COLORIZE_TTL` / `settings.colorize_result_ttl` | 零引用，连 YAML 里都没有 | 删。`result_ttl` 已经管所有 run 目录；要一个"上色专用 TTL"就必须在清扫器里再查第二个地方 |

### 门禁本身

新增 `tests/unit/test_declared_but_unwired.py`，两条扫描：

1. **`src/` 里每个公开模块级名字必须在生产代码里被引用过**（含自己所在文件，因为
   私有但活着的名字不是这一类；判据是"出现次数 > 1"，即声明之外还有引用）。
   豁免三类，全部推导而非手写：FastAPI 路由处理器（按 `@router.<verb>` 装饰器）、
   框架类（`BaseModel` / `Protocol` / `BaseSettings`，由框架按 `response_model` /
   `isinstance` 触达）、以及 6 条写明理由的 `TEST_ONLY`。
2. **`Settings` 的每个字段必须在 `config.py` 之外被读过。** `FIXIMG_*` 名字早就有这条
   门禁（`test_settings.py`），`Settings` 属性没有——而 YAML profile 绑的正是后者。

第二条是被**第一次没咬住的变异**逼出来的：

```
GREEN  a settings field detached from its only reader
```

把 `settings.auto_grayscale_sat` 换成字面量 `16.0`，第一条扫描不响——因为那个名字在
`config.py` 里**还声明着**。字段从此无人读，而 `configs/base.yaml` 照样列着它，运维调它
不会有任何效果，也不会有任何提示。这条"检测器第一版自己没咬住"已经写进注释里。

门禁的四条变异验证（各自变红、按字节恢复）：

| 变异 | 结果 |
|---|---|
| 加一个没人读的新常量 | RED |
| 把阈值读取与它的 settings 来源脱钩 | RED |
| 塞回一个死函数（`write_yaml_atomic` 的形状） | RED |
| 把一条豁免指向一个已不存在的名字 | RED |

第 4 条是豁免本身的门禁：豁免是对代码的**断言**，过期豁免就是一条永久的静默通过。

### 行为类修复的变异验证

`.gates/mutations_b39.py`：**16 处全部变红、对照全绿**。其中三处是**第一版没咬住、
第二版才咬住的**：

| 逃过的变异 | 为什么第一版是绿的 | 补强 |
|---|---|---|
| 身份指标改写脸链的 `face_count` | 测试用 `add_metric` 间谍观测，而**三个计数根本不进 metrics 表**，只进 `_evaluate` 返回的 dict | 夹具多返回一个 view，断言在返回值上 |
| `ADMIN_ONLY_TABS` 与 `tab_visibility` 漂移 | 测试分别断言"常量 ⊆ 已构建的标签"和"`tab_visibility` 给 `admin_panel`"——**两个都对，而两者之间没有任何约束** | 通过 demo 真实构建的 id→elem_id 映射把两个集合对起来 |
| 提交回调不再校验开关数量 | 变异只替换了 `make_preview_handler` 里那一处（`str.replace(old, new, 1)`），提交回调的守卫没动 | 锚点带上注释行，定位到 `make_submit_handler` |

**"两个断言各自为真、两者之间没有约束"是这一批最值得记住的一条**：它们都比"断言函数
怎么写"强，而合起来仍然放过了缺陷。

### 一次自己造成、被自己门禁抓住的孤儿化

删掉 `append_history` 之后，`_enforce_history_cap` 失去了唯一调用方——**整套仍然绿**。
这是"删除一个只被测试调用的生产函数"的必然结果：它同时带走了一个私有辅助函数的调用。
新门禁只看公开名字，所以没抓到；抓它的是后来那条 `Settings` 扫描（`FIXIMG_HISTORY_MAX`
和 `history_max` 一起变成了没人读的配置）。**两条门禁互相补位，而不是一条万能的。**

### 本批门禁

| 门禁 | 环境 | 结果 |
|---|---|---|
| `pytest -m "not gpu"` | 部署解释器 `fixoldimg-gpu`（3.11 / torch 2.7.1+cu128 / dlib 20.0.1） | **1387 passed, 20 skipped**，exit 0 |
| `pytest -m "not gpu"` | dev 3.14 | exit 0 |
| `pytest tests/gpu -m gpu` | 部署解释器（真 CUDA + dlib） | **30 passed, 0 skipped**，exit 0 |
| `pytest tests/integration/test_postgres_server.py` | 真 PostgreSQL 18.6（127.0.0.1:55462） | 9 passed，`-rs` 零 skip |
| 迁移往返（真 PG，全新库） | 3.11 | `upgrade → status(0005) → 9 表 → downgrade base → upgrade → check` 五步全 exit 0 |
| `scripts/smoke_services.py`（真 PG） | 3.11 | database PASS / queue PASS / redis·S3 SKIP（未配置） |
| `ruff check`（CI 完整路径表） | 3.11 + 3.14 | All checks passed |
| `compileall`（3.11 语法） | 3.11 | exit 0 |
| `mypy`（`check_untyped_defs`） | 3.14 | **0 发现 / 121 文件** |
| `benchmark.latency --runs 2 --sizes small` | 3.11 | p50 0.0517 s 对基线 0.0514 s → x1.01，exit 0 |
| `scripts/export_locks.py --check` | 3.11 | exit 0 |
| 变异 | 3.11 | 16 处全部变红、按字节恢复；对照全绿 |

跳过 20 条：真 PG 层 9 条（只在设了 `FIXIMG_TEST_POSTGRES_URL` 时跑——**本批已在真
服务器上单独跑过 9 条全过**）、`moto` 未装 8 条、"这台机器有可用 GPU"让路 2 条、Windows
不实施 POSIX 文件模式 1 条。

新增测试 **36 条**：`test_declared_but_unwired.py`(14)、
`test_metric_readers_agree.py`(3)、`test_run_py_entry.py`(6)、
`test_stats_events.py`(6)、`test_metrics.py`（重写，+11）、
`test_identity.py`(+2)、`test_task_progress.py`(+3)、`test_ui_roles_and_progress.py`(+1)、
`test_rate_limit_backend.py`(+1)、`test_scheduler.py`(+1)。

**用例数从 1351 涨到 1387，而 `diff_junit.py` 对比两次 junit：missing 0** —— 这一点是
本批最需要的检查，因为我确实弄丢过 11 条。

### 一类值得记住的

这一批的九个真实缺陷里，有**四个的注释都声称它被谁读了**：
`estimate_scratch` 说"按 SCRATCH_FRACTION_FULL 归一化"（实际用 settings）、
`passwords` 的模块 docstring 说有原子写（实际没有写入方）、
`ADMIN_ONLY_TABS` 的注释说"visibility 按它检查"（实际只有我新加的测试检查）、
`reset_cache` 说"测试和设置覆盖都用它"（实际只有测试）。

**写注释是一个会产生假证据的动作**：它让下一个读者（包括我）不再去查。而这一类缺陷
（声明在、名字在、没人调）永远不会表现为失败，只会表现为"通过"。所以这一批的产出
是一条会在**下一次**同类缺陷出现时就变红的门禁，而不是又一份散文——散文正是它自己
抓不到的那 eleven 个用例里六个的藏身之处。

## 四十一、结论

报告 P0（5 项）、P1（6 项）全部完成；P2 中 Redis / PostgreSQL / MinIO / 多 GPU /
模型热更新的**代码**已完成，验收标准 37 项除"真实外部服务上的运行验证"外均已满足。
第二十九批按同一证据标准重核时，这句话曾要收紧两处（#16 上色链的显存记到没跑过的卡上、
#4 只有 global_restore 真走 `.infer()`）；第三十批把这两处做实了：**#16 与 #4 现在成立**，
判据是真实 GPU 上的一次 `colorize` 任务——stage、任务级与 `/stats` 三处给出同一个
`gpu_memory_peak_mb=1768.0`，而上色调用的是 `DDColorBackend.infer()`。
剩下的缺口只是环境：#37 的多卡路由、真实 Redis 与真实 S3 端点仍需那种机器。

推理 runtime 这条主线（§9 认定的核心）到第十二轮的落点：

| 链 | 状态 |
|---|---|
| DDColor 上色 | 常驻进程内（V2 起） |
| Global 质量修复 | **常驻 + 逐像素等价已证明** |
| Scratch 修复（stage 1 另一分支） | **常驻 + 图像与 mask 等价已证明**（对比参照流水线自身的噪声地板；参照位稳定时要求逐位一致。与 quality 同树，一个 global worker 常驻两条分支） |
| Face Enhancement（stage 3） | **常驻 + 逐字节等价已证明**（在 face worker 上启用） |
| Face Detection（stage 2） | **常驻 + 逐字节等价已证明**（在装 dlib 20.0.1 的真环境上跑通对拍，因此 `FIXIMG_FACE_DETECT_NATIVE` 默认开启） |

至此**四条 legacy 子进程链全部有原生实现，且四条的等价测试都在能跑它们的环境里跑过**
（face detection 那台"能装 dlib 的机器"就是本机，第二十七批证明）。
唯一没转原生的是 `--HR` 分支：它要 `mapping_Patch_Attention`，上游根本没提供这个
权重，所以原生实现明确拒绝 HR 而不是静默跑随机权重。

"一棵树一个进程"这条约束没有被绕过、也没有硬凑：它被写成 `FIXIMG_NATIVE_TREE`
的部署配置，测试与文档都按它来（face enhancement 的等价测试把原生侧放进子进程）。

数据层（§2.7）的三条症状——时间当字符串、metrics 当字符串、统计拉回 Python 算——
全部落地，并且带一支可逆的 Alembic 修订与一份"库落后于代码"的自检命令。
工程化那条"type check"现在才是它字面上的意思：mypy 零发现、`check_untyped_defs`
打开、CI 去掉 `continue-on-error`。

附录 B 的 37 项按"生产调用点 + 测试"两条证据重审过一遍，抓出的全是
**"接口在、没人调用"** 这一类：Redis 投递、产物发布与回收（§2.8）、TTL 扫描、基准
门禁、UI 分层。逐条补完后，37 项里不再有"看着满足、其实没人走"的那一类；三条分层
规则也从散文变成静态门禁。重审前文档里有两处是**先写后做**的（"CI 门禁"跑不了红、
"queue wait 基线"根本没测），现在名副其实。

剩余项三类：

1. **仍缺外部服务端**：真实 Redis 与 MinIO/S3 端点（协议层测试齐了，`fakeredis` /
   `moto` 都不开 socket；本机也找不到 `redis-server` 二进制）。
   原本列在这里的三项**已经不再是缺口**：PostgreSQL 现在有真服务器验证层
   （`tests/integration/test_postgres_server.py` + CI `postgres` service job，
   见第二十七批），GPU 真模型用例与 dlib face 检测对拍都在 `fixoldimg-gpu` 里跑通。
   `make smoke-services` / `make db-check` 仍是部署时的自检入口。
2. **有前提的性能项**（§3.5.4 的 batch / pinned memory / CUDA stream）：不是没做，
   是**当前没有可以安全接入的位置**——四条原生链的价值全在"与 vendored 序列逐字节
   一致"，而 batching 恰恰要重写那段序列。前提见第十四节。
3. **需要新代码能力才能做的**：把某条链改成自己掌控张量路径的原生实现（那时
   batching / pinning / stream 才有真正的调用点），这属于新特性而非补齐。

方法论收获（七轮里反复出现）：**"测试变绿"与"被测的东西真在跑"是两件事**。先后碰上
sha256 校验全部返回 True 却什么都没查、tracing seam 没有任何生产调用点、golden 基线
只来自桩、`install_tracer()` 对已 import 的句柄无效、约 700 个用例静默共用开发库、
一例依赖导入顺序的假绿测试、**迁移测试因本机没装 alembic 而整体 skip**（装上第一次
真跑就抓到 fixture 只清了一半连接缓存）、**mypy 挂着 `continue-on-error` 而无人读它
的输出**、`pip-audit` 那一步末尾的 `|| true`，以及 **`importorskip("moto")` 让
一条 S3 测试从来没跑过**——装上 moto 后它第一次运行即失败，因为它断言的是手写 fake
才有的 `.objects` 字典。它们都不表现为失败，只表现为"通过"。

第十七批是另一类，前六轮的方法照不出来：**跑了、算了、算错了**。SSIM 抄错一项、两个
测量共用一个键、一个模型名被另一个盖掉、`skipped`  enum 没人写、等价测试把抽样运气
当阈值——每一条都带着完整的 HTTP 200 和 `completed`。它们只能被"独立实现互相对照"
（ skimage 对 SSIM）、"同一份数据自不自洽"（报告与 metrics 互相矛盾）和"同一件事再跑
一遍会不会变"（双峰参照）抓出来。所以补齐的最后一批不是再加一条门禁，而是把服务
启起来，把它报的数一个一个拿别的办法算一遍。

第十八批是这类里的第二类：单个数字对、单块报告对，但**块与块之间口径不一致**。
`auto_restore` 的身份指标承认它是修复，差值指标不承认；降级原因的循环列了三个候选阶段，
却只有"谁排最后"起作用。这类缺陷任何一块单独看都自洽，只能横向比对同一份数据里的两处
判断抓出来——所以判断集合该只有一处定义（`_EVALUATED_TYPES`），"多集合并集"那种写法
就是不一致的来源。
