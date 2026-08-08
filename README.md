# 基于深度学习的旧照片恢复、划痕修复与老照片上色系统

一个面向真实老照片的 Web 端图像修复系统：支持整体质量修复、划痕检测与修复、面部增强，以及黑白老照片自动上色。系统基于 Gradio 6 + FastAPI + PyTorch（CUDA 12.8）构建，附带完整的管理后台（任务记录、照片档案、用户管理）。

---

## 1. 项目解决的问题

老照片在保存和扫描过程中普遍存在以下问题：

- **整体退化**：褪色、模糊、噪点、对比度下降；
- **物理损伤**：折痕、划痕、污渍；
- **黑白化**：早期照片只有灰度信息，缺少自然色彩；
- **面部细节丢失**：人像区域细节严重退化，常规修复难以恢复；
- **修复过程不可见**：批量处理缺少任务记录、结果归档与质量指标。

本项目针对上述问题提供一条完整的处理链路：**整体质量修复 → 划痕检测/修复 → 人脸检测与增强 → 回卷合成**，并在同一平台内新增 **DDColor 老照片上色** 能力，同时提供用户登录、注册、任务历史、照片档案和用户管理等配套功能。

## 2. 主要功能

### 2.1 图像处理模块（四个标签页）

| 模块 | 说明 | 输出 |
| --- | --- | --- |
| 不带划痕的旧照片复原 | 整体质量提升 + 人脸检测与面部增强 | 修复后的彩色照片 + PSNR/SSIM/MAE 指标 |
| 带划痕的旧照片复原 | 自动划痕检测 → 划痕修复 → 质量提升 → 面部增强 | 修复后的彩色照片 + 指标 |
| 划痕检测 | 独立输出划痕位置 | 黑白二值 mask（白色为划痕） |
| 老照片上色 | DDColor 对黑白/灰度照片自动上色 | 彩色照片 |

### 2.2 管理面板（仅管理员可见）

- **任务管理**：处理历史记录、统计概览（任务数/用户数/各类型次数/平均指标）、清空记录；
- **照片档案**：按处理记录浏览原始图片与修复结果；
- **用户管理**：添加、修改（密码/角色）、删除用户。

### 2.3 平台能力

- 用户登录（pbkdf2 哈希、常量时间比对）与公开注册（IP/全局/用户名三重限流）；
- 健康检查接口 `GET /health`；
- 启动时权重完整性自检（SHA-256 清单，缺失或篡改拒绝启动）；
- 请求级目录隔离 + 结果 TTL 自动回收；
- SQLite 存储（用户/历史），首次启动自动从旧版 users.yaml / JSONL 迁移；
- 深色主题界面，管理员/普通用户界面自动区分。

## 3. 技术栈

| 类别 | 技术 |
| --- | --- |
| 语言/环境 | Python 3.11（conda 环境 `fixoldimg-gpu`） |
| Web 框架 | Gradio 6.22 + FastAPI 0.141 + Uvicorn 0.52 |
| 深度学习 | PyTorch 2.7.1+cu128（原生支持 RTX 50 系 sm_120） |
| 视觉/科学计算 | OpenCV 5.0、scikit-image 0.26、scipy 1.17、numpy 2.4、Pillow 12.3 |
| 人脸/上色 | dlib 20.0.1（conda-forge）、timm 0.9.2、DDColor（Apache-2.0） |
| 存储 | SQLite（WAL 模式） |
| 模型 | Bringing Old Photos Back to Life（MIT）+ DDColor（Apache-2.0），共 29 个权重文件 |

## 4. 安装方法

### 4.1 Windows（conda）

一键脚本方式：

```bat
setup_gpu.bat
```

或手动执行：

```bash
# 1. 创建环境
conda create -n fixoldimg-gpu python=3.11 -y
conda activate fixoldimg-gpu

# 2. 安装 PyTorch cu128（支持 RTX 50 系；国内网络可换
#    https://mirror.sjtu.edu.cn/pytorch-wheels/cu128）
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# 3. 安装项目依赖（含 Gradio/FastAPI/OpenCV/DDColor 所需 timm 等）
pip install -r requirements.txt

# 4. 安装 dlib 预编译包（conda-forge，无需本机编译工具链）
conda install -n fixoldimg-gpu -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge \
    --override-channels -y dlib=20.0.1

# 5. 下载模型权重（BOB 修复链路 + DDColor 上色，约 1.5GB）
bash scripts/download_weights.sh

# 6. 若权重来源与仓库清单不同，重新生成并校验
python -m config.weights_check generate
python -m config.weights_check verify
```

> 注意：`config/users.yaml` 会被自动迁移到 SQLite（`admin_data/fixoldimg.db`）。若该文件缺失，服务启动后没有任何账号，请从备份恢复或使用管理端添加。

### 4.2 Docker（Linux + NVIDIA GPU）

```bash
docker build -t fixoldimg .
docker run --gpus all -p 9502:9502 fixoldimg
```

镜像会自动安装依赖、下载全部权重、生成默认管理员账号（`admin/admin123`）并重建权重清单。首次构建时 dlib 为源码编译，耗时约 5-10 分钟。

## 5. 使用方法

### 5.1 启动 Web 服务

```bash
conda activate fixoldimg-gpu
python main.py
```

浏览器访问 <http://127.0.0.1:9502>。

默认账号（**部署前必须修改**）：

| 用户名 | 密码 | 角色 |
| --- | --- | --- |
| admin | 安装/初始化时设置 | 管理员（仅管理面板） |
| user1 | 安装/初始化时设置 | 普通用户 |
| YCTU | 安装/初始化时设置 | 普通用户 |

> ⚠️ 本地开发环境为方便测试保留了演示口令（如 `admin/admin123`），
> 仅限开发使用；公开部署前请通过“管理面板 → 用户管理”轮换全部账号口令。

管理员登录后只显示“管理面板”；普通用户可使用四个图像处理标签页。

### 5.2 Web 操作流程

1. 登录（或点击“立即注册”创建账号）；
2. 选择对应功能标签页；
3. 上传图片（或点击标签页下方示例图自动填充）；
4. 点击“开始修复 / 提交复原 / 开始上色”；
5. 等待处理完成（GPU 环境单张约 1-60 秒，取决于模块与图片大小），查看结果与指标；
6. 管理员可在“管理面板”查看任务记录、照片档案并管理用户。

### 5.3 命令行批量处理

```bash
# 不带划痕的旧照片复原
python run.py --input_folder ./test_images/old --output_folder ./output

# 带划痕的旧照片复原
python run.py --input_folder ./test_images/old_w_scratch --output_folder ./output --with_scratch

# 高清人脸增强（可选）
python run.py --input_folder ./test_images/old --output_folder ./output --HR

# 显式指定 GPU / CPU
python run.py --input_folder ./test_images/old --output_folder ./output --GPU 0
```

输出目录包含 `final_output/`（最终结果）、各阶段中间产物与 `pipeline_report.json`（降级报告）。

### 5.4 常用环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `FIXIMG_HOST` | `127.0.0.1` | 监听地址（局域网访问设为 `0.0.0.0`） |
| `FIXIMG_PORT` | `9502` | 监听端口 |
| `FIXIMG_RESULT_TTL` | `7200` | 推理产物保留秒数 |
| `FIXIMG_HISTORY_MAX` | `2000` | 历史记录上限 |
| `FIXIMG_ARCHIVE_TTL` | `604800` | 归档照片保留秒数 |
| `FIXIMG_DDCOLOR_MODEL` | `pretrained/ddcolor/pytorch_model.pt` | DDColor 权重路径 |
| `FIXIMG_REGISTER_MAX` / `_GLOBAL_MAX` / `_USERNAME_MAX` | `5/20/3` | 注册限流阈值 |

## 6. 输入输出示例

以下示例均由当前系统实际生成（GPU 环境）。

### 6.1 不带划痕的旧照片复原

输入（退化旧照）→ 输出（修复结果）：

| 输入 | 输出 |
| --- | --- |
| ![复原输入](docs/examples/restore_input.png) | ![复原输出](docs/examples/restore_output.png) |

### 6.2 划痕检测

输入（带划痕照片）→ 输出（划痕 mask，白色为划痕）：

| 输入 | 输出 |
| --- | --- |
| ![检测输入](docs/examples/detect_input.png) | ![检测 mask](docs/examples/detect_mask.png) |

### 6.3 老照片上色

输入（黑白灰度照片）→ 输出（DDColor 上色结果）：

| 输入 | 输出 |
| --- | --- |
| ![上色输入](docs/examples/colorize_input.jpg) | ![上色输出](docs/examples/colorize_output.png) |

更多测试样例见 `gradio_examples/`（`old/`、`old_w_scratch/`、`color/`）与 `test_images/`。

## 7. 目录结构

```text
.
├── main.py                  # Web 服务入口（Gradio 6 + FastAPI）
├── run.py                   # 四阶段推理流水线 CLI
├── admin_panel.py           # 管理面板（Gradio 组件）
├── app/                     # 应用层：db(SQLite)/pipeline/colorizer
├── ddcolor/                 # DDColor 上色模型（Apache-2.0）
├── basicsr/                 # DDColor 所需最小子集
├── Global/                  # 整体质量修复 / 划痕检测模型
├── Face_Detection/          # dlib 人脸检测 / 回卷
├── Face_Enhancement/        # 面部增强模型
├── config/                  # users.yaml / 限流 / 安全 / 权重清单
├── gradio_examples/         # Web 示例图片
├── docs/examples/           # README 示例输入输出
├── pretrained/ddcolor/      # DDColor 权重（不入库）
├── scripts/download_weights.sh
├── Dockerfile
└── setup_gpu.bat
```

## 8. 常见问题

**Q1：启动提示“权重完整性校验失败”**
权重缺失或哈希不符。运行 `bash scripts/download_weights.sh` 补齐，然后执行 `python -m config.weights_check generate` 重新生成清单（本地权重与仓库清单不一致时同样处理）。

**Q2：CUDA 不可用 / 提示 sm_120 不兼容**
请确认安装的是 torch 2.7.1+cu128 及以上（RTX 50 系需要 cu128 构建）；老版本 cu121 不支持 Blackwell 架构。

**Q3：老照片上色首次很慢**
首次调用需要加载 DDColor 权重（约 3 秒），之后每张约 1 秒（GPU）。

**Q4：Windows 控制台中文乱码**
部分终端默认 GBK 编码，运行 Python 前可执行 `set PYTHONIOENCODING=utf-8`。

**Q5：想从局域网访问**
以 `FIXIMG_HOST=0.0.0.0` 启动，并确保防火墙放行 `FIXIMG_PORT`。

## 9. 第三方组件与许可

- **本项目自身代码**：MIT License，见根目录 [LICENSE](LICENSE)；
- **Bringing Old Photos Back to Life**（修复/检测/面部增强模型）：MIT License，见 [LICENSE-Bringing-Old-Photos-Back-to-Life](LICENSE-Bringing-Old-Photos-Back-to-Life)；
- **DDColor**（老照片上色，ICCV 2023）：Apache-2.0，见 [ddcolor/LICENSE](ddcolor/LICENSE)；
- 完整说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
