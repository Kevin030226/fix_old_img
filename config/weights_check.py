"""权重文件完整性自检（P1-②）。

动机：项目权重（*.pth / *.pt / *.dat）体积大、未被 git 跟踪，也从未做
完整性校验。若权重损坏或被替换，模型会静默产出错误结果或崩溃，难以定位。
本模块在**服务启动**时校验已知基线清单，缺失/篡改即拒绝启动并给出明确报错。

设计：
- 基线清单 config/weights_manifest.json 由本模块的 generate 生成并提交，
  作为防篡改基准（清单只存哈希，不含权重本体）。
- verify_weights() 在服务启动时调用一次（权重为静态文件，运行期不变），
  采用“启动即失败”（fail-fast）策略。
- 清单缺失时明确提示先生成；清单内任一文件缺失或哈希不符则抛出异常。

⚠️ 局限：清单本身也需保护（它不是机密，但应随权重一同受控）。本检查用于
检测“意外损坏/错误替换”，不抵御“同时篡改权重与清单”的有意攻击。
"""
import hashlib
import json
import os
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST_PATH = os.path.join(BASE_DIR, "config", "weights_manifest.json")
ALGORITHM = "sha256"
WEIGHT_EXTS = (".pth", ".pt", ".dat")
EXCLUDE_DIRS = {"__pycache__", ".git", "node_modules", ".workbuddy", ".arts"}


class WeightsIntegrityError(RuntimeError):
    """权重完整性校验失败。"""

    pass


def _discover():
    """递归发现所有权重文件，返回 [(相对路径, 绝对路径), ...]（按相对路径排序）。"""
    found = []
    for root, dirs, files in os.walk(BASE_DIR):
        # 原地修剪待遍历目录，跳过无关目录
        dirs[:] = [
            d
            for d in dirs
            if d not in EXCLUDE_DIRS and not d.startswith(".")
        ]
        for f in files:
            if f.lower().endswith(WEIGHT_EXTS):
                ap = os.path.join(root, f)
                rel = os.path.relpath(ap, BASE_DIR).replace(os.sep, "/")
                found.append((rel, ap))
    found.sort(key=lambda x: x[0])
    return found


def compute_sha256(path, chunk_size=1 << 20):
    """分块计算 SHA-256，避免大文件一次性读入内存。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def generate_manifest(manifest_path=None):
    """扫描当前权重并写出基线清单。返回清单 dict。

    注意：manifest_path 默认值必须为 None 并在运行期解析模块级 MANIFEST_PATH。
    若写成 `manifest_path=MANIFEST_PATH`，默认值会在 import 时即绑定为生产路径，
    测试中覆写 wc.MANIFEST_PATH 将不生效，导致测试产物覆盖真实基线清单
    （该问题曾真实发生：生产清单一度被测试的 2 个假文件覆盖）。
    """
    manifest_path = manifest_path or MANIFEST_PATH
    files = []
    for rel, ap in _discover():
        files.append({"path": rel, "sha256": compute_sha256(ap)})
    manifest = {
        "algorithm": ALGORITHM,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "file_count": len(files),
        "files": files,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def verify_weights(manifest_path=None):
    """校验基线清单：缺失清单/文件缺失/哈希不符均抛 WeightsIntegrityError。

    返回通过校验的文件数。

    manifest_path 同 generate_manifest：默认 None，运行期解析，避免 import 期绑定。
    """
    manifest_path = manifest_path or MANIFEST_PATH
    if not os.path.exists(manifest_path):
        raise WeightsIntegrityError(
            f"未找到权重校验清单 {manifest_path}。\n"
            f"请先生成基线：python -m config.weights_check generate"
        )
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    expected_alg = manifest.get("algorithm", ALGORITHM)
    if expected_alg != ALGORITHM:
        raise WeightsIntegrityError(
            f"清单算法为 {expected_alg}，与当前期望 {ALGORITHM} 不一致，请重新生成清单。"
        )

    problems = []
    for entry in manifest.get("files", []):
        rel = entry["path"]
        ap = os.path.join(BASE_DIR, rel)
        if not os.path.exists(ap):
            problems.append(f"文件缺失: {rel}")
            continue
        actual = compute_sha256(ap)
        expected = entry.get("sha256")
        if actual != expected:
            problems.append(
                f"哈希不符: {rel}\n    期望 {expected}\n    实际 {actual}"
            )

    if problems:
        raise WeightsIntegrityError(
            "权重完整性校验失败，服务拒绝启动：\n- " + "\n- ".join(problems)
        )
    return len(manifest.get("files", []))


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "generate":
        m = generate_manifest()
        print(f"已生成基线清单：{MANIFEST_PATH}")
        print(f"权重文件数：{m['file_count']}")
    elif cmd == "verify":
        try:
            n = verify_weights()
            print(f"权重完整性校验通过 ✅（{n} 个文件）")
        except WeightsIntegrityError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
    else:
        print("用法: python -m config.weights_check [generate|verify]", file=sys.stderr)
        sys.exit(2)
