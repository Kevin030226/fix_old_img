"""
凭据安全工具（修复 B4：明文口令 / 非原子写）。

设计要点：
- 口令哈希使用标准库 hashlib.pbkdf2_hmac（无第三方依赖），
  格式 self-describing： pbkdf2_<alg>$<rounds>$<salt_hex>$<hash_hex>
- 校验使用 hmac.compare_digest 做常量时间比较，抵御时序侧信道；
- 写入采用「临时文件 + os.replace」原子提交，并叠加 filelock 跨进程锁，
  避免并发写造成半截文件或丢失更新。
"""
import hashlib
import hmac
import os
import secrets
import tempfile
import threading
from contextlib import contextmanager

import yaml

try:
    from filelock import FileLock
except Exception:  # pragma: no cover - 极少数环境无 filelock 时退化为进程内锁
    FileLock = None

ALG = "sha256"
DEFAULT_ROUNDS = 260_000  # 与 config/users.example.yaml 对齐

# 进程内串行化（单一 uvicorn 进程内的并发请求）
_WRITE_LOCK = threading.Lock()


def hash_password(password: str, *, rounds: int = DEFAULT_ROUNDS) -> str:
    """对明文口令生成 self-describing 的哈希字符串。"""
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac(ALG, password.encode("utf-8"), salt, rounds)
    return f"pbkdf2_{ALG}${rounds}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """常量时间校验。stored 非法/非本格式一律返回 False。"""
    if not stored or "$" not in stored:
        return False
    try:
        alg, rounds_s, salt_hex, hash_hex = stored.split("$", 3)
    except ValueError:
        return False
    if alg != f"pbkdf2_{ALG}":
        return False
    try:
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
        rounds = int(rounds_s)
    except ValueError:
        return False
    dk = hashlib.pbkdf2_hmac(ALG, password.encode("utf-8"), salt, rounds)
    return hmac.compare_digest(dk, expected)


def needs_rehash(stored: str, *, rounds: int = DEFAULT_ROUNDS) -> bool:
    """判断是否需要根据当前参数重新哈希（如提升迭代次数时）。"""
    try:
        alg, rounds_s, _, _ = stored.split("$", 3)
    except ValueError:
        return True
    return alg != f"pbkdf2_{ALG}" or int(rounds_s) != rounds


def is_plaintext(stored: str) -> bool:
    """判断存储值是否为历史明文（未哈希）。"""
    return bool(stored) and not stored.startswith(f"pbkdf2_{ALG}")


@contextmanager
def _cross_process_lock(path: str):
    if FileLock is not None:
        with FileLock(path + ".lock"):
            yield
    else:
        yield


def write_yaml_atomic(path: str, data: dict) -> None:
    """原子写 YAML：临时文件落盘后 os.replace 提交，叠加 filelock 防并发竞态。"""
    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    with _WRITE_LOCK:
        with _cross_process_lock(path):
            fd, tmp = tempfile.mkstemp(dir=directory, suffix=".yaml.tmp")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    yaml.dump(
                        data, f,
                        allow_unicode=True,
                        default_flow_style=False,
                        sort_keys=False,
                    )
                os.replace(tmp, path)
            except BaseException:
                if os.path.exists(tmp):
                    os.remove(tmp)
                raise
