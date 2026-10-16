"""用户配置加载模块"""

import os
import sys
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.yaml")


def load_users() -> dict:
    """
    从 users.yaml 加载用户配置。

    Returns:
        {username: {"password": str, "role": str}, ...}

    Raises:
        FileNotFoundError: users.yaml 不存在
        KeyError: users.yaml 格式不正确
    """
    if not os.path.exists(CONFIG_PATH):
        print(f"错误：用户配置文件不存在 - {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or "users" not in data:
        print("错误：用户配置文件格式不正确，缺少 'users' 字段")
        sys.exit(1)

    return data["users"]
