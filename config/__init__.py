"""User configuration loading module"""

import os
import sys
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "users.yaml")


def load_users() -> dict:
    """
    Load user configuration from users.yaml.

    Returns:
        {username: {"password": str, "role": str}, ...}

    Raises:
        FileNotFoundError: users.yaml does not exist
        KeyError: users.yaml format is invalid
    """
    if not os.path.exists(CONFIG_PATH):
        print(f"Error: user configuration file not found - {CONFIG_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not data or "users" not in data:
        print("Error: invalid user configuration file format, missing 'users' field")
        sys.exit(1)

    return data["users"]
