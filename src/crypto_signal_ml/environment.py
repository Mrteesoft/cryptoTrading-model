"""Helpers for loading local environment variables from a .env file."""

import os
from pathlib import Path
from typing import Dict


def load_env_file(env_path: Path) -> Dict[str, str]:
    """
    Load key-value pairs from a local `.env` file into `os.environ`.

    Notes:
    - existing shell environment variables win over the file
    - blank lines and `#` comments are ignored
    - optional `export KEY=value` syntax is supported
    - simple quoted values are unwrapped for convenience
    """

    loaded_values: Dict[str, str] = {}

    if not env_path.exists():
        return loaded_values

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line[len("export ") :].strip()

        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)
        loaded_values[key] = value

    return loaded_values
