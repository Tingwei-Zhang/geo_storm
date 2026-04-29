import json
import os
import pickle
import sys

try:
    import toml
except ImportError:
    toml = None


def load_api_key(toml_file_path: str) -> None:
    """Load API keys from a TOML file into os.environ. If file is missing, do nothing."""
    if not toml_file_path or not os.path.isfile(toml_file_path):
        return
    if toml is None:
        return
    try:
        with open(toml_file_path, "r") as file:
            data = toml.load(file)
    except Exception as e:
        print(f"Error loading {toml_file_path}: {e}", file=sys.stderr)
        return
    for key, value in data.items():
        if isinstance(value, str):
            os.environ[key] = value


def makeStringRed(message):
    return f"\033[91m {message}\033[00m"



class FileIOHelper:
    @staticmethod
    def dump_json(obj, file_name, encoding="utf-8"):
        with open(file_name, 'w', encoding=encoding) as fw:
            json.dump(obj, fw, default=FileIOHelper.handle_non_serializable, ensure_ascii=False)

    @staticmethod
    def handle_non_serializable(obj):
        return "non-serializable contents"  # mark the non-serializable part

    @staticmethod
    def load_json(file_name, encoding="utf-8"):
        with open(file_name, 'r', encoding=encoding) as fr:
            return json.load(fr)

    @staticmethod
    def write_str(s, path):
        with open(path, 'w') as f:
            f.write(s)

    @staticmethod
    def load_str(path):
        with open(path, 'r') as f:
            return '\n'.join(f.readlines())

    @staticmethod
    def dump_pickle(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def load_pickle(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

