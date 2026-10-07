import hashlib
import json
import os

from src.llm.config import MANIFEST_PATH


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# Records an index built for an arm under the manifest's index_variants: each file with its sha256
def record_index_variant(name, **files):
    entry = {}
    for key, path in files.items():
        entry[key] = path
        entry[f"{key}_sha256"] = file_sha256(path)
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    manifest.setdefault("index_variants", {})[name] = entry
    tmp = f"{MANIFEST_PATH}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, MANIFEST_PATH)
    return entry
