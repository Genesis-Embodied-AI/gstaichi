import hashlib


def hash_string(v: str) -> str:
    return hashlib.sha256(v.encode("utf-8")).hexdigest()
