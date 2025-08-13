import os

from .. import impl


class PythonSideCache:
    def __init__(self) -> None:
        _cache_parent_folder = impl.get_runtime().prog.config().offline_cache_file_path
        self.cache_folder = os.path.join(_cache_parent_folder, "python_side_cache")
        os.makedirs(self.cache_folder, exist_ok=True)

    def _get_filepath(self, key: str) -> str:
        filepath = os.path.join(self.cache_folder, f"{key}.cache.txt")
        return filepath

    def _touch(self, filepath):
        """
        We can then simply remove any files older than ~24 hours or so,
        to keep the cache automatically clean.
        No need for metadata and stuff :)
        """
        with open(filepath, "a"):
            os.utime(filepath, None)

    def store(self, key: str, value: str) -> None:
        filepath = self._get_filepath(key)
        with open(filepath, "w") as f:
            f.write(value)

    def try_load(self, key: str) -> str | None:
        filepath = self._get_filepath(key)
        if not os.path.isfile(filepath):
            return None
        self._touch(filepath)
        with open(filepath) as f:
            return f.read()
