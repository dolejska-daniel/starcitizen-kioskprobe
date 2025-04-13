import abc
import json
from pathlib import Path

from kiosk_probe.uex_corp.objects import ResponseBase


class Cache(metaclass=abc.ABCMeta):

    def get_cache_key(self, url: str) -> str:
        return url.strip("/").replace("/", "-")

    @abc.abstractmethod
    def load(self, url: str) -> ResponseBase | None:
        pass

    @abc.abstractmethod
    def save(self, url: str, data: ResponseBase) -> None:
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        pass


class FileCache(Cache):

    def load(self, url: str) -> ResponseBase | None:
        cache_key = self.get_cache_key(url)
        cache_file_path = Path(f"cache/{cache_key}.json")
        if cache_file_path.exists():
            with cache_file_path.open() as f:
                return json.load(f)

        return None

    def save(self, url: str, data: ResponseBase) -> None:
        cache_key = self.get_cache_key(url)
        cache_file_path = Path(f"cache/{cache_key}.json")
        with cache_file_path.open("w") as f:
            json.dump(data, f, indent=4)

    def clear(self) -> None:
        for path in Path(f"cache/").glob("*.json"):
            path.unlink()


class MemoryCache(Cache):
    data: dict[str, ResponseBase]

    def __init__(self):
        self.data = {}

    def load(self, url: str) -> ResponseBase | None:
        cache_key = self.get_cache_key(url)
        return self.data.get(cache_key, None)

    def save(self, url: str, data: ResponseBase) -> None:
        cache_key = self.get_cache_key(url)
        self.data[cache_key] = data

    def clear(self) -> None:
        self.data.clear()
