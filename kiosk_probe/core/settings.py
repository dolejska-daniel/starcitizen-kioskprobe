import json
import logging
from pathlib import Path

from kiosk_probe.uex_corp.objects import UEXCorpApiConfig, OpenAIApiConfig

log = logging.getLogger("kiosk_probe." + __name__)


class Settings:
    def __init__(self):
        log.debug("loading settings")
        self.base_path = Path(__file__).parent.parent.parent / "config"
        self.show_images = False
        self.show_report_links = True
        self.include_screenshots = True
        self.dry_run = False
        self.uex_corp_config = self.load_uex_corp()
        self.openai_config = self.load_openai()

    def load_uex_corp(self) -> UEXCorpApiConfig:
        filepath = self.base_path / "uex.json"
        if not filepath.exists():
            raise FileNotFoundError(f"UEXCorp config file not found at {filepath.absolute()}")

        with filepath.open() as file:
            data = json.load(file)
            return UEXCorpApiConfig(**data)

    def load_openai(self) -> OpenAIApiConfig:
        filepath = self.base_path / "openai.json"
        if not filepath.exists():
            raise FileNotFoundError(f"OpenAI config file not found at {filepath.absolute()}")

        with filepath.open() as file:
            data = json.load(file)
            return OpenAIApiConfig(**data)
