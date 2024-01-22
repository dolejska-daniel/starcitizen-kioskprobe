import json
import logging
from pathlib import Path

from kiosk_probe.uexcorp.objects import UEXCorpApiConfig

log = logging.getLogger("kiosk_probe." + __name__)


class Settings:
    def __init__(self):
        log.debug("loading settings")
        self.base_path = Path(__file__).parent / "config"
        self.show_images = False
        self.show_report_links = True
        self.show_all_text_nodes = False
        self.crop_resulting_image = True
        self.dry_run = False
        self.uexcorp_api_config = self.load_uexcorp()

    def load_uexcorp(self) -> UEXCorpApiConfig:
        filepath = self.base_path / "uex.json"
        if not filepath.exists():
            raise FileNotFoundError(f"UEXCorp API config file not found at {filepath.absolute()}")

        with filepath.open() as file:
            data = json.load(file)
            return UEXCorpApiConfig(**data)
