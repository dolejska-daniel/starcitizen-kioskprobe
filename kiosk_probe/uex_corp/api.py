import dataclasses
import http.cookiejar
import logging
from http.cookiejar import CookieJar

import requests

from .cache import FileCache, MemoryCache
from .objects import *

log = logging.getLogger("kiosk_probe." + __name__)


class UEXCorpApi:

    def __init__(self, config: UEXCorpApiConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.config.application_token}",
            "secret_key": self.config.user_token,
        })
        cookies = http.cookiejar.MozillaCookieJar()
        if self.config.cookies_filepath:
            cookies.load(filename=config.cookies_filepath)

        self.session.cookies = cookies
        self.cache = FileCache() if self.config.use_file_cache else MemoryCache()

    def get(self, url: str) -> ResponseBase:
        if cached_data := self.cache.load(url):
            return cached_data

        response = self.session.get(f"{self.config.base_url}{url}")
        response.raise_for_status()

        data = response.json()
        self.cache.save(url, data)

        return data

    def get_star_systems(self) -> list[StarSystem]:
        data = self.get(f"/star_systems")
        return [StarSystem(**star_system) for star_system in data["data"]]

    def get_terminals(self) -> list[Terminal]:
        data = self.get(f"/terminals")
        return [Terminal(**terminal) for terminal in data["data"]]

    def get_commodities(self) -> list[Commodity]:
        data = self.get(f"/commodities")
        return [Commodity(**commodity) for commodity in data["data"]]

    def get_commodity_prices_by_terminal(self, id_terminal: int) -> list[CommodityPrice]:
        data = self.get(f"/commodities_prices/id_terminal/{id_terminal}")
        return [CommodityPrice(**commodity_price) for commodity_price in data["data"]]

    def submit_data_run(self, data_run: DataRun) -> DataRunResponse | None:
        request_data = dataclasses.asdict(data_run)
        try:
            response = self.session.post(f"{self.config.base_url}/data_submit", json=request_data)
            log.debug("submission response response: %s", response.text)

        except Exception as e:
            log.exception("failed to properly send data run to UEX API: %s", request_data, exc_info=e)
            return None

        try:
            response.raise_for_status()

            response_data: ResponseBase = response.json()
            return DataRunResponse(**response_data)

        except Exception as e:
            log.exception("failed processing UEX API response: %s", response.text, exc_info=e)
            return None


class UEXCorp:
    def __init__(self, config: UEXCorpApiConfig):
        self.api = UEXCorpApi(config)
        self.star_systems = self.api.get_star_systems()
        self.terminals = self.api.get_terminals()
        self.commodities = self.api.get_commodities()

    def get_star_system_by_id(self, id_system: int) -> StarSystem:
        for star_system in self.star_systems:
            if star_system.id == id_system:
                return star_system

        raise ValueError(f"Star system with id {id_system} not found")

    def get_terminal_by_id(self, id_terminal: int) -> Terminal:
        for terminal in self.terminals:
            if terminal.id == id_terminal:
                return terminal

        raise ValueError(f"Terminal with id {id_terminal} not found")

    def get_commodity_by_id(self, id_commodity: int) -> Commodity:
        for commodity in self.commodities:
            if commodity.id == id_commodity:
                return commodity

        raise ValueError(f"Commodity with id {id_commodity} not found")

    def get_commodity_by_code(self, code: str) -> Commodity:
        for commodity in self.commodities:
            if commodity.code == code:
                return commodity

        raise ValueError(f"Commodity with code {code} not found")

    def get_commodity_price_by_terminal(self, id_terminal: int, id_commodity: int) -> CommodityPrice | None:
        for commodity_price in self.api.get_commodity_prices_by_terminal(id_terminal):
            if commodity_price.id_commodity == id_commodity:
                return commodity_price

        return None
