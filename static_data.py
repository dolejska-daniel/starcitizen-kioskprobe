import csv
import logging
from pathlib import Path
from typing import TypedDict

from kiosk_probe.uexcorp.objects import InventoryStatus

log = logging.getLogger("kiosk_probe." + __name__)


class StaticData:
    class System(TypedDict):
        name: str
        code: str

    class TradePort(TypedDict):
        name: str
        code: str
        system_code: str

    class Commodity(TypedDict):
        name: str
        code: str
        title: str

    trade_port: dict[str, TradePort]
    commodity_by_name: dict[str, Commodity]
    commodity_by_code: dict[str, Commodity]

    def __init__(self, base_path: Path = None):
        base_path = base_path or Path(__file__).parent / "static"
        log.debug("loading static data from %s", base_path.absolute())
        self.load_commodities(base_path)
        self.load_stations(base_path)
        self.inventory_states = [
            InventoryStatus(1, "       OUT OF STOCK"),
            InventoryStatus(2, " VERY LOW INVENTORY"),
            InventoryStatus(3, "      LOW INVENTORY"),
            InventoryStatus(4, "   MEDIUM INVENTORY"),
            InventoryStatus(5, "     HIGH INVENTORY"),
            InventoryStatus(6, "VERY HIGH INVENTORY"),
            InventoryStatus(7, "      MAX INVENTORY"),
            InventoryStatus(7, "          NO DEMAND"),
        ]

    def load_commodities(self, base_path: Path):
        filepath = base_path / "commodities.csv"
        with filepath.open() as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            self.commodity_by_name = {row["name"]: row for row in rows}
            self.commodity_by_code = {row["code"]: row for row in rows}

    def load_stations(self, base_path: Path):
        filepath = base_path / "stations.csv"
        with filepath.open() as file:
            reader = csv.DictReader(file)
            self.trade_port = {row["code"]: row for row in list(reader)}
