import logging
import re
from dataclasses import dataclass, field

from kiosk_probe.core.enums import InventoryAvailability

log = logging.getLogger("kiosk_probe." + __name__)


@dataclass
class InventoryEntry:
    float_pattern = re.compile(r".*?(?P<value>\d+(\.\d+)?).*")

    entry_name: str | None = field(default=None)
    entry_availability: str | None = field(default=None)
    entry_stock: str | None = field(default=None)
    entry_price: str | None = field(default=None)

    @property
    def availability(self) -> InventoryAvailability | None:
        text = self.entry_availability.upper() if self.entry_availability else None
        log.debug("parsing availability from %s (%s)", text, self.entry_name)
        match text:
            case "OUT OF STOCK": result = InventoryAvailability.OUT_OF_STOCK
            case "VERY LOW INVENTORY": result = InventoryAvailability.VERY_LOW_INVENTORY
            case "LOW INVENTORY": result = InventoryAvailability.LOW_INVENTORY
            case "MEDIUM INVENTORY": result = InventoryAvailability.MEDIUM_INVENTORY
            case "HIGH INVENTORY": result = InventoryAvailability.HIGH_INVENTORY
            case "VERY HIGH INVENTORY": result = InventoryAvailability.VERY_HIGH_INVENTORY
            case "MAX INVENTORY": result = InventoryAvailability.MAX_INVENTORY
            case _: result = None

        log.debug("parsed availability: %s (%s)", result, self.entry_name)
        return result

    @property
    def stock(self) -> float | None:
        text = self.entry_stock
        try:
            log.debug("parsing stock from %s (%s)", text, self.entry_name)
            text = text.replace(",", "")

            value = float(self.float_pattern.match(text or "").group("value"))
            if text.endswith("cSCU"):
                value /= 100
            if text.endswith("uSCU") or text.endswith("μSCU"):
                value /= 1_000_000

            log.debug("parsed stock: %s (%s)", value, self.entry_name)
            return value

        except Exception as ex:
            log.warning("could not parse stock from %s", text, exc_info=ex)
            return None

    @property
    def price(self) -> float | None:
        text = self.entry_price.upper() if self.entry_price else "0"
        try:
            log.debug("parsing price from %s (%s)", text, self.entry_name)
            text = text.replace(",", "")

            value = float(self.float_pattern.match(text or "").group("value"))
            if text.endswith("K"):
                value *= 1_000
            if text.endswith("M"):
                value *= 1_000_000
            if text.endswith("G"):
                value *= 1_000_000_000

            log.debug("parsed price: %s (%s)", value, self.entry_name)
            return value

        except Exception as ex:
            log.warning("could not parse price from %s", text, exc_info=ex)
            return None


@dataclass
class InventoryEntriesResponse:
    entries: list[InventoryEntry] = field(default_factory=list)

    # noinspection PyArgumentList
    def __post_init__(self):
        self.entries = [InventoryEntry(**entry) for entry in self.entries]
