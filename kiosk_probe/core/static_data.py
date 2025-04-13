import logging
from pathlib import Path

from kiosk_probe.core.processing.objects import InventoryAvailability
from kiosk_probe.uex_corp.objects import InventoryStatus

log = logging.getLogger("kiosk_probe." + __name__)


class StaticData:

    def __init__(self, base_path: Path = None):
        base_path = base_path or Path(__file__).parent / "static"
        log.debug("loading static data from %s", base_path.absolute())
        self.inventory_states = [
            InventoryStatus(1, "OUT OF STOCK"),
            InventoryStatus(2, "VERY LOW INVENTORY"),
            InventoryStatus(3, "LOW INVENTORY"),
            InventoryStatus(4, "MEDIUM INVENTORY"),
            InventoryStatus(5, "HIGH INVENTORY"),
            InventoryStatus(6, "VERY HIGH INVENTORY"),
            InventoryStatus(7, "MAX INVENTORY"),
            InventoryStatus(7, "NO DEMAND", visible=False),
        ]
        self.inventory_availability_to_status = {
            InventoryAvailability.OUT_OF_STOCK: InventoryStatus(1, "OUT OF STOCK"),
            InventoryAvailability.VERY_LOW_INVENTORY: InventoryStatus(2, "VERY LOW INVENTORY"),
            InventoryAvailability.LOW_INVENTORY: InventoryStatus(3, "LOW INVENTORY"),
            InventoryAvailability.MEDIUM_INVENTORY: InventoryStatus(4, "MEDIUM INVENTORY"),
            InventoryAvailability.HIGH_INVENTORY: InventoryStatus(5, "HIGH INVENTORY"),
            InventoryAvailability.VERY_HIGH_INVENTORY: InventoryStatus(6, "VERY HIGH INVENTORY"),
            InventoryAvailability.MAX_INVENTORY: InventoryStatus(7, "MAX INVENTORY"),
            InventoryAvailability.NO_DEMAND: InventoryStatus(7, "NO DEMAND", visible=False),
            InventoryAvailability.Undetected: InventoryStatus(1, "OUT OF STOCK"),
            None: None,
        }
