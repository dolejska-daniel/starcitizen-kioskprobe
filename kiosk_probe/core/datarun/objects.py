from kiosk_probe.uex_corp.objects import InventoryStatus, Commodity, DataRunCommodityBuyEntry, DataRunCommoditySellEntry


class DetectedCommodity:
    def __init__(self, name: str | None, commodity: Commodity | None, price: float | None, stock: float | None, inventory_status: InventoryStatus | None, order: int):
        self.name = name
        self.commodity = commodity
        self.price = price if price is not None else float("nan")
        self.stock = stock if stock is not None else float("nan")
        self.inventory_status = inventory_status
        self.order = order
        self.trust = 1.0

    def __repr__(self):
        return f"DetectedCommodity({self.name}, {self.price:,.5f} aUEC, {self.stock:,.3f} SCU, {self.inventory_status}, {self.trust:.1f} trust)"

    def __str__(self):
        name = self.name if self.name is not None else "Unknown"
        inventory = self.inventory_status.name if self.inventory_status is not None else "inventory unknown"
        return f"{name:30} {self.price:13,.2f} aUEC {self.stock:7,.0f} SCU {inventory:>20}"

    def __eq__(self, other):
        return isinstance(other, DetectedCommodity) and self.matches(other)

    def matches(self, other: "DetectedCommodity") -> bool:
        return self.commodity.id == other.commodity.id

    def merge(self, other: "DetectedCommodity") -> bool:
        if not self.matches(other) \
                or self.price != other.price \
                or self.stock != other.stock:
            return False

        self.trust += other.trust
        return True

    def is_valid(self):
        return self.name is not None \
            and self.commodity is not None \
            and self.price >= 0 \
            and self.stock >= 0 \
            and self.inventory_status is not None

    def create_buy_entry(self) -> DataRunCommodityBuyEntry:
        return DataRunCommodityBuyEntry(
            id_commodity=self.commodity.id,
            scu_buy=int(self.stock),
            price_buy=self.price,
            status_buy=self.inventory_status.value,
        )

    def create_sell_entry(self) -> DataRunCommoditySellEntry:
        return DataRunCommoditySellEntry(
            id_commodity=self.commodity.id,
            scu_sell=int(self.stock),
            price_sell=self.price,
            status_sell=self.inventory_status.value,
        )
