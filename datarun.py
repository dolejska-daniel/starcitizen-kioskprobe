import logging
from typing import Sequence

import numpy as np
from InquirerPy import inquirer
from InquirerPy.base import Choice
from rich.console import Console
from rich.table import Table

from enums import EditTarget, ItemType
from kiosk_probe.uexcorp.api import UEXCorp
from kiosk_probe.uexcorp.objects import InventoryStatus, Terminal, Commodity, CommodityPrice
from utils import TextNode

log = logging.getLogger("kiosk_probe." + __name__)


class DetectedCommodity:
    def __init__(self, name: str | None, commodity: Commodity | None, price: float | None, stock: float | None,
                 inventory: InventoryStatus | None = None, position_y: int | None = None, source_node: TextNode | None = None):
        self.name = name
        self.commodity = commodity
        self.price = price if price is not None else float("nan")
        self.stock = stock if stock is not None else float("nan")
        self.inventory = inventory
        self.position_y = position_y
        self.source_node = source_node
        self.trust = 1.0

    def __repr__(self):
        return f"DetectedCommodity({self.name}, {self.price:,.5f} aUEC, {self.stock:,.3f} SCU, {self.inventory}, {self.trust:.1f} trust)"

    def __str__(self):
        name = self.name if self.name is not None else "Unknown"
        inventory = self.inventory.name if self.inventory is not None else "inventory unknown"
        return f"{name:30} {self.price:12,.2f} aUEC {self.stock:7,.0f} SCU, {inventory:19}"

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
        return self.name is not None and self.commodity is not None and self.price >= 0 and self.stock >= 0 and self.inventory is not None

    def get_all_nodes(self) -> set[TextNode]:
        result = set()
        if self.source_node is not None:
            result.add(self.source_node)
            result.update(self.source_node.get_connected())

        return result


class DataRunManager:
    buy: list[DetectedCommodity]
    sell: list[DetectedCommodity]
    images: list[np.ndarray]

    def __init__(self, uexcorp: UEXCorp):
        self.uexcorp = uexcorp
        log.debug("initializing data run manager")
        self.buy = []
        self.sell = []
        self.images = []
        self.terminal: Terminal | None = None

    def add_image(self, image: np.ndarray):
        log.debug("adding image to data run")
        self.images.append(image)

    def add_buy(self, commodity: DetectedCommodity):
        self.buy.append(commodity)
        self.buy = self.merge_duplicates(self.buy)
        log.debug("current buy list: %s", self.buy)

    def extend_buy(self, items: Sequence[DetectedCommodity]):
        log.debug("extending buy list with: %s", items)
        self.buy.extend(items)
        self.buy = self.merge_duplicates(self.buy)
        log.debug("current buy list: %s", self.buy)

    def add_sell(self, commodity: DetectedCommodity):
        self.sell.append(commodity)
        self.sell = self.merge_duplicates(self.sell)
        log.debug("current sell list: %s", self.sell)

    def extend_sell(self, items: Sequence[DetectedCommodity]):
        log.debug("extending sell list with: %s", items)
        self.sell.extend(items)
        self.sell = self.merge_duplicates(self.sell)
        log.debug("current sell list: %s", self.sell)

    def clear(self):
        log.debug("clearing current data run")
        self.buy.clear()
        self.sell.clear()
        self.images.clear()

    def is_dirty(self):
        return len(self.buy) > 0 or len(self.sell) > 0

    def merge_duplicates(self, items: Sequence[DetectedCommodity]) -> list[DetectedCommodity]:
        items_merged = []
        for item in items:
            for item_merged in items_merged:
                if item_merged.merge(item):
                    log.debug("merged item %s into %s", item, item_merged)
                    break

            else:
                items_merged.append(item)

        # select highest trust by commodity code
        result_items = {}
        for item in items_merged:
            id_commodity = item.commodity.id
            if id_commodity not in result_items or item.trust > result_items[id_commodity].trust:
                result_items[id_commodity] = item

            elif item.trust == result_items[id_commodity].trust:
                item_kept: DetectedCommodity = inquirer.select(
                    message="Cannot decide based on trust, select an item to keep:",
                    choices=[
                        Choice(item, name=str(item)),
                        Choice(result_items[id_commodity], name=str(result_items[id_commodity])),
                    ],
                ).execute()
                result_items[id_commodity] = item_kept

        return list(result_items.values())

    def item_overview(self, items: Sequence[DetectedCommodity] | None, print_results: bool = True, prefix: str = "Detected", item_type: ItemType = ItemType.UNDEFINED)\
            -> tuple[list[DetectedCommodity], list[DetectedCommodity]]:
        items = items if items is not None else self.buy + self.sell

        items_valid = list(filter(lambda _item: _item.is_valid(), items))
        items_invalid = list(filter(lambda _item: not _item.is_valid(), items))

        if print_results:
            console = Console()
            table_valid = Table(title=f"{prefix} {len(items_valid)} valid items")
            table_valid.add_column("Name")
            table_valid.add_column("Price", justify="right")
            table_valid.add_column("Stock", justify="right")
            table_valid.add_column("Inventory")
            table_valid.add_column("Price Change", justify="right")
            table_valid.add_column("Stock Change", justify="right")

            for item in sorted(items_valid, key=lambda i: i.position_y):
                price_change = None
                stock_change = None
                change_is_large = False
                change_is_new = False
                prices = self.uexcorp.get_commodity_price_by_terminal(self.terminal.id, item.commodity.id) if self.terminal is not None else None
                if prices is None:
                    price_change_text = f"+ NEW at {self.terminal.code}" if self.terminal is not None else "terminal unset"
                    stock_change_text = f"+ NEW at {self.terminal.code}" if self.terminal is not None else "terminal unset"
                    change_is_new = True

                else:
                    sus_attributes = self.get_sus_attrs(item, prices, item_type)
                    match item_type:
                        case ItemType.BUY:
                            price_change_text, price_change = self.percentage_diff(item.price, prices.price_buy)
                            if EditTarget.PRICE in sus_attributes:
                                price_change_text = f"[bright_red bold blink]{price_change_text}[/bright_red bold blink]"
                                change_is_large = True

                            stock_change_text, stock_change = self.percentage_diff(item.stock, prices.scu_buy)
                            if EditTarget.STOCK in sus_attributes:
                                stock_change_text = f"[bright_red bold blink]{stock_change_text}[/bright_red bold blink]"
                                change_is_large = True

                        case ItemType.SELL:
                            price_change_text, price_change = self.percentage_diff(item.price, prices.price_sell)
                            if EditTarget.PRICE in sus_attributes:
                                price_change_text = f"[bright_red bold blink]{price_change_text}[/bright_red bold blink]"
                                change_is_large = True

                            stock_change_text, stock_change = self.percentage_diff(item.stock, prices.scu_sell)
                            if EditTarget.STOCK in sus_attributes:
                                stock_change_text = f"[bright_red bold blink]{stock_change_text}[/bright_red bold blink]"
                                change_is_large = True

                        case _:
                            price_change_text = "N/A"
                            stock_change_text = "N/A"

                change_is_new = change_is_new or price_change is None or stock_change is None
                table_valid.add_row(
                    item.name,
                    f"{item.price:,.2f} aUEC",
                    f"{item.stock:,.0f} SCU",
                    item.inventory.name,
                    price_change_text,
                    stock_change_text,
                    style="yellow" if change_is_new else "bright_green" if not change_is_large else "",
                )

            console.print(table_valid)

            if len(items_invalid) > 0:
                table_invalid = Table(title=f"{prefix} {len(items_invalid)} invalid items")
                table_invalid.add_column("Name")
                table_invalid.add_column("Price", justify="right")
                table_invalid.add_column("Stock", justify="right")
                table_invalid.add_column("Inventory")

                for item in sorted(items_invalid, key=lambda i: i.position_y):
                    table_invalid.add_row(
                        item.name,
                        f"{item.price:,.2f} aUEC",
                        f"{item.stock:,.0f} SCU",
                        item.inventory.name if item.inventory is not None else "unknown",
                    )

                print()
                console.print(table_invalid)

            print()

        return items_valid, items_invalid

    def percentage_diff(self, value_current: float, value_previous: float, suffix: str = "") -> tuple[str, float | None]:
        try:
            if value_previous == 0:
                if value_current == 0:
                    price_pct_diff = 0.0
                else:
                    price_pct_diff = float("inf")

            else:
                price_pct_diff = (value_current - value_previous) / value_previous * 100

            prefix = "⌃ " if price_pct_diff > 0 else "⌄ " if price_pct_diff < 0 else "= "
            return f"{prefix}{price_pct_diff:+6.1f}%{suffix}", price_pct_diff

        except ArithmeticError:
            return "N/A", None

    def sync_item_changes(self):
        log.debug("syncing item changes")
        self.buy = list(self.filter_untrusted(self.buy))
        self.sell = list(self.filter_untrusted(self.sell))

    def filter_untrusted(self, items: Sequence[DetectedCommodity]) -> list[DetectedCommodity]:
        for item in items:
            if item.trust <= 0.0:
                log.debug("ignoring untrusted item %s", repr(item))
                continue

            yield item

    def current_terminal_name(self):
        return t.name if (t := self.terminal) is not None else "unset"

    def change_terminal(self, uex: UEXCorp, use_default: bool = True):
        try:
            terminals = {t.id: t for t in uex.terminals if t.type == "commodity"}
            request = inquirer.fuzzy(
                message="Select terminal:",
                choices=[Choice(key, name=t.name_full) for key, t in terminals.items()],
                default=self.terminal.name_full if self.terminal and use_default else None,
            )
            _id = request.execute()
            self.terminal = terminals[_id]

        except KeyboardInterrupt:
            log.debug("aborting trade port change based on user input")

    def get_sus_attrs(self, item: DetectedCommodity, price: CommodityPrice, item_type: ItemType) -> list[EditTarget]:
        result = []
        match item_type:
            case ItemType.BUY:
                if not price.price_buy_min_month <= item.price <= price.price_buy_max_month:
                    result.append(EditTarget.PRICE)

                if not price.scu_buy_min_month <= item.stock <= price.scu_buy_max_month:
                    result.append(EditTarget.STOCK)

            case ItemType.SELL:
                if not price.price_sell_min_month <= item.price <= price.price_sell_max_month:
                    result.append(EditTarget.PRICE)

                if not price.scu_sell_min_month <= item.stock <= price.scu_sell_max_month:
                    result.append(EditTarget.STOCK)

        return result
