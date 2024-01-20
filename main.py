import argparse
import csv
import json
import logging.config
import pickle
from pathlib import Path
from typing import Sequence, TypedDict
from enums import *

import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import requests
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator
from PIL import ImageGrab
from rich.console import Console
from rich.table import Table

print("Preparing...")

logging_config_file = Path(__file__).parent / "config" / "logging.ini"
logging.config.fileConfig(logging_config_file.absolute())

log = logging.getLogger("KioskProbe." + __name__)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

color_prob_low = np.array([0, 0, 255], dtype=np.uint8)
color_prob_high = np.array([0, 255, 0], dtype=np.uint8)


class DependencyContainer:
    def __init__(self):
        log.debug("initializing dependency container")
        self.settings = Settings()
        self.static_data = StaticData()
        self.run_manager = DataRunManager()

        log.debug("preparing image reader service")
        self.image_reader = easyocr.Reader(["en"], gpu=True)


class Settings:
    def __init__(self):
        log.debug("loading settings")
        self.show_images = False
        self.dry_run = False


class Commodity:
    def __init__(self, name: str | None, code: str | None, price: float | None, stock: float | None):
        self.name = name
        self.code = code
        self.price = price if price is not None else float("nan")
        self.stock = stock if stock is not None else float("nan")
        self.trust = 1.0

    def __repr__(self):
        return f"Commodity({self.code}/{self.name}, {self.price:,.5f} aUEC, {self.stock:,.3f} SCU, {self.trust:.1f} trust)"

    def __str__(self):
        name = self.name if self.name is not None else "Unknown"
        return f"{name:30} {self.price:12,.2f} aUEC {self.stock:7,.0f} SCU"

    def __eq__(self, other):
        return isinstance(other, Commodity) and self.matches(other)

    def matches(self, other: "Commodity") -> bool:
        return self.code == other.code

    def merge(self, other: "Commodity") -> bool:
        if not self.matches(other) \
                or self.price != other.price \
                or self.stock != other.stock:
            return False

        self.trust += other.trust
        return True

    def is_valid(self):
        return self.name is not None and self.code is not None and self.price >= 0 and self.stock >= 0


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


class DataRunManager:
    buy: list[Commodity]
    sell: list[Commodity]

    previous_buy_by_code: dict[str, Commodity]
    previous_sell_by_code: dict[str, Commodity]

    def __init__(self):
        log.debug("initializing data run manager")
        self.buy = []
        self.sell = []
        self.previous_buy_by_code = {}
        self.previous_sell_by_code = {}
        self.trade_port: StaticData.TradePort | None = None

    def add_buy(self, commodity: Commodity):
        self.buy.append(commodity)
        self.buy = self.merge_duplicates(self.buy)
        log.debug("current buy list: %s", self.buy)

    def extend_buy(self, items: Sequence[Commodity]):
        log.debug("extending buy list with: %s", items)
        self.buy.extend(items)
        self.buy = self.merge_duplicates(self.buy)
        log.debug("current buy list: %s", self.buy)

    def add_sell(self, commodity: Commodity):
        self.sell.append(commodity)
        self.sell = self.merge_duplicates(self.sell)
        log.debug("current sell list: %s", self.sell)

    def extend_sell(self, items: Sequence[Commodity]):
        log.debug("extending sell list with: %s", items)
        self.sell.extend(items)
        self.sell = self.merge_duplicates(self.sell)
        log.debug("current sell list: %s", self.sell)

    def clear(self):
        log.debug("clearing current data run")
        self.buy.clear()
        self.sell.clear()

    def is_dirty(self):
        return len(self.buy) > 0 or len(self.sell) > 0

    def merge_duplicates(self, items: Sequence[Commodity]) -> list[Commodity]:
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
            if item.code not in result_items or item.trust > result_items[item.code].trust:
                result_items[item.code] = item

        return list(result_items.values())

    def item_overview(self, items: Sequence[Commodity] | None, print_results: bool = True, prefix: str = "Detected")\
            -> tuple[list[Commodity], list[Commodity]]:
        items = items if items is not None else self.buy + self.sell

        items_valid = list(filter(lambda _item: _item.is_valid(), items))
        items_invalid = list(filter(lambda _item: not _item.is_valid(), items))

        if print_results:
            print(f"{prefix} {len(items_valid)} valid item(s):")
            for item in sorted(items_valid, key=lambda i: i.name):
                print(f"\t{item}", end="")

                previous_trade = self.previous_buy_by_code.get(item.code) or self.previous_sell_by_code.get(item.code)

                print(" " * 4, end="")
                if previous_trade is not None:
                    self.print_percentage_diff(item.price, previous_trade.price, suffix=" aUEC")
                    print(" " * 2, end="")
                    self.print_percentage_diff(item.stock, previous_trade.stock, suffix=" SCU")

                elif self.trade_port is not None:
                    print(f"+ NEW at {self.current_trade_port_name()}", end="")

                print()

            if len(items_invalid) > 0:
                print(f"\n{prefix} {len(items_invalid)} invalid item(s):")
                for item in sorted(items_invalid, key=lambda i: i.name or "zzz"):
                    print(f"\t{item}")

            print()

        return items_valid, items_invalid

    def print_percentage_diff(self, value_current: float, value_previous: float, suffix: str = ""):
        try:
            price_pct_diff = (value_current - value_previous) / value_previous * 100
            if price_pct_diff == 0.0:
                raise ArithmeticError

            prefix = "⌃ " if price_pct_diff > 0 else "⌄ "
            print(f"{prefix}{price_pct_diff:+5.1f}%{suffix}", end="")

        except ArithmeticError:
            print(" " * (9 + len(suffix)), end="")

    def sync_item_changes(self):
        log.debug("syncing item changes")
        self.buy = list(self.filter_untrusted(self.buy))
        self.sell = list(self.filter_untrusted(self.sell))

    def filter_untrusted(self, items: Sequence[Commodity]) -> list[Commodity]:
        for item in items:
            if item.trust <= 0.0:
                log.debug("ignoring untrusted item %s", repr(item))
                continue

            yield item

    def current_trade_port_name(self):
        return f"{port['code']}: {port['name']}" if (port := self.trade_port) is not None else None

    def change_trade_port(self, static_data: StaticData, default: bool = True):
        try:
            self.trade_port = inquirer.fuzzy(
                message="Select trade port:",
                choices=[
                    Choice(port, name=f"{port['code']}: {port['name']}") for port in static_data.trade_port.values()
                ],
                default=self.current_trade_port_name() if default else None,
            ).execute()

            self.load()

        except KeyboardInterrupt:
            log.debug("aborting trade port change based on user input")

    def load(self, update_previous: bool = True) -> dict:
        assert self.trade_port is not None, "trade port must be set"

        filepath = Path("stats.pickle")
        log.debug("loading data run from %s", filepath.absolute())
        if not filepath.exists():
            if update_previous:
                self.previous_buy_by_code = {}
                self.previous_sell_by_code = {}
                log.info("no previous data runs found")

            return {}

        with filepath.open("rb") as file:
            data = pickle.load(file)
            port_code = self.trade_port["code"]
            if update_previous:
                if port_code in data:
                    self.previous_buy_by_code = data[port_code]["buy"]
                    self.previous_sell_by_code = data[port_code]["sell"]
                    log.info("loaded %d previous buy items for trade port %s", len(self.previous_buy_by_code), port_code)
                    log.info("loaded %d previous sell items for trade port %s", len(self.previous_sell_by_code), port_code)

                else:
                    self.previous_buy_by_code = {}
                    self.previous_sell_by_code = {}
                    log.info("no previous data runs found for trade port %s", self.current_trade_port_name())

        return data

    def store(self):
        assert self.trade_port is not None, "trade port must be set"

        filepath = Path("stats.pickle")
        log.debug("storing data run to %s", filepath.absolute())

        data = self.load(update_previous=False)
        with filepath.open("wb") as file:
            self.previous_buy_by_code = {i.code: i for i in self.buy}
            self.previous_sell_by_code = {i.code: i for i in self.sell}
            data[self.trade_port["code"]] = {
                "buy": self.previous_buy_by_code,
                "sell": self.previous_sell_by_code,
            }
            pickle.dump(data, file)


deps = DependencyContainer()


class TextNode:
    left: "TextNode" = None
    right: "TextNode" = None
    top: "TextNode" = None
    bottom: "TextNode" = None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._value = value
        self.text = str(value)

    @property
    def boundary_min_y(self):
        return self.bounds[:, 1].min()

    @property
    def boundary_max_y(self):
        return self.bounds[:, 1].max()

    @property
    def boundary_min_x(self):
        return self.bounds[:, 0].min()

    @property
    def boundary_max_x(self):
        return self.bounds[:, 0].max()

    def __init__(self, value, bounds: np.array, probability: float):
        self.type = NodeType.UNKNOWN
        self.value = value
        self.bounds = bounds
        self.probability = probability
        self.boundary_center: np.array = sum(bounds, start=np.array([0, 0], dtype=np.int32)) // len(bounds)
        self.merged: bool = False

    def __repr__(self):
        return f"TextNode({self.text}, {self.boundary_center})"

    def __hash__(self):
        return hash(self.text + str(self.bounds))

    def display_link(self, node: "TextNode", image: np.ndarray, color=(0, 0, 255)):
        if node is not None:
            cv2.line(image, self.boundary_center, node.boundary_center, color, 1)
            node.display(image)

    def display(self, image: np.ndarray):
        boundary_coords = np.array(self.bounds, dtype=np.int32)
        boundary_color = color_prob_high * self.probability + color_prob_low * (1 - self.probability)

        cv2.polylines(image, [boundary_coords], True, boundary_color, 1)

        font = cv2.FONT_HERSHEY_COMPLEX
        top_left, top_right, bottom_right, bottom_left = boundary_coords
        font_scale = .5
        font_thickness = 1
        text_pos = top_left
        cv2.putText(image, self.text, text_pos, font, font_scale, (0, 255, 0), font_thickness)
        cv2.drawMarker(image, self.boundary_center, (0, 0, 255), cv2.MARKER_CROSS, 10, 1)

        # cv2.line(image, (0, self.boundary_min_y), (image.shape[0], self.boundary_min_y), (0, 0, 255), 1)
        # cv2.line(image, (0, self.boundary_max_y), (image.shape[0], self.boundary_max_y), (0, 0, 255), 1)

        if True:
            # yellow
            self.display_link(self.top, image, (0, 255, 255))
            self.display_link(self.bottom, image, (0, 255, 255))
            # turquoise
            self.display_link(self.right, image, (255, 255, 0))
            self.display_link(self.left, image, (255, 255, 0))

    def distance_to(self, node: "TextNode"):
        return np.linalg.norm(self.boundary_center - node.boundary_center)

    def filter_within_column(self, nodes: Sequence["TextNode"], min_x=None, max_x=None, tolerance=.025):
        min_x = (min_x or self.boundary_min_x) * (1.0 - tolerance)
        max_x = (max_x or self.boundary_max_x) * (1.0 + tolerance)
        # select only nodes with boundary center within min and max X bounds
        return filter(lambda node: min_x <= node.boundary_center[0] <= max_x, nodes)

    def filter_within_row(self, nodes: Sequence["TextNode"], min_y=None, max_y=None, tolerance=.025):
        min_y = (min_y or self.boundary_min_y) * (1.0 - tolerance)
        max_y = (max_y or self.boundary_max_y) * (1.0 + tolerance)
        # select only nodes with boundary center within min and max Y bounds
        return filter(lambda node: min_y <= node.boundary_center[1] <= max_y, nodes)

    def sort_by_distance(self, nodes: Sequence["TextNode"], max_distance: int = None):
        nodes = sorted(nodes, key=lambda node: self.distance_to(node))
        if max_distance is not None:
            nodes = filter(lambda node: self.distance_to(node) <= max_distance, nodes)

        return nodes

    def find_and_connect_bottom(self, nodes: Sequence["TextNode"], node_type: NodeType = None):
        if node_type is not None:
            nodes = filter_by_type(nodes, node_type)

        nodes = self.filter_within_column(nodes)
        nodes = filter_by_direction(nodes, self, Direction.DOWN)
        nodes = self.sort_by_distance(nodes)
        if len(nodes):
            closest_node = nodes[0]
            self.bottom = closest_node

    def find_and_connect_left(self, nodes: Sequence["TextNode"], node_type: NodeType = None):
        if node_type is not None:
            nodes = filter_by_type(nodes, node_type)

        nodes = self.filter_within_row(nodes)
        nodes = filter_by_direction(nodes, self, Direction.LEFT)
        nodes = filter(lambda node: node is not self.bottom, nodes)
        nodes = self.sort_by_distance(nodes)
        if len(nodes):
            closest_node = nodes[0]
            self.left = closest_node


def filter_by_type(nodes: Sequence["TextNode"], node_type: NodeType):
    return filter(lambda node: node.type == node_type, nodes)


def filter_by_direction(nodes: Sequence["TextNode"], source: "TextNode", direction: Direction):
    if direction == Direction.UP:
        return filter(lambda node: node.boundary_center[1] < source.boundary_center[1], nodes)

    elif direction == Direction.RIGHT:
        return filter(lambda node: node.boundary_center[0] > source.boundary_center[0], nodes)

    elif direction == Direction.DOWN:
        return filter(lambda node: node.boundary_center[1] > source.boundary_center[1], nodes)

    elif direction == Direction.LEFT:
        return filter(lambda node: node.boundary_center[0] < source.boundary_center[0], nodes)

    else:
        raise ValueError(f"unknown direction {direction}")


def levenshtein(source, target):
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
            current_row[1:],
            np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
            current_row[1:],
            current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]


def convert_node_type(node: TextNode, static_data: StaticData):
    if levenshtein(node.text[-3:], "SCU") <= 1:
        from string import ascii_letters
        value = node.text[:-3].strip().strip(ascii_letters)
        value = value.replace("O", "0")
        value = value.replace(",", "")
        value = value.replace(" ", "")
        value = float(value)
        if node.text.endswith("cSCU"):
            value /= 100
        elif node.text.endswith("uSCU"):
            value /= 1_000_000

        node.value = float(value)
        node.type = NodeType.COMMODITY_STOCK

    elif levenshtein(node.text[-5:], "/UNIT") <= 2:
        price = node.text.replace(" ", "")
        price = price.replace("O", "0")
        price = price.replace(",", ".")
        price = price[1:-5]
        price = float(price[:-1]) * 1000 if price.endswith("K") else float(price)

        node.value = price
        node.type = NodeType.COMMODITY_PRICE

    else:
        commodities = static_data.commodity_by_name.values()
        commodity_name_matching = map(lambda c: levenshtein(c["name"].upper()[:16], node.text), commodities)
        commodity_name_pairs = zip(commodities, commodity_name_matching)
        commodity_match, distance = min(commodity_name_pairs, key=lambda x: x[1])
        distance_max = len(commodity_match["name"]) * 0.33
        if distance <= distance_max:
            node.type = NodeType.COMMODITY_NAME
            node.value = commodity_match["name"]

        else:
            log.debug(f"unmatched commodity '%s', best guess '%s' (distance %d/%d)", node.text, commodity_match["name"],
                      distance, distance_max)


def process_image(image: np.ndarray, reader: easyocr.Reader, static_data: StaticData) \
        -> tuple[list[Commodity], np.ndarray]:
    if image is None or len(image.shape) != 3:
        log.error("failed to read image for processing, shape %s", image.shape if image is not None else None)
        return [], np.array([])

    print("Processing image...")
    log.info("processing image, shape %s", image.shape)
    texts = reader.readtext(
        image,
        # general
        decoder='greedy',
        batch_size=8,
        # contrast
        contrast_ths=0.1,
        adjust_contrast=0.5,
        # text detection
        text_threshold=0.5,
        low_text=0.25,
        link_threshold=0.4,
        canvas_size=max(image.shape[:2]),
        # bounding box merging
        slope_ths=0.15,
        ycenter_ths=0.30,
        height_ths=0.9,
        width_ths=0.9,
    )

    inventory_cutoff_coords = np.array([0, 0])
    for (bounds, text, _) in texts:
        text = text.upper()
        if levenshtein(text, "SHOP INVENTORY") <= 2:
            inventory_cutoff_coords = bounds[0]
            break

    texts_filtered = []
    for t in texts:
        bounds, text, prob = t
        boundary_x, boundary_y = bounds[BoundaryPosition.BOTTOM_RIGHT.value]
        if boundary_x < inventory_cutoff_coords[0] or boundary_y < inventory_cutoff_coords[1]:
            log.debug("ignoring text '%s', below inventory cutoff", text)
            continue

        texts_filtered.append(t)

    nodes = set()
    for (bounds, text, prob) in texts_filtered:
        try:
            node = TextNode(text.upper(), np.array(bounds, dtype=np.int32), prob)
            convert_node_type(node, static_data)
            nodes.add(node)

        except Exception as e:
            log.warning(f"failed to parse node {text}: {e}", exc_info=e)
            continue

    items = []
    for stock_node in filter_by_type(list(nodes), NodeType.COMMODITY_STOCK):
        stock_node.find_and_connect_bottom(nodes, node_type=NodeType.COMMODITY_PRICE)
        stock_node.find_and_connect_left(nodes, node_type=NodeType.COMMODITY_NAME)
        name_node = stock_node.left
        price_node = stock_node.bottom

        commodity_name = None
        commodity_code = None
        commodity_price = price_node.value if price_node is not None else None
        commodity_stock = stock_node.value if stock_node is not None else None
        if name_node is not None:
            if (commodity := static_data.commodity_by_name.get(name_node.value)) is not None:
                commodity_name = commodity["name"]
                commodity_code = commodity["code"]

            else:
                log.error("could not find commodity for node: %s", name_node)

        result_item = Commodity(
            name=commodity_name,
            code=commodity_code,
            price=commodity_price,
            stock=commodity_stock,
        )
        items.append(result_item)

        stock_node.display(image)

    min_coordinate = (np.inf, np.inf)
    max_coordinate = (-np.inf, -np.inf)
    for node in filter(lambda n: n.type != NodeType.UNKNOWN, nodes):
        for boundary_coord in node.bounds:
            min_coordinate = np.minimum(min_coordinate, boundary_coord)
            max_coordinate = np.maximum(max_coordinate, boundary_coord)

    min_x, min_y = int(min_coordinate[0]), int(min_coordinate[1])
    max_x, max_y = int(max_coordinate[0]), int(max_coordinate[1])
    image = image[min_y:max_y, min_x:max_x]

    if deps.settings.show_images:
        image_converted = cv2.cvtColor(image[min_y:max_y, min_x:max_x], cv2.COLOR_BGR2RGB)
        plt.imshow(image_converted)
        plt.show()

    return items, image


def edit_items(items: Sequence[Commodity], deps: DependencyContainer) -> Sequence[Commodity]:
    while True:
        items = list(sorted(items, key=lambda item: item.name or "zzz"))
        choices_invalid = [Choice(_id, name=str(item)) for _id, item in enumerate(items) if not item.is_valid()]
        choices_valid = [Choice(_id, name=str(item)) for _id, item in enumerate(items) if item.is_valid()]
        item_index: int | None = inquirer.select(
            message="Select an item to edit:",
            choices=[
                # invalid items
                *([Choice(EditAction.FINALIZE, name="Discard invalid items"), Separator()] if len(choices_invalid) else []),
                *choices_invalid,
                # valid items
                *([Separator()] if len(choices_invalid) and len(choices_valid) else []),
                *choices_valid,
                # other options
                Separator(),
                *([Choice(EditAction.FINALIZE, name="Finish editing")] if len(choices_invalid) == 0 else []),
                *([Choice(EditAction.DISCARD_ALL, name="Discard all")] if len(items) else []),
            ],
            default=EditAction.FINALIZE,
        ).execute()

        try:
            match item_index:
                case EditAction.FINALIZE:
                    for item in items:
                        if not item.is_valid():
                            item.trust = 0

                    break

                case EditAction.DISCARD_ALL:
                    for item in items:
                        item.trust = 0

                    break

            fix_target: EditTarget = inquirer.select(
                message="Select an attribute to fix:",
                choices=[
                    Choice(EditTarget.DISCARD, name="Discard item"),
                    Separator(),
                    Choice(EditTarget.ALL, name="Fix all"),
                    Choice(EditTarget.NAME, name="Change commodity"),
                    Choice(EditTarget.PRICE, name="Change price"),
                    Choice(EditTarget.STOCK, name="Change stock"),
                ],
                default=EditTarget.NAME,
            ).execute()

            item = items[item_index]
            match fix_target:
                case EditTarget.DISCARD:
                    item.trust = 0

                case EditTarget.ALL:
                    edit_item_name(deps.static_data, item)
                    edit_item_price(item)
                    edit_item_stock(item)
                    item.trust = float("inf")

                case EditTarget.NAME:
                    edit_item_name(deps.static_data, item)
                    item.trust = float("inf")

                case EditTarget.PRICE:
                    edit_item_price(item)
                    item.trust = float("inf")

                case EditTarget.STOCK:
                    edit_item_stock(item)
                    item.trust = float("inf")

        except Exception as e:
            log.exception(f"failed to process changes to item: {e}", exc_info=e)
            continue

        except KeyboardInterrupt:
            log.debug("aborting fix based on user input")
            continue

        finally:
            items = deps.run_manager.filter_untrusted(items)
            deps.run_manager.sync_item_changes()

    return deps.run_manager.filter_untrusted(items)


def edit_item_stock(item):
    item.stock = inquirer.number(
        message="Enter stock:",
        float_allowed=True,
        min_allowed=0.0,
        filter=lambda result: float(result),
        default=item.stock,
        transformer=lambda result: "%s SCU" % result,
    ).execute()


def edit_item_price(item):
    item.price = inquirer.number(
        message="Enter price:",
        float_allowed=True,
        min_allowed=0.0,
        filter=lambda result: float(result),
        default=item.price,
        transformer=lambda result: "%s aUEC" % result,
    ).execute()


def edit_item_name(static_data: StaticData, item: Commodity):
    commodity: StaticData.Commodity = inquirer.fuzzy(
        message="Select commodity:",
        choices=[
            Choice(commodity, name=f"{commodity['code']}: {commodity['name']}") for commodity in
            static_data.commodity_by_code.values()
        ],
        default=item.code if item.code is not None else None,
    ).execute()
    item.name = commodity["name"]
    item.code = commodity["code"]


def run(action: Action, deps: DependencyContainer):
    reader = deps.image_reader
    static_data = deps.static_data
    run_manager = deps.run_manager

    try:
        clipboard_image = ImageGrab.grabclipboard()
        image = np.array(clipboard_image)
        if len(image.shape) != 3:
            print("There is no image in the clipboard.")
            return

    except Exception as e:
        log.exception(f"failed to grab clipboard image: {e}", exc_info=e)
        return

    try:
        items, _ = process_image(image, reader, static_data)
        log.debug(f"detected items: {items}")

    except Exception as e:
        log.exception(f"failed to process image: {e}", exc_info=e)
        return

    try:
        print()
        deps.run_manager.item_overview(items)
        edit_items(items, deps)

        # items were fixed through references, it is safe to work with all items
        result_items = list(filter(lambda _item: _item.is_valid(), items))
        match action:
            case Action.PROCESS_BUY:
                run_manager.extend_buy(result_items)

            case Action.PROCESS_SELL:
                run_manager.extend_sell(result_items)

            case _:
                log.error("unknown action %s", action)

    except Exception as e:
        log.exception(f"failed to process detected items: {e}", exc_info=e)
        return


def commit(deps: DependencyContainer):
    static_data = deps.static_data
    run_manager = deps.run_manager

    if not run_manager.is_dirty():
        print("There are no data run changes to commit.")
        log.info("no changes to commit")
        return

    session_file = Path(__file__).parent / "config" / "session.json"
    with session_file.open() as file:
        session_cookies = json.load(file)

    if not session_cookies:
        log.error("no UEX user session found, copy the 'PHPSESSID' cookie value from your browser to %s", session_file.absolute())
        return

    run_manager.change_trade_port(static_data)

    trade_port_code = run_manager.trade_port["code"]
    trade_port_system = run_manager.trade_port["system_code"]
    run_data = {
        "tradeport": trade_port_code,
        "system": trade_port_system,
        "id_user_helper": "",
        "screenshot": "",
        "return_to_page": 0,
        "details": "",
        "new_buy[]": "",
        "new_sell[]": "",
        "new_commodity[]": "",
        **{f"sell[{item.code}]": item.price for item in run_manager.sell},
        **{f"scu_sell[{item.code}]": item.stock for item in run_manager.sell},
        **{f"buy[{item.code}]": item.price for item in run_manager.buy},
        **{f"scu_buy[{item.code}]": item.stock for item in run_manager.buy},
    }

    def confirm_submission() -> bool:
        print()
        run_manager.item_overview(run_manager.buy, prefix="Tracked BUY contains")
        run_manager.item_overview(run_manager.sell, prefix="Tracked SELL contains")
        return inquirer.confirm(message="Proceed with submission?", default=True).execute()

    while not confirm_submission():
        action: CommitRejectAction = inquirer.select(
            message="Select an action:",
            choices=[
                Choice(CommitRejectAction.ABORT, name="Abort submission"),
                Choice(CommitRejectAction.CONTINUE, name="Continue with submission"),
                Separator(),
                Choice(CommitRejectAction.EDIT, name="EDIT entries"),
                Choice(CommitRejectAction.DISCARD, name="DISCARD selected entries"),
                Choice(CommitRejectAction.CLEAR, name="CLEAR all entries"),
            ],
            default=CommitRejectAction.CONTINUE,
        ).execute()
        match action:
            case CommitRejectAction.EDIT:
                edit_items(run_manager.buy + run_manager.sell, deps)

            case CommitRejectAction.DISCARD:
                entries = {
                    **{("buy", item.code): item for item in run_manager.buy},
                    **{("sell", item.code): item for item in run_manager.sell},
                }
                choices_buy = [Choice(key, name=str(item)) for key, item in entries.items() if key[0] == "buy"]
                choices_sell = [Choice(key, name=str(item)) for key, item in entries.items() if key[0] == "sell"]
                keys_to_discard: list[tuple[str, str]] = inquirer.select(
                    message="Select entries to discard:",
                    choices=[
                        *choices_buy,
                        *([Separator()] if len(choices_buy) and len(choices_sell) else []),
                        *choices_sell,
                    ],
                    multiselect=True,
                ).execute()
                for entry_key in keys_to_discard:
                    entries.get(entry_key).trust = 0

                run_manager.sync_item_changes()

            case CommitRejectAction.CLEAR:
                run_manager.clear()

            case CommitRejectAction.ABORT:
                log.debug("aborting submission based on user input")
                return

        if not run_manager.is_dirty():
            print("There are no data run changes to commit anymore.")
            log.info("no changes to commit anymore")
            return

    try:
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Origin": "https://portal.uexcorp.space",
            "Referer": f"https://portal.uexcorp.space/dataruns/submit/system/{trade_port_system}/tradeport/{trade_port_code}/",
        }
        if not deps.settings.dry_run:
            response = requests.post(
                "https://portal.uexcorp.space/dataruns/submit",
                data=run_data,
                cookies=session_cookies,
                allow_redirects=False,
                headers=headers,
            )
            log.debug("submission response: %s", response.text)
            response.raise_for_status()

        log.info("successfully submitted data run (dry run: %s)", deps.settings.dry_run)
        run_manager.store()
        run_manager.clear()
        print("Data successfully submitted:", headers["Referer"])
        print()

    except Exception as e:
        log.exception(f"failed to submit data run: {e}", exc_info=e)
        return


def run_choices():
    while True:
        action_prompt = inquirer.select(
            message="Select an action:",
            choices=[
                Choice(Action.CHANGE_TRADE_PORT, name=f"Change trade port ({deps.run_manager.current_trade_port_name()})"),
                Choice(Action.PROCESS_BUY, name="Process BUY screenshot"),
                Choice(Action.PROCESS_SELL, name="Process SELL screenshot"),
                *([
                    Separator(),
                    Choice(Action.COMMIT, name="COMMIT data run", enabled=deps.run_manager.is_dirty()),
                    Choice(Action.CLEAR, name="CLEAR data run", enabled=deps.run_manager.is_dirty()),
                ] if deps.run_manager.is_dirty() else []),
                Separator(),
                Choice(None, name="Exit"),
            ],
            long_instruction="b=Process BUY, s=Process SELL, f=COMMIT, c=CLEAR, p=Change port, q=EXIT",
            default=Action.CHANGE_TRADE_PORT if deps.run_manager.trade_port is None
            else Action.PROCESS_BUY if not deps.run_manager.is_dirty()
            else Action.PROCESS_SELL,
        )

        @action_prompt.register_kb("b")
        def _handle_process_buy(event): event.app.exit(result=Action.PROCESS_BUY)

        @action_prompt.register_kb("s")
        def _handle_process_sell(event): event.app.exit(result=Action.PROCESS_SELL)

        @action_prompt.register_kb("f", filter=deps.run_manager.is_dirty())
        def _handle_commit(event): event.app.exit(result=Action.COMMIT)

        @action_prompt.register_kb("c", filter=deps.run_manager.is_dirty())
        def _handle_clear(event): event.app.exit(result=Action.CLEAR)

        @action_prompt.register_kb("p")
        def _handle_change_port(event): event.app.exit(result=Action.CHANGE_TRADE_PORT)

        @action_prompt.register_kb("q")
        def _handle_exit(event): event.app.exit(result=None)

        action = action_prompt.execute()
        match action:
            case Action.CHANGE_TRADE_PORT:
                deps.run_manager.change_trade_port(deps.static_data, default=False)

            case Action.PROCESS_BUY | Action.PROCESS_SELL:
                run(action, deps)

            case Action.COMMIT:
                commit(deps)

            case Action.CLEAR:
                deps.run_manager.clear()

            case _:
                break


class BenchmarkResults(TypedDict):
    item_count: int
    valid_item_count: int
    detected_name_count: int
    detected_price_count: int
    detected_stock_count: int
    data_names: list[str | None]
    data_prices: list[float | None]
    data_stocks: list[float | None]


def run_benchmark(images_dir: Path = None):
    images_dir = images_dir or Path(__file__).parent / "images"
    log.info("running benchmark on images in %s", images_dir.absolute())

    all_previous_results: dict[str, BenchmarkResults] = {}
    all_current_results: dict[str, BenchmarkResults] = {}

    for image_path in images_dir.glob("*.jpg"):
        image = cv2.imread(image_path.absolute().as_posix())
        items, image = process_image(image, deps.image_reader, deps.static_data)
        items = list(sorted(items, key=lambda item: item.name or "zzz"))

        previous_results: BenchmarkResults | None = None
        results_filepath = image_path.with_name(f"{image_path.stem}-results.json")
        if results_filepath.exists():
            with results_filepath.open("r") as file:
                previous_results = json.load(file)

        results: BenchmarkResults = {
            "item_count": len(items),
            "valid_item_count": len(list(filter(lambda i: i.is_valid(), items))),
            "detected_name_count": len(list(filter(lambda i: i.name is not None, items))),
            "detected_price_count": len(list(filter(lambda i: i.price is not None, items))),
            "detected_stock_count": len(list(filter(lambda i: i.stock is not None, items))),
            "data_names": [i.name for i in items],
            "data_prices": [i.price for i in items],
            "data_stocks": [i.stock for i in items],
        }

        processed_image_path = image_path.with_name(f"{image_path.stem}-processed.png")
        if not processed_image_path.exists():
            log.debug("saving processed image to %s", processed_image_path.absolute())
            cv2.imwrite(processed_image_path.absolute().as_posix(), image)

        if previous_results is None:
            previous_results = results
            log.debug("saving new results to %s", results_filepath.absolute())
            with results_filepath.open("w") as file:
                json.dump(results, file, indent=4)

        all_previous_results[image_path.name] = previous_results
        all_current_results[image_path.name] = results

    # print result percentages in a table using fixed width columns
    print()
    table = Table(title="Results")
    columns = ["image_name", "items", "valid_items", "names", "prices", "stocks", "matching_names", "matching_prices", "matching_stocks"]
    success_measure_key = "valid_item_count"
    for column in columns:
        table.add_column(column)

    for key, current_results in all_current_results.items():
        previous_results = all_previous_results[key]
        max_values = {
            "valid_item_count": previous_results["item_count"],
            "detected_name_count": previous_results["item_count"],
            "detected_price_count": previous_results["item_count"],
            "detected_stock_count": previous_results["item_count"],
        }
        row = [key]
        success_measure = 1.0
        for attribute_key in current_results.keys():
            if attribute_key.startswith("data_"):
                continue

            max_value = max_values.get(attribute_key)
            attribute_current = current_results[attribute_key]
            attribute_previous = previous_results[attribute_key]

            pct = attribute_current / max_value * 100 if max_value else 0
            pct_value = f"{pct:5.1f}%  " if pct > 0 else ""
            pct_diff = (attribute_current - attribute_previous) / attribute_previous * 100 if attribute_previous else float("inf")

            row.append(f"{pct_value}{pct_diff:+6.1f}%")

            if attribute_key == success_measure_key:
                success_measure = pct_diff

        for attribute_key in current_results.keys():
            if not attribute_key.startswith("data_"):
                continue

            attribute_current = current_results[attribute_key]
            attribute_previous = previous_results[attribute_key]
            value_pairs = list(zip(attribute_current, attribute_previous))
            pairs_equal = map(lambda pair: pair[0] == pair[1], value_pairs)

            pct = sum(pairs_equal) / len(value_pairs) * 100 if len(value_pairs) else 0
            pct_value = f"{pct:5.1f}%  " if pct > 0 else ""
            pct_diff = pct - 100.0

            row.append(f"{pct_value}{pct_diff:+6.1f}%")

        row_style = "bright green" if success_measure > 0.0 else "dim" if success_measure == 0.0 else "red"
        table.add_row(*row, style=row_style)

    console = Console()
    console.print(table)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="run benchmark")
    parser.add_argument("--show-images", action="store_true", help="show processed images")
    parser.add_argument("--dry-run", action="store_true", help="do not submit data run")
    args = parser.parse_args()

    deps.settings.show_images = args.show_images
    deps.settings.dry_run = args.dry_run

    try:
        if args.benchmark:
            run_benchmark()
            exit(0)

        run_choices()

    except KeyboardInterrupt:
        log.debug("exiting based on user input")
        pass

    except Exception as e:
        log.exception(f"an unrecoverable error has occurred: {e}", exc_info=e)
        raise e
