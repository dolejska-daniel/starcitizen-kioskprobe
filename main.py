import argparse
import json
import logging.config
import math
from dataclasses import asdict
from pathlib import Path
from typing import Sequence, TypedDict, Callable, Any

import cv2
import easyocr
import matplotlib.pyplot as plt
import numpy as np
import requests
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator
from InquirerPy.validator import NumberValidator
from PIL import ImageGrab
from prompt_toolkit.validation import ValidationError
from rich.console import Console
from rich.table import Table

from datarun import DetectedCommodity, DataRunManager
from enums import *
from kiosk_probe.uexcorp.api import UEXCorp
from kiosk_probe.uexcorp.objects import DataRun, DataRunCommodityBuyEntry, DataRunCommoditySellEntry, Commodity
from settings import Settings
from static_data import StaticData
from utils import filter_by_type, levenshtein, find_best_string_match, TextNode

print("Preparing...")

logging_config_file = Path(__file__).parent / "config" / "logging.ini"
logging.config.fileConfig(logging_config_file.absolute())

log = logging.getLogger("kiosk_probe." + __name__)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class DependencyContainer:
    def __init__(self):
        log.debug("initializing dependency container")
        self.settings = Settings()
        self.static_data = StaticData()
        self.uexcorp = UEXCorp(self.settings.uexcorp_api_config)
        self.run_manager = DataRunManager(self.uexcorp)

        log.debug("preparing image reader service")
        self.image_reader = easyocr.Reader(["en"], gpu=True)


deps = DependencyContainer()


def convert_node_type(node: TextNode, static_data: StaticData):
    from string import ascii_letters
    if levenshtein(node.text[-3:], "SCU") <= 1:
        log.debug("trying to parse commodity stock from '%s'", node.text)
        value = node.text.strip().strip(ascii_letters)
        value = value.strip("/")
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
        log.debug("parsed commodity stock: %f", value)
        return

    if levenshtein(node.text[-5:], "/UNIT") <= 2:
        log.debug("trying to parse commodity price from '%s'", node.text)
        price = node.text[:-5].strip()
        price = price.replace(" ", "")
        price = price.replace("O", "0")
        price = price.replace(",", ".")
        price = price.strip("/")
        price = float(price[:-1]) * 1000 if price.upper().endswith("K") else float(price)

        node.value = price
        node.type = NodeType.COMMODITY_PRICE
        log.debug("parsed commodity price: %f", price)
        return

    inventory_match, distance = find_best_string_match(node.text, static_data.inventory_states, key=lambda i: i.name.upper())
    distance_max = len(inventory_match.name) * 0.33
    if distance <= distance_max:
        node.type = NodeType.COMMODITY_INVENTORY
        node.value = inventory_match.name
        node.reference_object = inventory_match
        log.debug("matched inventory state: %s", inventory_match)
        return

    log.debug(
        "unmatched inventory state '%s', best guess '%s' (distance %d/%d)",
        node.text, inventory_match.name, distance, distance_max
    )

    commodities = deps.uexcorp.api.get_commodities()
    commodity_match, distance = find_best_string_match(node.text, commodities, key=lambda c: c.name.upper()[:16])
    distance_max = len(commodity_match.name) * 0.33
    if distance <= distance_max:
        node.type = NodeType.COMMODITY_NAME
        node.value = commodity_match.name
        node.reference_object = commodity_match
        log.debug("matched commodity: %s", commodity_match)
        return

    log.debug(
        f"unmatched commodity '%s', best guess '%s' (distance %d/%d)",
        node.text, commodity_match.name, distance, distance_max
    )


def process_image(image: np.ndarray, reader: easyocr.Reader, static_data: StaticData, image_name: str = "image") \
        -> list[DetectedCommodity]:
    if image is None or len(image.shape) != 3:
        log.error("failed to read image for processing, shape %s", image.shape if image is not None else None)
        return [], np.array([])

    print(f"Processing {image_name}...")
    log.info("preprocessing image")
    remove_currency_symbol(image)

    log.info("processing image, shape %s", image.shape)
    texts = reader.readtext(
        image,
        # general
        decoder="greedy",
        batch_size=8,
        # contrast
        contrast_ths=0.15,
        adjust_contrast=0.5,
        # text detection
        text_threshold=0.5,
        low_text=0.25,
        link_threshold=0.4,
        canvas_size=max(image.shape[:2]),
        # bounding box merging
        slope_ths=0.25,
        ycenter_ths=0.30,
        height_ths=0.9,
        width_ths=1.1,
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

    nodes = list()
    for (bounds, text, prob) in texts_filtered:
        try:
            node = TextNode(text.upper(), np.array(bounds, dtype=np.int32), prob)
            convert_node_type(node, static_data)
            nodes.append(node)

        except Exception as e:
            log.warning(f"failed to parse node {text}: {e}", exc_info=e)
            continue

    items = []
    used_name_nodes = set()
    for stock_node in filter_by_type(nodes, NodeType.COMMODITY_STOCK):
        stock_node.find_and_connect_bottom(nodes, node_type=NodeType.COMMODITY_PRICE)
        stock_node.find_and_connect_left(nodes, node_type=NodeType.COMMODITY_NAME)
        name_node = stock_node.left
        price_node = stock_node.bottom
        inventory_node = None
        if name_node is not None:
            name_node.find_and_connect_bottom(nodes, node_type=NodeType.COMMODITY_INVENTORY)
            inventory_node = name_node.bottom
            used_name_nodes.add(name_node)

        commodity: Commodity | None = None
        commodity_name = None
        commodity_price = price_node.value if price_node is not None else None
        commodity_stock = stock_node.value if stock_node is not None else None
        commodity_inventory = inventory_node.reference_object if inventory_node is not None else static_data.inventory_states[0] if commodity_stock == 0 else None
        if name_node is not None:
            commodity = name_node.reference_object
            commodity_name = commodity.name

        result_item = DetectedCommodity(
            name=commodity_name,
            commodity=commodity,
            price=commodity_price,
            stock=commodity_stock,
            inventory=commodity_inventory,
            position_y=stock_node.boundary_center[1],
            source_node=stock_node,
        )
        items.append(result_item)

    for name_node in filter_by_type(nodes, NodeType.COMMODITY_NAME):
        if name_node in used_name_nodes:
            continue

        name_node.find_and_connect_bottom(nodes, node_type=NodeType.COMMODITY_INVENTORY)
        inventory_node = name_node.bottom
        commodity_inventory = inventory_node.reference_object if inventory_node is not None else None

        result_item = DetectedCommodity(
            name=name_node.reference_object.name,
            commodity=name_node.reference_object,
            price=None,
            stock=None,
            inventory=commodity_inventory,
            position_y=name_node.boundary_center[1],
            source_node=name_node,
        )
        items.append(result_item)

    if deps.settings.show_all_text_nodes:
        for node in nodes:
            node.display(image)

    return items


def crop_image(image: np.ndarray, nodes: list[TextNode], margin: int = 15) -> np.ndarray:
    margin_vector = np.array([margin, margin])
    min_coordinate = image.shape[1], image.shape[0]
    max_coordinate = (0, 0)
    for node in filter(lambda n: n.type != NodeType.UNKNOWN, nodes):
        for boundary_coord in node.bounds:
            min_coordinate = np.minimum(min_coordinate, boundary_coord - margin_vector)
            max_coordinate = np.maximum(max_coordinate, boundary_coord + margin_vector)

    min_x, min_y = int(min_coordinate[0]), int(min_coordinate[1])
    max_x, max_y = int(max_coordinate[0]), int(max_coordinate[1])
    return image[min_y:max_y, min_x:max_x]


def remove_currency_symbol(image: np.ndarray):
    log.info("removing templated symbols")
    static_dir = Path(__file__).parent / "static" / "templates_to_remove"
    for template_file in static_dir.glob("*"):
        log.debug("matching template %s", template_file.absolute().as_posix())
        template_source = cv2.imread(template_file.absolute().as_posix())
        for scale in np.linspace(0.9, 1.1, 5):
            template = cv2.resize(template_source, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            template_height, template_width = template.shape[:2]
            image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            match = cv2.matchTemplate(image=image_grey, templ=template_gray, method=cv2.TM_CCOEFF_NORMED)

            (y_points, x_points) = np.where(match >= .85)
            for (x, y) in zip(x_points, y_points):
                log.debug("removing matched template at (%d, %d)", x, y)
                color = [int(c) for c in image[y, x - 1]]
                cv2.rectangle(image, (x, y), (x + template_width, y + template_height), color, -1)


def edit_items(items: Sequence[DetectedCommodity], deps: DependencyContainer, item_type: ItemType = ItemType.UNDEFINED, before_select: Callable[[], Any] | None = None, sus_attrs: Callable[[DetectedCommodity], list[EditTarget]] | None = None) -> Sequence[DetectedCommodity]:
    while True:
        if before_select:
            before_select()

        items = list(sorted(items, key=lambda item: item.position_y))
        choices_invalid = [Choice(_id, name=str(item)) for _id, item in enumerate(items) if not item.is_valid()]
        choices_valid = [Choice(_id, name=str(item)) for _id, item in enumerate(items) if item.is_valid()]
        item_index: int | None = inquirer.select(
            message="Select an item to edit:",
            choices=[
                # invalid items
                Choice(EditAction.ADD_NEW, name="ADD new item"),
                *([Choice(EditAction.FINALIZE, name="DISCARD invalid items")] if len(choices_invalid) else []),
                Separator(),
                *choices_invalid,
                # valid items
                *([Separator()] if len(choices_invalid) and len(choices_valid) else []),
                *choices_valid,
                # other options
                Separator(),
                *([Choice(EditAction.FINALIZE, name="FINISH editing")] if len(choices_invalid) == 0 else []),
                *([Choice(EditAction.DISCARD_ALL, name="DISCARD ALL")] if len(items) else []),
            ],
            default=EditAction.FINALIZE,
        ).execute()

        try:
            match item_index:
                case EditAction.ADD_NEW:
                    new_item = add_item(deps.run_manager, deps.uexcorp, item_type=item_type)
                    items.append(new_item)
                    continue

                case EditAction.FINALIZE:
                    for item in items:
                        if not item.is_valid():
                            item.trust = 0

                    break

                case EditAction.DISCARD_ALL:
                    for item in items:
                        item.trust = 0

                    break

            item = items[item_index]
            sus_list = sus_attrs(item) if sus_attrs else []

            fix_targets: list[EditTarget] = inquirer.select(
                message="Select attributes to change:",
                choices=[
                    Choice(EditTarget.NAME, name="Change COMMODITY", enabled=item.name is None or EditTarget.NAME in sus_list),
                    Choice(EditTarget.PRICE, name="Change PRICE", enabled=(not item.price <= 0 and not item.price >= 0) or EditTarget.PRICE in sus_list),
                    Choice(EditTarget.STOCK, name="Change STOCK", enabled=(not item.stock <= 0 and not item.stock >= 0) or EditTarget.STOCK in sus_list),
                    Choice(EditTarget.INVENTORY, name="Change INVENTORY", enabled=item.inventory is None or EditTarget.INVENTORY in sus_list),
                ],
                multiselect=True,
            ).execute()

            fix_targets = sorted(fix_targets, key=lambda t: t.value, reverse=True)
            while fix_targets:
                fix_target = fix_targets.pop()
                match fix_target:
                    case EditTarget.DISCARD:
                        item.trust = 0

                    case EditTarget.ALL:
                        edit_item_name(item, deps.uexcorp)
                        edit_item_price(item)
                        edit_item_stock(item)
                        edit_item_inventory(item, deps.static_data)
                        item.trust = float("inf")

                    case EditTarget.NAME:
                        edit_item_name(item, deps.uexcorp)
                        item.trust = float("inf")

                    case EditTarget.PRICE:
                        edit_item_price(item)
                        item.trust = float("inf")

                    case EditTarget.STOCK:
                        edit_item_stock(item)
                        item.trust = float("inf")

                    case EditTarget.INVENTORY:
                        edit_item_inventory(item, deps.static_data)
                        item.trust = float("inf")

        except Exception as e:
            log.exception(f"failed to process changes to item: {e}", exc_info=e)
            continue

        except KeyboardInterrupt:
            log.debug("aborting fix based on user input")
            continue

        finally:
            deps.run_manager.sync_item_changes()
            items = deps.run_manager.filter_untrusted(items)

    deps.run_manager.sync_item_changes()
    return deps.run_manager.filter_untrusted(items)


class FloatValidator(NumberValidator):
    def __init__(self, min_value: float = float("-inf"), max_value: float = float("inf")):
        super().__init__(float_allowed=True)
        self.min_value = min_value
        self.max_value = max_value

    def validate(self, document):
        super().validate(document)
        value = float(document.text)
        if not value >= self.min_value:
            raise ValidationError(
                message=f"Value must be >= {self.min_value}",
                cursor_position=document.cursor_position
            )

        if not value <= self.max_value:
            raise ValidationError(
                message=f"Value must be <= {self.min_value}",
                cursor_position=document.cursor_position
            )


def edit_item_stock(item):
    item.stock = inquirer.text(
        message="Enter stock:",
        validate=FloatValidator(min_value=0.0),
        filter=lambda result: float(result),
        default="",
        transformer=lambda result: "%s SCU" % result,
    ).execute()


def edit_item_price(item):
    item.price = inquirer.text(
        message="Enter price:",
        validate=FloatValidator(min_value=0.0),
        filter=lambda result: float(result),
        default="",
        transformer=lambda result: "%s aUEC" % result,
    ).execute()


def edit_item_name(item: DetectedCommodity, uexcorp: UEXCorp):
    commodity_id: int = inquirer.fuzzy(
        message="Select commodity:",
        choices=[Choice(c.id, name=f"{c.code}: {c.name}") for c in uexcorp.commodities],
        default=item.commodity.code if item.commodity is not None else None,
    ).execute()
    commodity = uexcorp.get_commodity_by_id(commodity_id)
    item.name = commodity.name
    item.commodity = commodity


def edit_item_inventory(item: DetectedCommodity, static_data: StaticData):
    inventory_status_index: int = inquirer.fuzzy(
        message="Select inventory status:",
        choices=[Choice(_id, name=f"{s.name:>19}") for _id, s in enumerate(static_data.inventory_states) if s.visible],
        default=item.inventory.name if item.inventory is not None else None,
    ).execute()
    item.inventory = static_data.inventory_states[inventory_status_index]


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
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        items = process_image(image, reader, static_data, image_name="clipboard screenshot")
        log.debug(f"detected items: {items}")

    except Exception as e:
        log.exception(f"failed to process image: {e}", exc_info=e)
        return

    try:
        match action:
            case Action.PROCESS_BUY:
                items_type = ItemType.BUY
            case Action.PROCESS_SELL:
                items_type = ItemType.SELL
            case _:
                items_type = ItemType.UNDEFINED

        def load_sus_attrs(item: DetectedCommodity) -> list[EditTarget]:
            prices = deps.uexcorp.get_commodity_price_by_terminal(run_manager.terminal.id, item.commodity.id) if run_manager.terminal is not None else None
            return run_manager.get_sus_attrs(item, prices, items_type) if prices is not None else []

        items = edit_items(
            items, deps,
            item_type=items_type,
            before_select=lambda: deps.run_manager.item_overview(items, item_type=items_type),
            sus_attrs=load_sus_attrs,
        )

        # items were fixed through references, it is safe to work with all items
        result_items = [i for i in items if i.is_valid()]
        match action:
            case Action.PROCESS_BUY:
                run_manager.extend_buy(result_items)

            case Action.PROCESS_SELL:
                run_manager.extend_sell(result_items)

            case _:
                log.error("unknown action %s", action)

        if len(result_items):
            nodes_to_display = set()
            for item in result_items:
                if item.source_node is not None:
                    log.debug("displaying source nodes for item %s", item)
                    item.source_node.display(image)
                    nodes_to_display.update(item.get_all_nodes())

            image_parsed = crop_image(image, list(nodes_to_display)) if deps.settings.crop_resulting_image else image
            run_manager.add_image(image_parsed)

    except Exception as e:
        log.exception(f"failed to process detected items: {e}", exc_info=e)
        return


def add_item(run_manager: DataRunManager, uexcorp: UEXCorp, item_type: ItemType = ItemType.UNDEFINED) -> DetectedCommodity:
    if item_type == ItemType.UNDEFINED:
        item_type = inquirer.select(
            message="Select item category:",
            choices=[
                Choice(ItemType.BUY, name="BUY list"),
                Choice(ItemType.SELL, name="SELL list"),
            ],
        ).execute()

    item = DetectedCommodity(
        name=None,
        commodity=None,
        price=None,
        stock=None,
        inventory=None,
        position_y=0,
    )

    edit_item_name(item, uexcorp)
    edit_item_price(item)
    edit_item_stock(item)
    edit_item_inventory(item, deps.static_data)
    item.trust = float("inf")

    match item_type:
        case ItemType.BUY:
            run_manager.add_buy(item)
        case ItemType.SELL:
            run_manager.add_sell(item)

    return item


def commit(deps: DependencyContainer):
    run_manager = deps.run_manager

    if not run_manager.is_dirty():
        print("There are no data run changes to commit.")
        log.info("no changes to commit")
        return

    run_manager.change_terminal(deps.uexcorp)

    def confirm_submission() -> bool:
        dry_run_info = ""
        if deps.settings.dry_run:
            dry_run_info = " [DRY RUN]"

        print()
        run_manager.item_overview(run_manager.buy, prefix="Tracked BUY contains", item_type=ItemType.BUY)
        run_manager.item_overview(run_manager.sell, prefix="Tracked SELL contains", item_type=ItemType.SELL)
        return inquirer.confirm(message="Proceed with submission?" + dry_run_info, default=True).execute()

    while not confirm_submission():
        action: CommitRejectAction = inquirer.select(
            message="Select an action:",
            choices=[
                Choice(CommitRejectAction.ABORT, name="Abort submission"),
                Choice(CommitRejectAction.CONTINUE, name="Continue with submission"),
                Separator(),
                Choice(CommitRejectAction.ADD_NEW, name="ADD new entries"),
                Choice(CommitRejectAction.EDIT, name="EDIT entries"),
                Choice(CommitRejectAction.DISCARD, name="DISCARD selected entries"),
                Choice(CommitRejectAction.CLEAR, name="CLEAR all entries"),
            ],
            default=CommitRejectAction.CONTINUE,
        ).execute()
        match action:
            case CommitRejectAction.ADD_NEW:
                add_item(run_manager, deps.uexcorp)

            case CommitRejectAction.EDIT:
                edit_items(run_manager.buy + run_manager.sell, deps)

            case CommitRejectAction.DISCARD:
                entries = {
                    **{("buy", item.commodity.id): item for item in run_manager.buy},
                    **{("sell", item.commodity.id): item for item in run_manager.sell},
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

    commit_new(run_manager)


def commit_new(run_manager: DataRunManager):
    screenshot_fig, _ = merge_images(
        run_manager.images,
        [f"Screenshot {i + 1}" for i in range(len(run_manager.images))],
        figure_size_base=(5, 8),
    )
    screenshot_base64 = figure_to_base64(screenshot_fig)
    if deps.settings.include_screenshots and deps.settings.show_images:
        plt.show()

    data_run = DataRun(
        id_terminal=run_manager.terminal.id,
        prices=[
            *[buy_entry_from_commodity(item) for item in run_manager.buy],
            *[sell_entry_from_commodity(item) for item in run_manager.sell],
        ],
        is_production=False if deps.settings.dry_run else True,
        screenshot=screenshot_base64 if deps.settings.include_screenshots else None,
    )

    log.debug("submitting data run: %s", asdict(data_run, dict_factory=lambda d: {k: v for k, v in d if k != "screenshot"}))
    response = deps.uexcorp.api.submit_data_run(data_run)

    log.info("successfully submitted data run (dry run: %s)", deps.settings.dry_run)
    if deps.settings.dry_run:
        print("Real submission skipped.")
        return

    if deps.settings.show_report_links:
        table = Table(title="Data Run Reports")
        table.add_column("Commodity")
        table.add_column("Report Link")

        buy_commodities = {item.commodity.id: item.commodity for item in run_manager.buy}
        for report_id, commodity in zip(response.data.ids_reports[:len(buy_commodities)], buy_commodities.values()):
            table.add_row(
                commodity.name,
                f"https://ptu.uexcorp.space/data/info/id/{report_id}",
            )

        sell_commodities = {item.commodity.id: item.commodity for item in run_manager.sell}
        for report_id, commodity in zip(response.data.ids_reports[len(buy_commodities):], sell_commodities.values()):
            table.add_row(
                commodity.name,
                f"https://ptu.uexcorp.space/data/info/id/{report_id}",
            )

        print()
        Console().print(table)
        print()

    print("Data successfully submitted:")
    print(f" - https://ptu.uexcorp.space/data/home/type/commodity/id_terminal/{run_manager.terminal.id}/datarunner/{response.data.username}")
    print()
    run_manager.clear()


def commit_old(run_manager, session_cookies, trade_port_code, trade_port_system):
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
        **{f"sell[{item.commodity}]": item.price for item in run_manager.sell},
        **{f"scu_sell[{item.commodity}]": item.stock for item in run_manager.sell},
        **{f"buy[{item.commodity}]": item.price for item in run_manager.buy},
        **{f"scu_buy[{item.commodity}]": item.stock for item in run_manager.buy},
    }

    log.debug("submitting data run: %s", run_data)
    if deps.settings.dry_run:
        print("Real submission skipped.")
        return

    try:
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Origin": "https://portal.uexcorp.space",
            "Referer": f"https://portal.uexcorp.space/dataruns/submit/system/{trade_port_system}/tradeport/{trade_port_code}/",
        }
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
        run_manager.clear()
        print("Data successfully submitted:", headers["Referer"])
        print()

    except Exception as e:
        log.exception(f"failed to submit data run: {e}", exc_info=e)


def buy_entry_from_commodity(commodity: DetectedCommodity) -> DataRunCommodityBuyEntry:
    return DataRunCommodityBuyEntry(
        id_commodity=commodity.commodity.id,
        scu_buy=commodity.stock,
        price_buy=commodity.price,
        status_buy=commodity.inventory.value,
    )


def sell_entry_from_commodity(commodity: DetectedCommodity) -> DataRunCommoditySellEntry:
    return DataRunCommoditySellEntry(
        id_commodity=commodity.commodity.id,
        scu_sell=commodity.stock,
        price_sell=commodity.price,
        status_sell=commodity.inventory.value,
    )


def run_choices():
    while True:
        action_prompt = inquirer.select(
            message="Select an action:",
            choices=[
                Choice(Action.CHANGE_TERMINAL, name=f"Change TERMINAL ({deps.run_manager.current_terminal_name()})"),
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
            long_instruction="b=BUY, s=SELL, f=COMMIT, c=CLEAR, t=TERMINAL, q=EXIT",
            default=Action.CHANGE_TERMINAL if deps.run_manager.terminal is None
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

        @action_prompt.register_kb("t")
        def _handle_change_port(event): event.app.exit(result=Action.CHANGE_TERMINAL)

        @action_prompt.register_kb("q")
        def _handle_exit(event): event.app.exit(result=None)

        action = action_prompt.execute()
        match action:
            case Action.CHANGE_TERMINAL:
                deps.run_manager.change_terminal(deps.uexcorp, use_default=False)

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
    detected_inventory_count: int
    data_names: list[str | None]
    data_prices: list[float]
    data_stocks: list[float]
    data_inventories: list[str | None]


def merge_images(images: list[np.ndarray], titles: list[str], max_cols: int = 3, figure_size_base: tuple[int, int] = None, **kwargs) -> tuple[plt.Figure, list[plt.Axes]]:
    figure_size = figure_size_base or (5, 8)
    rows, cols = math.ceil(len(images) / float(max_cols)), min(len(images), max_cols)
    figure_size = (figure_size[0] * cols, figure_size[1] * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figure_size, **kwargs)
    if rows > 1 and cols > 1:
        axes = axes.flat
    elif rows > 1 or cols > 1:
        axes = axes
    else:
        axes = [axes]

    for _id, image in enumerate(images):
        axes[_id].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[_id].set_title(titles[_id])

    for _id in range(rows * cols):
        axes[_id].get_xaxis().set_visible(False)
        axes[_id].get_yaxis().set_visible(False)

    fig.tight_layout()
    return fig, axes


def figure_to_base64(fig: plt.Figure) -> str:
    import io
    import base64
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def run_benchmark(images_dir: Path = None, override_results: bool = False):
    images_dir = images_dir or Path(__file__).parent / "images"
    log.info("running benchmark on images in %s", images_dir.absolute())

    all_previous_results: dict[str, BenchmarkResults] = {}
    all_current_results: dict[str, BenchmarkResults] = {}

    for image_path in images_dir.glob("*.jpg"):
        image = cv2.imread(image_path.absolute().as_posix())
        items, image = process_image(image, deps.image_reader, deps.static_data, image_name=image_path.name)
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
            "detected_price_count": len(list(filter(lambda i: i.price != float("nan"), items))),
            "detected_stock_count": len(list(filter(lambda i: i.stock != float("nan"), items))),
            "detected_inventory_count": len(list(filter(lambda i: i.inventory is not None, items))),
            "data_names": [i.name for i in items],
            "data_prices": [i.price if i.price != float("nan") else "nan" for i in items],
            "data_stocks": [i.stock if i.stock != float("nan") else "nan" for i in items],
            "data_inventories": [i.inventory.name if i.inventory is not None else None for i in items],
        }

        processed_image_path = image_path.with_name(f"{image_path.stem}-processed.png")
        if not processed_image_path.exists() or override_results:
            log.debug("saving processed image to %s", processed_image_path.absolute())
            cv2.imwrite(processed_image_path.absolute().as_posix(), image)

        image_previous = cv2.imread(processed_image_path.absolute().as_posix())
        fig, axes = merge_images([image_previous, image], ["Previous", "Current"], figsize=(12, 10))
        plt.savefig(image_path.with_name(f"{image_path.stem}-comparison.png").absolute().as_posix())
        plt.close(fig)

        with results_filepath.with_name(f"{image_path.stem}-results-current.json").open("w") as file:
            json.dump(results, file, indent=4)

        if previous_results is None or override_results:
            previous_results = results
            log.debug("saving new results to %s", results_filepath.absolute())
            with results_filepath.open("w") as file:
                json.dump(results, file, indent=4)

        all_previous_results[image_path.name] = previous_results
        all_current_results[image_path.name] = results

    # print result percentages in a table using fixed width columns
    print()
    table = Table(title="Results")
    columns = ["image_name", "items", "valid_items", "names", "prices", "stocks", "inventories", "matching_names", "matching_prices", "matching_stocks", "matching_inventories"]
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
            "detected_inventory_count": previous_results["item_count"],
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
            pct_value = f"{pct:5.1f}%  " if pct > 0 else " " * 8
            pct_diff = (attribute_current - attribute_previous) / attribute_previous * 100 if attribute_previous else float("nan")

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
            pct_value = f"{pct:5.1f}%  " if pct > 0 else " " * 8
            pct_diff = pct - 100.0 if pct else float("inf")

            row.append(f"{pct_value}{pct_diff:+6.1f}%")

        row_style = "bright_green" if success_measure > 0.0 else "red" if success_measure < 0.0 else "dim"
        table.add_row(*row, style=row_style)

    console = Console()
    console.print(table)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action="store", help="run for a single image", type=Path)
    parser.add_argument("--benchmark", action="store_true", help="run benchmark")
    parser.add_argument("--benchmark-override", action="store_true", help="store benchmark results as new baseline")
    # parser.add_argument("--benchmark-glob", action="store_true", help="run benchmark")
    parser.add_argument("--show-images", action="store_true", help="show processed images")
    parser.add_argument("--show-all-nodes", action="store_true", help="show all text nodes")
    parser.add_argument("--no-crop", action="store_true", help="do not crop resulting image")
    parser.add_argument("--dry-run", action="store_true", help="do not submit data run")
    args = parser.parse_args()

    deps.settings.show_images = args.show_images
    deps.settings.dry_run = args.dry_run
    deps.settings.crop_resulting_image = args.no_crop is False
    deps.settings.show_all_text_nodes = args.show_all_nodes or args.benchmark

    try:
        if args.image:
            image = cv2.imread(args.image.absolute().as_posix())
            result, image = process_image(image, deps.image_reader, deps.static_data, image_name=args.image.name)
            print(result)
            exit(0)

        if args.benchmark:
            run_benchmark(override_results=args.benchmark_override)
            exit(0)

        run_choices()

    except KeyboardInterrupt:
        log.debug("exiting based on user input")
        pass

    except Exception as e:
        log.exception(f"an unrecoverable error has occurred: {e}", exc_info=e)
        raise e
