import logging
from dataclasses import asdict
from typing import Sequence, Callable, Any
from urllib.error import HTTPError

import cv2
import matplotlib.pyplot as plt
import numpy as np
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator
from PIL import ImageGrab
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from kiosk_probe.core.datarun.manager import DataRunManager
from kiosk_probe.core.datarun.objects import DetectedCommodity
from kiosk_probe.core.dependency_container import DependencyContainer
from kiosk_probe.core.enums import *
from kiosk_probe.core.validators import FloatValidator
from kiosk_probe.uex_corp.objects import DataRun
from kiosk_probe.core.utils import figure_to_base64, merge_images

log = logging.getLogger("kiosk_probe." + __name__)


class Controls:

    def __init__(self, deps: DependencyContainer):
        self.deps = deps

    def edit_items(self,
                   items: Sequence[DetectedCommodity],
                   item_type: ItemType = ItemType.UNDEFINED,
                   sus_attrs: Callable[[DetectedCommodity], list[EditTarget]] | None = None,
                   sort_by: Callable[[DetectedCommodity], Any] = lambda i: i.order,
                   show_items: bool = True,
                   ) -> Sequence[DetectedCommodity]:
        while True:
            if show_items:
                print()
                self.deps.run_manager.item_overview(items, item_type=item_type)

            items = list(sorted(items, key=sort_by))
            choices_invalid = [Choice(_id, name=str(item)) for _id, item in enumerate(items) if not item.is_valid()]
            choices_valid = [Choice(_id, name=str(item)) for _id, item in enumerate(items) if item.is_valid()]

            try:
                item_index: int | None = inquirer.select(
                    message="Select an item to edit:",
                    choices=[
                        # invalid items
                        Choice(EditAction.ADD_NEW, name="Manually ADD new item"),
                        *([Choice(EditAction.FINALIZE, name="DISCARD invalid items")] if len(choices_invalid) else []),
                        *([Separator()] if len(choices_invalid) else []),
                        *choices_invalid,
                        # valid items
                        *([Separator()] if len(choices_invalid) and len(choices_valid) else []),
                        *choices_valid,
                        # other options
                        Separator(),
                        *([Choice(EditAction.FINALIZE, name="FINISH editing")] if len(choices_invalid) == 0 else []),
                        *([Choice(EditAction.DISCARD_SELECTED, name="DISCARD selected")] if len(items) else []),
                        *([Choice(EditAction.DISCARD_ALL, name="DISCARD ALL")] if len(items) else []),
                    ],
                    default=0 if len(choices_invalid) else EditAction.ADD_NEW if not len(choices_valid) else EditAction.FINALIZE,
                ).execute()

                match item_index:
                    case EditAction.ADD_NEW:
                        new_item = self.add_item(item_type=item_type)
                        if new_item:
                            items.append(new_item)
                        continue

                    case EditAction.FINALIZE:
                        for item in items:
                            if not item.is_valid():
                                item.trust = 0

                        break

                    case EditAction.DISCARD_SELECTED:
                        keys_to_discard: list[int] = inquirer.select(
                            message="Select entries to discard:",
                            choices=[
                                Choice(_id, name=str(i))
                                for _id, i in enumerate(items)
                            ],
                            multiselect=True,
                        ).execute()
                        for item_key in keys_to_discard:
                            items[item_key].trust = 0

                        items = list(self.deps.run_manager.filter_untrusted(items))

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
                        Choice(EditTarget.INVENTORY, name="Change INVENTORY", enabled=item.inventory_status is None or EditTarget.INVENTORY in sus_list),
                        Separator(),
                        Choice(EditTarget.DISCARD, name="DISCARD this entry"),
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
                            self.edit_item_name(item)
                            self.edit_item_price(item)
                            self.edit_item_stock(item)
                            self.edit_item_inventory(item)
                            item.trust = float("inf")

                        case EditTarget.NAME:
                            self.edit_item_name(item)
                            item.trust = float("inf")

                        case EditTarget.PRICE:
                            self.edit_item_price(item)
                            item.trust = float("inf")

                        case EditTarget.STOCK:
                            self.edit_item_stock(item)
                            item.trust = float("inf")

                        case EditTarget.INVENTORY:
                            self.edit_item_inventory(item)
                            item.trust = float("inf")

            except KeyboardInterrupt:
                log.debug("aborting fix based on user input")
                break

            except Exception as e:
                log.exception(f"failed to process changes to item: {e}", exc_info=e)
                continue

            finally:
                self.deps.run_manager.sync_item_changes()
                items = self.deps.run_manager.filter_untrusted(items)

        self.deps.run_manager.sync_item_changes()
        return self.deps.run_manager.filter_untrusted(items)


    def edit_item_stock(self, item) -> bool:
        try:
            item.stock = inquirer.text(
                message="Enter stock:",
                validate=FloatValidator(min_value=0.0),
                filter=lambda result: float(result),
                transformer=lambda result: "%s SCU" % result,
            ).execute()
            return True

        except KeyboardInterrupt:
            return False


    def edit_item_price(self, item) -> bool:
        try:
            item.price = inquirer.text(
                message="Enter price:",
                validate=FloatValidator(min_value=0.0),
                filter=lambda result: float(result),
                transformer=lambda result: "%s aUEC" % result,
            ).execute()
            return True

        except KeyboardInterrupt:
            return False


    def edit_item_name(self, item: DetectedCommodity) -> bool:
        try:
            commodity_id: int = inquirer.fuzzy(
                message="Select commodity:",
                choices=[Choice(c.id, name=f"{c.code}: {c.name}") for c in self.deps.uex_corp.commodities],
            ).execute()
            commodity = self.deps.uex_corp.get_commodity_by_id(commodity_id)
            item.name = commodity.name
            item.commodity = commodity
            return True

        except KeyboardInterrupt:
            return False


    def edit_item_inventory(self, item: DetectedCommodity) -> bool:
        try:
            inventory_status_index: int = inquirer.fuzzy(
                message="Select inventory status:",
                choices=[Choice(_id, name=f"{s.name:>19}") for _id, s in enumerate(self.deps.static_data.inventory_states) if s.visible],
            ).execute()
            item.inventory_status = self.deps.static_data.inventory_states[inventory_status_index]
            return True

        except KeyboardInterrupt:
            return False


    def run(self, action: Action):
        run_manager = self.deps.run_manager

        try:
            clipboard_image = ImageGrab.grabclipboard()
            clipboard_image = np.array(clipboard_image)

        except Exception as e:
            log.exception(f"failed to grab clipboard image: {e}", exc_info=e)
            return

        try:
            items = []
            if len(clipboard_image.shape) == 3:
                clipboard_image = cv2.cvtColor(clipboard_image, cv2.COLOR_RGB2BGR)
                image_used, items = self.deps.image_processing.process_image(clipboard_image, image_name="clipboard screenshot")
                log.debug(f"detected items: {items}")

            else:
                self.deps.output.report_note("There is no image in the clipboard.")

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
                prices = self.deps.uex_corp.get_commodity_price_by_terminal(run_manager.terminal.id, item.commodity.id) if run_manager.terminal is not None and item.commodity is not None else None
                return run_manager.get_sus_attrs(item, prices, items_type) if prices is not None else []

            items = self.edit_items(
                items,
                item_type=items_type,
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
                run_manager.add_image(image_used)

        except Exception as e:
            log.exception(f"failed to process detected items: {e}", exc_info=e)
            return


    def add_item(self, item_type: ItemType = ItemType.UNDEFINED) -> DetectedCommodity | None:
        if item_type == ItemType.UNDEFINED:
            item_type = inquirer.select(
                message="Select item category:",
                choices=[
                    Choice(ItemType.BUY, name="BUY list"),
                    Choice(ItemType.SELL, name="SELL list"),
                ],
            ).execute()

        item = DetectedCommodity(
            name="",
            commodity=None,
            price=0,
            stock=0,
            inventory_status=None,
            container_sizes=None,
            order=0,
        )

        if not self.edit_item_name(item):
            return None
        if not self.edit_item_price(item):
            return None
        if not self.edit_item_stock(item):
            return None
        if not self.edit_item_inventory(item):
            return None

        item.trust = float("inf")
        if not item.is_valid():
            self.deps.output.report_warning("The created item is not valid.")
            return None

        match item_type:
            case ItemType.BUY:
                self.deps.run_manager.add_buy(item)
            case ItemType.SELL:
                self.deps.run_manager.add_sell(item)

        return item


    def commit(self):
        run_manager = self.deps.run_manager

        if not run_manager.is_dirty():
            self.deps.output.report_warning("There are no data run changes to commit.")
            log.info("no changes to commit")
            return

        run_manager.change_terminal(self.deps.uex_corp)

        def confirm_submission() -> bool:
            dry_run_info = ""
            if self.deps.settings.dry_run:
                dry_run_info = " [DRY RUN]"

            print()
            run_manager.item_overview(run_manager.buy, prefix="Tracked BUY contains", item_type=ItemType.BUY, sort_by=lambda i: i.name if i.name is not None else "zzz")
            run_manager.item_overview(run_manager.sell, prefix="Tracked SELL contains", item_type=ItemType.SELL, sort_by=lambda i: i.name if i.name is not None else "zzz")
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
                    self.add_item()

                case CommitRejectAction.EDIT:
                    self.edit_items(run_manager.buy + run_manager.sell, show_items=False, sort_by=lambda i: i.name if i.name is not None else "zzz")

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
                self.deps.output.report_warning("There are no data run changes to commit anymore.")
                log.info("no changes to commit anymore")
                return

        try:
            self.commit_new(run_manager)

        except HTTPError as e:
            log.exception("failed submitting data: %s", str(e), exc_info=e)


    def commit_new(self, run_manager: DataRunManager):
        container_sizes = None
        if len(run_manager.sell):
            container_sizes: list[ContainerSize] | None = inquirer.select(
                message="Select available container sizes:",
                choices=[
                    Choice(size, name=f"{size.value:2} SCU", enabled=size.value <= self.deps.run_manager.terminal.max_container_size)
                    for size in list(ContainerSize)
                ],
                multiselect=True,
            ).execute()

        with Progress(transient=True) as progress:
            task_id = progress.add_task("Preprocessing...", total=None)

            screenshot_base64 = None
            if run_manager.images:
                screenshot_fig, _ = merge_images(
                    run_manager.images,
                    [f"Screenshot {i + 1}" for i in range(len(run_manager.images))],
                    figure_size_base=(5, 8),
                )
                screenshot_base64 = figure_to_base64(screenshot_fig)
                if self.deps.settings.include_screenshots and self.deps.settings.show_images:
                    plt.show()

            data_run = DataRun(
                id_terminal=run_manager.terminal.id,
                prices=[
                    *[item.create_buy_entry() for item in run_manager.buy],
                    *[item.create_sell_entry() for item in run_manager.sell],
                ],
                container_sizes=",".join([str(size.value) for size in container_sizes]),
                is_production=False if self.deps.settings.dry_run else True,
                screenshot=screenshot_base64 if self.deps.settings.include_screenshots and screenshot_base64 else None,
            )

            log.debug("submitting data run: %s", asdict(data_run, dict_factory=lambda d: {k: v for k, v in d if k != "screenshot"}))
            progress.update(task_id, description=f"Submitting...")
            response = self.deps.uex_corp.api.submit_data_run(data_run)

        log.info("successfully submitted data run (dry run: %s)", self.deps.settings.dry_run)
        if self.deps.settings.dry_run:
            self.deps.output.report_note("Real submission skipped. (dry run is enabled)")
            return

        if self.deps.settings.show_report_links:
            table = Table(title="Data Run Reports")
            table.add_column("Commodity")
            table.add_column("Report Link")

            buy_commodities = {item.commodity.id: item.commodity for item in run_manager.buy}
            for report_id, commodity in zip(response.data.ids_reports[:len(buy_commodities)], buy_commodities.values()):
                table.add_row(
                    commodity.name,
                    f"https://uexcorp.space/data/info/id/{report_id}",
                )

            sell_commodities = {item.commodity.id: item.commodity for item in run_manager.sell}
            for report_id, commodity in zip(response.data.ids_reports[len(buy_commodities):], sell_commodities.values()):
                table.add_row(
                    commodity.name,
                    f"https://uexcorp.space/data/info/id/{report_id}",
                )

            print()
            Console().print(table)
            print()

        self.deps.output.report_success("Data run successfully submitted!")
        self.deps.output.report(f" - https://uexcorp.space/data/home/type/commodity/id_terminal/{run_manager.terminal.id}/datarunner/{response.data.username}")
        self.deps.output.report("")
        run_manager.clear()


    def run_choices(self):
        data_run = self.deps.run_manager
        while True:
            if data_run.is_dirty():
                print()
                data_run.tracked_items_overview()

            action_prompt = inquirer.select(
                message="Select an action:",
                choices=[
                    Choice(Action.CHANGE_TERMINAL, name=f"Change TERMINAL ({data_run.current_terminal_name()})"),
                    Choice(Action.PROCESS_BUY, name="Process BUY commodities (screenshot/manual)"),
                    Choice(Action.PROCESS_SELL, name="Process SELL commodities (screenshot/manual)"),
                    *([
                        Separator(),
                        Separator(f"  There are {data_run.tracked_items} tracked items."),
                        Choice(Action.COMMIT, name="COMMIT data run", enabled=data_run.is_dirty()),
                        Choice(Action.CLEAR, name="CLEAR data run", enabled=data_run.is_dirty()),
                    ] if data_run.is_dirty() else []),
                    Separator(),
                    Choice(None, name="Exit"),
                ],
                long_instruction="b=BUY, s=SELL, f=COMMIT, c=CLEAR, t=TERMINAL, q=EXIT",
                default=Action.CHANGE_TERMINAL if data_run.terminal is None
                else Action.PROCESS_BUY if not data_run.is_dirty()
                else Action.PROCESS_SELL,
            )

            @action_prompt.register_kb("b")
            def _handle_process_buy(event): event.app.exit(result=Action.PROCESS_BUY)

            @action_prompt.register_kb("s")
            def _handle_process_sell(event): event.app.exit(result=Action.PROCESS_SELL)

            @action_prompt.register_kb("f", filter=data_run.is_dirty())
            def _handle_commit(event): event.app.exit(result=Action.COMMIT)

            @action_prompt.register_kb("c", filter=data_run.is_dirty())
            def _handle_clear(event): event.app.exit(result=Action.CLEAR)

            @action_prompt.register_kb("t")
            def _handle_change_port(event): event.app.exit(result=Action.CHANGE_TERMINAL)

            @action_prompt.register_kb("q")
            def _handle_exit(event): event.app.exit(result=None)

            action = action_prompt.execute()
            match action:
                case Action.CHANGE_TERMINAL:
                    data_run.change_terminal(self.deps.uex_corp, use_default=False)

                case Action.PROCESS_BUY | Action.PROCESS_SELL:
                    self.run(action)

                case Action.COMMIT:
                    self.commit()

                case Action.CLEAR:
                    data_run.clear()

                case _:
                    break

