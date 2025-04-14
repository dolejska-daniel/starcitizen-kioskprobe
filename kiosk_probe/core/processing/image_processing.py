import base64
import json
import logging.config
from pathlib import Path

import cv2
import numpy as np
from openai import OpenAI
from openai.types.responses import ResponseUsage
from rich.progress import Progress

from kiosk_probe.core.datarun.objects import DetectedCommodity
from kiosk_probe.core.processing.objects import InventoryEntriesResponse, InventoryEntry, InventoryAvailability
from kiosk_probe.core.utils import find_best_string_match
from kiosk_probe.uex_corp.objects import Commodity, InventoryStatus
from main import DependencyContainer

log = logging.getLogger("kiosk_probe." + __name__)


class ImageProcessing:

    tokens_used: int = 0
    tokens_cached: int = 0
    reported_tokens_used: int = 0

    def __init__(self, deps: DependencyContainer):
        self.deps = deps
        self.config = deps.settings.openai_config
        self.client = OpenAI(api_key=self.config.api_key)

        schema_file = Path(__file__).parent / "openai.response.schema.json"
        with open(schema_file, "r") as f:
            self.output_schema = json.load(f)

    @staticmethod
    def crop_image(image: np.ndarray) -> np.ndarray:
        max_x, max_y = image.shape[1], image.shape[0]
        shortest_side = min(max_x, max_y)
        min_x, min_y = max_x - shortest_side, max_y - shortest_side

        return image[min_y:max_y, min_x:max_x]

    def process_image(self, image: np.ndarray, image_name: str = "image") -> tuple[np.ndarray, list[DetectedCommodity]]:
        image_used, inventory = self.detect_inventory_entries(image, image_name)
        commodities = [
            self.create_detected_commodity(entry, index)
            for index, entry in enumerate(inventory.entries)
        ]

        return image_used, commodities

    def detect_inventory_entries(self, image: np.ndarray, image_name: str) -> tuple[np.ndarray, InventoryEntriesResponse]:
        with Progress(transient=True) as progress:
            log.debug("pre-processing %s", image_name)
            task_id = progress.add_task("Preprocessing...", total=None)

            image_used = self.crop_image(image)
            image_used = cv2.cvtColor(image_used, cv2.COLOR_BGR2RGB)
            _, image_used_png = cv2.imencode('.png', image_used)

            if self.deps.settings.show_images:
                from matplotlib import pyplot as plt
                plt.imshow(image_used)
                plt.show()

            image_used_base64 = base64.b64encode(image_used_png.tobytes()).decode("ascii")
            image_used_base64 = f"data:image/png;base64,{image_used_base64}"

            log.debug("processing %s", image_name)
            progress.update(task_id, description=f"Prompting...")
            response = self.client.responses.create(
                model=self.config.model,
                input=[
                    {
                        "role": "developer",
                        "content": [
                            {
                                "type": "input_text",
                                "text": "Screen capture with a list of entries. Each entry has a `entry_name` and `entry_availability` on the left and `entry_stock` and `entry_price` on the right. Prices may have SI suffixes. Do not parse numeric values, only determine text contents for all properties. If some are incomplete, report the properties you detected with null values otherwise.""",
                            }
                        ]
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": image_used_base64,
                            },
                        ],
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        **self.output_schema
                    }
                },
                temperature=0.25,
                store=True,
            )

            log.debug("parsing detection response")
            progress.update(task_id, description=f"Processing response...")
            self.track_usage(response.usage)
            response_content = json.loads(response.output_text)
            return image_used, InventoryEntriesResponse(**response_content)

    def track_usage(self, usage: ResponseUsage):
        log.info("used %d input tokens (%d cached)", usage.input_tokens, usage.input_tokens_details.cached_tokens)
        log.info("used %d output tokens (%d reasoning)", usage.output_tokens, usage.output_tokens_details.reasoning_tokens)

        self.tokens_used += usage.total_tokens
        self.tokens_cached += usage.input_tokens_details.cached_tokens
        if self.tokens_used - self.reported_tokens_used > self.config.report_tokens_used:
            self.deps.output.report_note(f"Tokens used this session so far: {self.tokens_used} ({self.tokens_cached} cached)")

    def find_commodity(self, entry: InventoryEntry) -> Commodity:
        commodities = self.deps.uex_corp.api.get_commodities()
        commodity_match, distance = find_best_string_match(entry.entry_name, commodities, key=lambda c: c.name.upper()[:16])
        distance_max = len(commodity_match.name) * 0.33
        if distance <= distance_max:
            log.debug("matched commodity: %s", commodity_match)
            return commodity_match

        log.debug(
            f"unmatched commodity '%s', best guess '%s' (distance %d/%d)",
            entry.entry_name, commodity_match.name, distance, distance_max
        )
        return commodity_match

    def get_inventory_status(self, availability: InventoryAvailability | None) -> InventoryStatus | None:
        return self.deps.static_data.inventory_availability_to_status[availability]

    def create_detected_commodity(self, entry: InventoryEntry, index: int) -> DetectedCommodity:
        log.debug("creating detected commodity from %s", entry)
        return DetectedCommodity(
            name=entry.entry_name,
            price=entry.price,
            stock=entry.stock,
            inventory_status=self.get_inventory_status(entry.availability),
            container_sizes=None,
            order=index,
            commodity=self.find_commodity(entry),
        )
