from typing import Sequence, Callable, TypeVar

import cv2
import numpy as np

from enums import NodeType, Direction, BoundaryPosition

T = TypeVar("T")


def filter_by_type(nodes: Sequence["TextNode"], node_type: NodeType) -> list["TextNode"]:
    nodes = filter(lambda node: node.type == node_type, nodes)
    return list(nodes)


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


def convert_similar_letters_to_numbers(text: str) -> str:
    mapping = {
        "o": "0",
        "O": "0",
        "i": "1",
        "I": "1",
        "l": "1",
        "L": "1",
        "s": "5",
        "S": "5",
        "B": "8",
        "b": "6",
        "G": "6",
        "g": "9",
        "q": "9",
        "z": "2",
        "Z": "2",
    }
    for key, value in mapping.items():
        text = text.replace(key, value)

    return text


def find_best_string_match(text: str, strings: Sequence[T], key: Callable[[T], str] | None = None) -> tuple[T, int]:
    key = key or (lambda s: str(s))
    string_scores = map(lambda s: levenshtein(text, key(s)), strings)
    string_score_pairs = zip(strings, string_scores)
    return min(string_score_pairs, key=lambda pair: pair[1])


color_prob_low = np.array([0, 0, 255], dtype=np.uint8)
color_prob_high = np.array([0, 255, 0], dtype=np.uint8)


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
        self.reference_object = None
        self.bounds = bounds
        self.probability = probability
        self.boundary_center: np.array = sum(bounds, start=np.array([0, 0], dtype=np.int32)) // len(bounds)
        self.radius: float = np.linalg.norm(bounds[BoundaryPosition.BOTTOM_RIGHT.value] - self.boundary_center)
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

    def distance_to(self, node: "TextNode", ratio: np.array = np.array([1.0, 1.0])):
        return np.linalg.norm((self.boundary_center - node.boundary_center) * ratio)

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

    def sort_by_distance(self, nodes: Sequence["TextNode"], max_distance: float = None, ratio_x: float = 1.0, ratio_y: float = 1.0) -> list["TextNode"]:
        ratio = np.array([ratio_x, ratio_y])
        nodes = list(nodes)
        distances = map(lambda node: self.distance_to(node, ratio=ratio), nodes)
        node_distance_pairs = zip(nodes, distances)
        if max_distance is not None:
            node_distance_pairs = filter(lambda pair: pair[1] <= max_distance, node_distance_pairs)

        node_distance_pairs = sorted(node_distance_pairs, key=lambda pair: pair[1])
        nodes = map(lambda pair: pair[0], node_distance_pairs)
        nodes = list(nodes)
        return nodes

    def find_and_connect_bottom(self, nodes: Sequence["TextNode"], node_type: NodeType = None):
        if node_type is not None:
            nodes = filter_by_type(nodes, node_type)

        nodes = filter_by_direction(nodes, self, Direction.DOWN)
        nodes = self.sort_by_distance(nodes, max_distance=self.radius * 2.5, ratio_x=1.5)
        if len(nodes):
            closest_node = nodes[0]
            self.bottom = closest_node

    def find_and_connect_left(self, nodes: Sequence["TextNode"], node_type: NodeType = None):
        if node_type is not None:
            nodes = filter_by_type(nodes, node_type)

        nodes = filter_by_direction(nodes, self, Direction.LEFT)
        nodes = filter(lambda node: node is not self.bottom, nodes)
        nodes = self.sort_by_distance(nodes, max_distance=self.radius * 10.0, ratio_y=5.0)
        if len(nodes):
            closest_node = nodes[0]
            self.left = closest_node
