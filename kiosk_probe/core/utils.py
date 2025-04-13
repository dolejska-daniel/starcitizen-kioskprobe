import math
from typing import Sequence, Callable, TypeVar

import cv2
import numpy as np
from matplotlib import pyplot as plt

T = TypeVar("T")


def merge_images(images: list[np.ndarray], titles: list[str], max_cols: int = 3, figure_size_base: tuple[int, int] = None, **kwargs) -> tuple[plt.Figure, list[plt.Axes]]:
    figure_size = figure_size_base or (5, 5)
    rows, cols = math.ceil(len(images) / float(max_cols)), min(len(images), max_cols)
    figure_size = (figure_size[0] * cols, figure_size[1] * rows)
    if "figsize" not in kwargs:
        kwargs["figsize"] = figure_size

    fig, axes = plt.subplots(rows, cols, **kwargs)
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
