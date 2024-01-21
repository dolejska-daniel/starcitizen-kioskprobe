from enum import Enum


class Action(Enum):
    CHANGE_TERMINAL = 0
    PROCESS_BUY = 1
    PROCESS_SELL = 2
    COMMIT = 3
    CLEAR = 4


class BoundaryPosition(Enum):
    TOP_LEFT = 0
    TOP_RIGHT = 1
    BOTTOM_RIGHT = 2
    BOTTOM_LEFT = 3


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class NodeType(Enum):
    UNKNOWN = 0
    COMMODITY_NAME = 1
    COMMODITY_PRICE = 2
    COMMODITY_STOCK = 3
    COMMODITY_INVENTORY = 4


class EditAction(Enum):
    FINALIZE = -1
    ADD_NEW = -2
    DISCARD_ALL = -3


class CommitRejectAction(Enum):
    ABORT = 0
    CONTINUE = 1
    ADD_NEW = 2
    EDIT = 3
    DISCARD = 4
    CLEAR = 5


class EditTarget(Enum):
    SKIP = -3
    DISCARD = -2
    ALL = -1
    NAME = 0
    PRICE = 1
    STOCK = 2
    INVENTORY = 3
