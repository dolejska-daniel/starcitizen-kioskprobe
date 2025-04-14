from enum import Enum


class Action(Enum):
    CHANGE_TERMINAL = 0
    PROCESS_BUY = 1
    PROCESS_SELL = 2
    COMMIT = 3
    CLEAR = 4


class EditAction(Enum):
    FINALIZE = -1
    ADD_NEW = -2
    DISCARD_SELECTED = -3
    DISCARD_ALL = -4


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


class ItemType(Enum):
    BUY = "buy"
    SELL = "sell"
    UNDEFINED = "undefined"


class InventoryAvailability(Enum):
    Undetected = 0
    OUT_OF_STOCK = 1
    VERY_LOW_INVENTORY = 2
    LOW_INVENTORY = 3
    MEDIUM_INVENTORY = 4
    HIGH_INVENTORY = 5
    VERY_HIGH_INVENTORY = 6
    MAX_INVENTORY = 7
    NO_DEMAND = 8


class ContainerSize(Enum):
    SCU_1 = 1
    SCU_2 = 2
    SCU_4 = 4
    SCU_8 = 8
    SCU_16 = 16
    SCU_24 = 24
    SCU_32 = 32
