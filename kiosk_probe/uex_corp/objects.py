from dataclasses import dataclass, field, KW_ONLY
from typing import TypedDict, Any


class ResponseBase(TypedDict):
    status: str
    http_code: int
    data: dict | list


@dataclass
class OpenAIApiConfig:
    api_key: str
    model: str
    report_tokens_used: int = field(default=100_000)


@dataclass
class UEXCorpApiConfig:
    application_token: str
    user_token: str
    use_file_cache: bool = field(default=True)
    base_url: str = field(default="https://api.uexcorp.space/2.0")
    cookies_filepath: str | None = field(default=None)


@dataclass
class InventoryStatus:
    value: int | None
    name: str
    visible: bool = field(default=True)

    def __str__(self):
        return self.name


@dataclass
class CommodityContainerSize:
    value: int | None
    name: str
    visible: bool = field(default=True)

    def __str__(self):
        return self.name


@dataclass
class Commodity:
    id: int
    id_parent: int
    name: str
    code: str
    kind: str
    price_buy: float
    price_sell: float
    is_available_live: bool
    is_available: bool
    is_extractable: bool
    is_mineral: bool
    is_refined: bool
    is_refinable: bool
    is_visible: bool
    is_raw: bool
    is_harvestable: bool
    is_buyable: bool
    is_sellable: bool
    is_temporary: bool
    is_illegal: bool
    is_volatile_qt: bool
    is_volatile_time: bool
    is_inert: bool
    is_explosive: bool
    is_fuel: bool
    is_buggy: bool
    wiki: str
    date_added: int
    date_modified: int
    weight_scu: int


@dataclass
class StarSystem:
    id: int
    name: str
    code: str
    id_faction: int
    faction_name: str
    id_jurisdiction: int
    jurisdiction_name: str
    is_available_live: bool
    is_available: bool
    is_visible: bool
    is_default: bool
    wiki: str
    date_added: int
    date_modified: int


@dataclass
class Terminal:
    id: int
    id_star_system: int
    id_planet: int
    id_orbit: int
    id_moon: int
    id_space_station: int
    id_outpost: int
    id_city: int
    id_poi: int
    id_faction: int
    id_company: int
    faction_name: str
    company_name: str
    contact_url: str
    mcs: str
    name: str
    nickname: str
    code: str
    type: str
    is_available_live: bool
    is_available: bool
    is_nqa: bool
    is_visible: bool
    is_jump_point: bool
    is_default_system: bool
    is_affinity_influenceable: bool
    is_habitation: bool
    is_refinery: bool
    is_cargo_center: bool
    is_medical: bool
    is_food: bool
    is_shop_fps: bool
    is_shop_vehicle: bool
    is_refuel: bool
    is_repair: bool
    is_player_owned: bool
    is_auto_load: bool
    has_loading_dock: bool
    has_docking_port: bool
    has_freight_elevator: bool
    max_container_size: int
    date_added: int
    date_modified: int
    star_system_name: str
    planet_name: str | None
    orbit_name: str | None
    moon_name: str | None
    space_station_name: str | None
    outpost_name: str | None
    city_name: str | None
    id_jurisdiction: int | None = field(default=None)
    jurisdiction_name: str | None = field(default=None)
    screenshot: str | None = field(default=None)
    has_container_transfer: bool = field(default=False)

    @property
    def name_full(self):
        return ' / '.join(filter(None, [
            self.moon_name,
            self.space_station_name,
            self.outpost_name,
            self.city_name,
            self.name,
        ]))


@dataclass
class CommodityPrice:
    id: int
    id_commodity: int
    id_star_system: int
    id_planet: int
    id_orbit: int
    id_moon: int
    id_city: int
    id_outpost: int
    id_poi: int
    id_faction: int
    id_terminal: int
    price_buy: float
    price_buy_min: float
    price_buy_min_week: float
    price_buy_min_month: float
    price_buy_max: float
    price_buy_max_week: float
    price_buy_max_month: float
    price_buy_avg: float
    price_buy_avg_week: float
    price_buy_avg_month: float
    price_buy_users: float
    price_buy_users_rows: int | None
    price_sell: float
    price_sell_min: float
    price_sell_min_week: float
    price_sell_min_month: float
    price_sell_max: float
    price_sell_max_week: float
    price_sell_max_month: float
    price_sell_avg: float
    price_sell_avg_week: float
    price_sell_avg_month: float
    price_sell_users: float
    price_sell_users_rows: int | None
    scu_buy: float
    scu_buy_min: float
    scu_buy_min_week: float
    scu_buy_min_month: float
    scu_buy_max: float
    scu_buy_max_week: float
    scu_buy_max_month: float
    scu_buy_avg: float
    scu_buy_avg_week: float
    scu_buy_avg_month: float
    scu_buy_users: float
    scu_buy_users_rows: int | None
    scu_sell_stock: float
    scu_sell_stock_avg: float
    scu_sell_stock_avg_week: float
    scu_sell_stock_avg_month: float
    scu_sell: float
    scu_sell_min: float
    scu_sell_min_week: float
    scu_sell_min_month: float
    scu_sell_max: float
    scu_sell_max_week: float
    scu_sell_max_month: float
    scu_sell_avg: float
    scu_sell_avg_week: float
    scu_sell_avg_month: float
    scu_sell_users: float
    scu_sell_users_rows: int | None
    status_buy: int
    status_buy_min: int
    status_buy_min_week: int
    status_buy_min_month: int
    status_buy_max: int
    status_buy_max_week: int
    status_buy_max_month: int
    status_buy_avg: int
    status_buy_avg_week: int
    status_buy_avg_month: int
    status_sell: float
    status_sell_min: float
    status_sell_min_week: float
    status_sell_min_month: float
    status_sell_max: float
    status_sell_max_week: float
    status_sell_max_month: float
    status_sell_avg: float
    status_sell_avg_week: float
    status_sell_avg_month: float
    volatility_buy: float
    volatility_price_buy: float
    volatility_price_sell: float
    volatility_sell: float
    volatility_scu_buy: float
    volatility_scu_sell: float
    faction_affinity: float | None
    container_sizes: str | None
    game_version: str
    date_added: int
    date_modified: int
    commodity_name: str
    commodity_code: str
    commodity_slug: str
    star_system_name: str
    planet_name: str
    orbit_name: str
    moon_name: str | None
    space_station_name: str | None
    outpost_name: str | None
    city_name: str | None
    faction_name: str
    poi_name: str
    terminal_mcs: str
    terminal_name: str
    terminal_code: str
    terminal_slug: str
    terminal_is_player_owned: bool


@dataclass
class DataRunCommodityBuyEntry:
    id_commodity: int
    price_buy: float
    scu_buy: int
    status_buy: int
    is_missing: bool = field(default=False)


@dataclass
class DataRunCommoditySellEntry:
    id_commodity: int
    price_sell: float
    scu_sell: int
    status_sell: int
    is_missing: bool = field(default=False)


@dataclass
class DataRun:
    id_terminal: int
    prices: list[DataRunCommodityBuyEntry | DataRunCommoditySellEntry]
    container_sizes: str | None = field(default=None)
    faction_affinity: int | None = field(default=None)
    details: str | None = field(default=None)
    game_version: str | None = field(default=None)
    is_production: bool = field(default=False)
    screenshot: str | None = field(default=None, repr=False)
    type: str = field(default="commodity")


@dataclass
class DataRunResponseData:
    ids_reports: list[str]
    date_added: int
    username: str


@dataclass
class DataRunResponse:
    status: str
    http_code: int
    data: DataRunResponseData
    message: str

    # noinspection PyArgumentList
    def __post_init__(self):
        self.data = DataRunResponseData(**self.data)
