import logging

log = logging.getLogger("kiosk_probe." + __name__)


class DependencyContainer:

    def __init__(self):
        from kiosk_probe.core.datarun.manager import DataRunManager
        from kiosk_probe.core.processing.image_processing import ImageProcessing
        from kiosk_probe.uex_corp.api import UEXCorp
        from kiosk_probe.core.settings import Settings
        from kiosk_probe.core.static_data import StaticData
        from kiosk_probe.core.output import UserOutput

        log.debug("initializing dependency container")
        self.settings = Settings()
        self.static_data = StaticData()
        self.uex_corp = UEXCorp(self.settings.uex_corp_config)
        self.run_manager = DataRunManager(self.uex_corp)
        self.image_processing = ImageProcessing(self)
        self.output = UserOutput()
