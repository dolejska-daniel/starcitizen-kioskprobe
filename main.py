import argparse
import logging.config
import pprint
from pathlib import Path

import cv2
import readchar

from kiosk_probe.core.controls import Controls
from kiosk_probe.core.dependency_container import DependencyContainer

print("\rLoading...", end="", flush=True)

logging_config_file = Path(__file__).parent / "config" / "logging.ini"
logging.config.fileConfig(logging_config_file.absolute())

log = logging.getLogger("kiosk_probe." + __name__)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", action="store", help="run for a single image", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="do not submit data run")
    parser.add_argument("--show-images", action="store_true", help="always show processed images")

    try:
        args = parser.parse_args()
        deps = DependencyContainer()
        deps.settings.show_images = args.show_images
        deps.settings.dry_run = args.dry_run

        if args.image:
            image = cv2.imread(args.image.absolute().as_posix())
            image_used, result = deps.image_processing.process_image(image, image_name=args.image.name)
            pprint.pprint(result)
            exit(0)

        deps.output.clear_transient()
        controls = Controls(deps)
        controls.run_choices()

    except KeyboardInterrupt:
        log.debug("exiting based on user input")
        pass

    except Exception as e:
        log.exception(f"an unrecoverable error has occurred: {e}", exc_info=e)
        print("Press any key to exit . . . ")
        readchar.readchar()
