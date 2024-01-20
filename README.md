# Star Citizen Kiosk Probe

## Installation instructions

Order of the installation steps is important!

1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. _(for GPU acceleration only)_ install from https://developer.nvidia.com/cuda-downloads
4. install from https://pytorch.org/
   - Compute Platform:
     1. _(for GPU acceleration only)_ **CUDA** (version based on step 3)
     2. **CPU** - no GPU acceleration
5. `pip install -r requirements.txt`

## Configuration
To be able to submit the data to UEX Corp servers, you will need to provide a valid session credentials.
Login to the UEX Corp website and copy the appropriate cookie values into the `config/session.json` file.
Those cookies are:
- `PHPSESSID`
- `uex_token`
- `uex_email`


## Usage
Run the following commands in the project root directory to start the program:
1. `source venv/bin/activate`
2. `python3 main.py`

After the program has started, you can use the CLI to interact with it.
The CLI is a simple text-based interface that allows you to select commands and actions to take.

The images for analysis are grabbed directly from clipboard.
Either copy an image to clipboard or take a screenshot and select any `Process` command.
