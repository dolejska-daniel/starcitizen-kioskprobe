# Star Citizen Kiosk Probe

<img src="https://uexcorp.space/img/api/uex-api-badge-powered.png" alt="Powered by https://uexcorp.space" width="100" title="Power by UEX Corp">

Hi, welcome to this tiny project.
This solution aims to provide a simple way to extract prices from commodity terminals ([Admin](https://uexcorp.space/terminals/info/name/admin-everus-harbor/), [TDD](https://uexcorp.space/terminals/info/name/tdd-trade-and-development-division-area-18/), [Outpost](https://uexcorp.space/terminals/info/name/hickes-research-outpost/), ...) in Star Citizen.

Unfortunately this solution requires the use of [OpenAI's API](https://platform.openai.com/) tools and hence is **not free** to use.
Attempts were made to implement a local free solution as well; however, this has proven to be pretty unreliable, especially between different terminal designs (colours, fonts, noise, artefacts).

## Installation instructions

You need to have [Python](https://www.python.org/downloads/) installed in version at least `3.10` with [pip](https://packaging.python.org/en/latest/tutorials/installing-packages/#ensure-you-can-run-pip-from-the-command-line) for this project to work properly.
Order of the installation steps is important!

1. `python3 -m venv venv`
2. `source venv/bin/activate`
3. `pip install -r requirements.txt`

## Configuration
To be able to submit the data to UEX Corp servers, you will need to provide a valid API credentials.
Login to the [UEX Corp website](https://uexcorp.space/account) and copy your secret key to the `user_token` field in the `config/uex.json` file.
You can create `config/uex.json` by cloning the sample configuration:

```shell
cp config/uex.sample.json config/uex.json
```

Furthermore, OpenAI API key will be necessary.
You can generate one in the [organization admin dashboard](https://platform.openai.com/settings/organization/api-keys).
Paste your API key to the `api_key` field in the `config/openai.json` file.
This file can again be created by cloning the sample:

```shell
cp config/openai.sample.json config/openai.json
```

Model that will be used for data extraction can also be configured in the same config file (`model` field).
Default is `gpt-4o-mini` which is currently has the ideal ratio between affordability, speed and intelligence.
This can be changed to any other model which supports both `Image input` and `Responses API`.
These could be, for example:
- `gpt-4o` which usually has better accuracy but is almost 20x more expensive per input token,
- or `gpt-4.5-preview` which rarely does some mistakes but is currently 500x more expensive than `gpt-4o-mini`.

## Usage
Run the following commands in the project root directory to start the program:
1. `source venv/bin/activate`
2. `python3 main.py`

After the program has started, you can use the CLI to interact with it.
The CLI is a simple text-based interface that allows you to select commands and actions to take.

The images for analysis are grabbed directly from clipboard.
Either copy an image to clipboard or take a screenshot and select any `Process` command.
