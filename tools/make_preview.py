"""Writes images/preview.png: a screenshot of a running llama-server's page.

Development only (not in the package). Start a server first, e.g.
``llama-server --port 8080 --models-dir <folder>``, then run
``python tools/make_preview.py --url http://127.0.0.1:8080`` with Playwright
and its Chromium installed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from playwright.sync_api import sync_playwright

TARGET = Path(__file__).resolve().parent.parent / "images" / "preview.png"


def main() -> None:
    """Takes the screenshot at 2560x1440, the size of the application's own previews."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="the running server's address")
    parser.add_argument("--scheme", choices=("light", "dark"), default="dark")
    args = parser.parse_args()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(
            viewport={"width": 1280, "height": 720},
            device_scale_factor=2,
            color_scheme=args.scheme,
        )
        page.goto(args.url, wait_until="load")
        page.wait_for_timeout(3000)
        page.screenshot(path=str(TARGET))
        browser.close()
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
