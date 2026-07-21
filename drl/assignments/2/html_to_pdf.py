"""Render the exported HTML notebook to a PDF using headless Chromium.

Avoids the nbconvert `webpdf` Windows asyncio bug by driving Playwright's sync
API directly (with the Proactor event-loop policy required for subprocesses on
Windows).
"""
import asyncio
import sys
from pathlib import Path

# Playwright needs the Proactor loop to spawn its driver subprocess on Windows.
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from playwright.sync_api import sync_playwright  # noqa: E402

HTML = Path("Team165_Q_learning_DQN_DDQN.html").resolve()
PDF = Path("Team165_Q_learning_DQN_DDQN.pdf").resolve()


def main():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        # Load the local HTML file and wait for images (plots) to finish.
        page.goto(HTML.as_uri(), wait_until="networkidle", timeout=120_000)
        page.pdf(
            path=str(PDF),
            format="A4",
            print_background=True,
            margin={"top": "12mm", "bottom": "12mm", "left": "10mm", "right": "10mm"},
        )
        browser.close()
    print(f"Wrote {PDF}  ({PDF.stat().st_size/1_048_576:.2f} MB)")


if __name__ == "__main__":
    main()
