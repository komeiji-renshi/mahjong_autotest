from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

try:
    from playwright.sync_api import Browser, Page, Playwright, sync_playwright  # type: ignore
    from playwright.sync_api import Error as PlaywrightError  # type: ignore
except ImportError:  # pragma: no cover - optional during bootstrap
    Browser = None  # type: ignore[assignment]
    Page = None  # type: ignore[assignment]
    Playwright = None  # type: ignore[assignment]
    sync_playwright = None  # type: ignore[assignment]
    PlaywrightError = Exception  # type: ignore[assignment]

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover - optional during bootstrap
    cv2 = None


@dataclass(slots=True)
class WebConfig:
    url: str
    browser: str = "chromium"  # chromium | firefox | webkit
    headless: bool = False
    viewport_width: int = 1280
    viewport_height: int = 720
    startup_click_selectors: tuple[str, ...] = ("button.tutorial-btn-primary",)


class WebController:
    """
    Browser-based controller for HTML5 Mahjong games.
    Coordinates used by tap() are page viewport coordinates.
    """

    def __init__(self, config: WebConfig) -> None:
        self._config = config
        self._log = logging.getLogger("vita_mahjong_bot.web")
        self._playwright_ctx: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None

    def connect(self) -> bool:
        if self._page is not None and not self._page.is_closed():
            return True
        if sync_playwright is None:
            raise RuntimeError(
                "playwright is required for web mode. Install with `pip install playwright` "
                "and run `python -m playwright install` once."
            )
        self._playwright_ctx = sync_playwright().start()
        launcher = getattr(self._playwright_ctx, self._config.browser, None)
        if launcher is None:
            raise ValueError(f"Unsupported browser in config: {self._config.browser!r}")
        self._browser = launcher.launch(
            headless=self._config.headless,
            args=["--force-device-scale-factor=1", "--high-dpi-support=1"],
        )
        context = self._browser.new_context(
            viewport={"width": self._config.viewport_width, "height": self._config.viewport_height}
        )
        self._page = context.new_page()
        self._page.goto(self._config.url, wait_until="domcontentloaded")
        self._stabilize_page_view()
        return True

    def is_device_online(self) -> bool:
        return self._page is not None and not self._page.is_closed()

    def start_app(self) -> None:
        self.connect()
        self._perform_startup_clicks()

    def stop_app(self) -> None:
        if self._page is not None and not self._page.is_closed():
            self._page.close()
        self._page = None
        if self._browser is not None:
            self._browser.close()
        self._browser = None
        if self._playwright_ctx is not None:
            self._playwright_ctx.stop()
        self._playwright_ctx = None

    def screencap(self) -> np.ndarray:
        try:
            page = self._require_page()
            return self._screencap_from_page(page)
        except PlaywrightError:
            self._log.warning("Page lost during screenshot, reconnecting and retrying once.")
            self._recover_page()
            page = self._require_page()
            return self._screencap_from_page(page)

    def tap(self, x: int, y: int) -> None:
        try:
            page = self._require_page()
            page.mouse.click(x, y)
        except PlaywrightError:
            self._log.warning("Page lost during tap, reconnecting and retrying once.")
            self._recover_page()
            self._require_page().mouse.click(x, y)

    def keyevent(self, keycode: int) -> None:
        if keycode == 1001:
            self._click_hint_button()
            return
        key_map = {4: "Escape", 66: "Enter"}
        key = key_map.get(keycode)
        if key is None:
            return
        try:
            page = self._require_page()
            page.keyboard.press(key)
        except PlaywrightError:
            self._log.warning("Page lost during keyevent, reconnecting and retrying once.")
            self._recover_page()
            self._require_page().keyboard.press(key)

    def _click_hint_button(self) -> None:
        selectors = [
            "button[title*='显示可能的移动']",
            "button.feature[title*='可能的移动']",
            "button:has(span.label:has-text('提示'))",
            "button:has-text('提示')",
        ]
        page = self._require_page()
        for selector in selectors:
            try:
                locator = page.locator(selector).first
                if locator.count() == 0:
                    continue
                if not locator.is_visible():
                    continue
                locator.click(timeout=1200)
                page.wait_for_timeout(220)
                self._log.info("Clicked hint selector: %s", selector)
                return
            except Exception:
                continue
        self._log.info("Hint button not found.")

    def _perform_startup_clicks(self) -> None:
        page = self._require_page()
        # Click onboarding/tutorial buttons in configured order.
        for selector in self._config.startup_click_selectors:
            clicked = False
            for _ in range(20):
                try:
                    locator = page.locator(selector).first
                    if locator.count() == 0:
                        page.wait_for_timeout(300)
                        continue
                    if not locator.is_visible():
                        page.wait_for_timeout(300)
                        continue
                    locator.click(timeout=2000)
                    page.wait_for_timeout(400)
                    self._log.info("Clicked startup selector: %s", selector)
                    clicked = True
                    break
                except Exception:
                    page.wait_for_timeout(300)
                    continue
            if not clicked:
                self._log.info("Startup selector not found or not clickable: %s", selector)

    def _stabilize_page_view(self) -> None:
        page = self._require_page()
        try:
            # Reset browser zoom in case previous run changed it.
            page.keyboard.press("Control+0")
        except PlaywrightError:
            pass
        try:
            page.evaluate(
                """
                () => {
                    const html = document.documentElement;
                    const body = document.body;
                    if (!html || !body) return;
                    html.style.overflow = 'hidden';
                    body.style.overflow = 'hidden';
                    html.style.zoom = '100%';
                    body.style.zoom = '100%';
                    body.style.transform = 'none';
                }
                """
            )
        except PlaywrightError:
            pass

    def _screencap_from_page(self, page: Page) -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("opencv-python is required for screenshot decoding")
        png_bytes = page.screenshot(full_page=False)
        raw = np.frombuffer(png_bytes, dtype=np.uint8)
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("failed to decode page screenshot")
        return image

    def _recover_page(self) -> None:
        self.stop_app()
        self.connect()
        self._perform_startup_clicks()

    def _require_page(self) -> Page:
        if self._page is None or self._page.is_closed():
            self.connect()
        assert self._page is not None
        return self._page
