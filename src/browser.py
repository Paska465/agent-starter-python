"""Playwright browser sessions backing the agent's browser tools.

On Windows the agent drives the user's real Microsoft Edge, launched with a
remote debugging port and attached to over CDP. That way the browser is visible
on screen, so the user can watch what the agent does and sign in normally.

If Edge is not available (for example on a server), the session falls back to
Playwright's bundled Chromium.

Each job gets its own browser, keyed by the job's AgentSession (see
get_browser_session). Tool calls within a job are serialized through a lock so
the LLM cannot issue overlapping actions.
"""

import asyncio
import contextlib
import logging
import os
import subprocess
import sys
import urllib.request

from playwright.async_api import Browser, Locator, Page, Playwright, async_playwright

logger = logging.getLogger("browser")

# How much visible page text to include in a snapshot, in characters.
MAX_PAGE_TEXT_CHARS = 2500
# Cap on how many interactive elements are listed in a snapshot.
MAX_SNAPSHOT_ELEMENTS = 100
# Truncation length for a single element's label.
MAX_LABEL_CHARS = 80

# Edge is launched on this local port and Playwright attaches to it over CDP.
# The port is only bound to localhost, so nothing outside the machine can reach it.
CDP_PORT = 9222
# A dedicated profile keeps this browser separate from the user's normal Edge
# windows, so closing it never closes the windows they are working in.
EDGE_PROFILE_DIR = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "IconicEdgeProfile"
)

_EDGE_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Microsoft",
        "Edge",
        "Application",
        "msedge.exe",
    ),
)

INTERACTIVE_SELECTOR = (
    "a, button, input, textarea, select, "
    "[role='button'], [role='link'], [role='tab'], [role='checkbox'], "
    "[role='radio'], [role='menuitem'], [role='option'], [contenteditable]"
)

# Collects interactive elements and tags each with a stable ref attribute in a
# single round-trip. Doing this per element from Python takes tens of seconds on
# large pages, which stalls the voice turn.
_SNAPSHOT_JS = """
(args) => {
  const [selector, maxElements, maxLabelChars] = args;
  document
    .querySelectorAll("[data-lk-ref]")
    .forEach((el) => el.removeAttribute("data-lk-ref"));

  const out = [];
  for (const el of document.querySelectorAll(selector)) {
    const rect = el.getBoundingClientRect();
    const style = window.getComputedStyle(el);
    const visible =
      rect.width > 0 && rect.height > 0 &&
      style.visibility !== "hidden" && style.display !== "none";
    if (!visible) continue;

    const tag = el.tagName.toLowerCase();
    const role = el.getAttribute("role") || tag;
    let label =
      el.getAttribute("aria-label") ||
      el.getAttribute("placeholder") ||
      el.getAttribute("value") ||
      (el.innerText || "").trim();
    label = label.replace(/\\s+/g, " ").slice(0, maxLabelChars);

    out.push({ role, label, href: el.getAttribute("href") });
    el.setAttribute("data-lk-ref", String(out.length));
    if (out.length >= maxElements) break;
  }
  return out;
}
"""


def _find_edge() -> str | None:
    """Return the path to Microsoft Edge, or None if it is not installed."""
    if sys.platform != "win32":
        return None
    for path in _EDGE_CANDIDATES:
        if path and os.path.isfile(path):
            return path
    return None


def _cdp_is_up(port: int) -> bool:
    """Whether a browser is already listening for CDP on the port.

    This does a blocking socket connect, so callers on the event loop must run
    it on a worker thread; a probe that stalls would otherwise freeze audio and
    turn handling for the duration.
    """
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=1):
            return True
    except Exception:
        return False


async def _cdp_is_up_async(port: int) -> bool:
    """Non-blocking wrapper around _cdp_is_up for use on the event loop."""
    return await asyncio.to_thread(_cdp_is_up, port)


class BrowserSession:
    """Manages one browser, context, and page per agent session."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._page: Page | None = None
        self._count: int = 0
        # True when we attached to a browser that we do not own, such as an Edge
        # the user already had open. Closing such a browser would close the
        # user's own windows, so we only disconnect from it.
        self._attached: bool = False
        self._edge_proc: subprocess.Popen | None = None

    @property
    def lock(self) -> asyncio.Lock:
        """Serializes tool calls; tools must hold this while acting."""
        return self._lock

    async def _launch_edge(self) -> Browser:
        """Launch the user's Edge with a debug port and attach to it over CDP."""
        if not await _cdp_is_up_async(CDP_PORT):
            edge = _find_edge()
            if edge is None:
                raise RuntimeError("Microsoft Edge not found")
            logger.info("launching Microsoft Edge for browser tools")
            os.makedirs(EDGE_PROFILE_DIR, exist_ok=True)
            self._edge_proc = subprocess.Popen(
                [
                    edge,
                    f"--remote-debugging-port={CDP_PORT}",
                    f"--user-data-dir={EDGE_PROFILE_DIR}",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-features=msEdgeSidebarV2",
                    "about:blank",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Edge needs a moment before its debug port accepts connections.
            for _ in range(40):
                if await _cdp_is_up_async(CDP_PORT):
                    break
                await asyncio.sleep(0.25)
            else:
                raise RuntimeError("Edge did not open its debugging port")
        else:
            logger.info("attaching to Edge already listening for browser tools")

        browser = await self._playwright.chromium.connect_over_cdp(
            f"http://127.0.0.1:{CDP_PORT}"
        )
        # An attached browser is not ours to close: that would shut down the
        # user's own Edge windows.
        self._attached = True
        return browser

    async def get_page(self) -> Page:
        """Return the shared page, launching the browser if needed."""
        if self._page is not None and not self._page.is_closed():
            return self._page

        if (
            self._playwright is None
            or self._browser is None
            or not self._browser.is_connected()
        ):
            if self._playwright is not None:
                await self._close_browser()
            if self._playwright is None:
                self._playwright = await async_playwright().start()
            try:
                self._browser = await self._launch_edge()
            except Exception as e:
                logger.warning("Edge unavailable (%s); using bundled Chromium", e)
                self._attached = False
                self._browser = await self._playwright.chromium.launch(headless=False)

        # Reuse the visible tab when attaching to Edge, so the user sees the
        # agent working in a window that is already on screen.
        context = (
            self._browser.contexts[0]
            if self._attached and self._browser.contexts
            else None
        )
        if context is None:
            context = await self._browser.new_context()
        page = context.pages[0] if context.pages else await context.new_page()
        self._page = page
        self._count = 0
        return self._page

    async def navigate(self, url: str) -> str:
        """Open a URL in the shared page and return a short status line."""
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        page = await self.get_page()
        await page.bring_to_front()
        await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        self._count = 0
        return f'Landed on "{await page.title()}" ({page.url})'

    async def snapshot(self) -> str:
        """Build a text snapshot of the page with numbered interactive refs."""
        page = await self.get_page()

        # The JS tags each element with a stable ref attribute so the click and
        # type tools can find it again even as the DOM changes.
        raw = await page.evaluate(
            _SNAPSHOT_JS,
            [INTERACTIVE_SELECTOR, MAX_SNAPSHOT_ELEMENTS, MAX_LABEL_CHARS],
        )

        lines: list[str] = []
        for i, item in enumerate(raw, start=1):
            if item["href"]:
                lines.append(
                    f'[{i}] {item["role"]} "{item["label"]}" -> {item["href"]}'
                )
            else:
                lines.append(f'[{i}] {item["role"]} "{item["label"]}"')

        self._count = len(raw)

        body_text = ""
        with contextlib.suppress(Exception):
            body_text = await page.locator("body").inner_text(timeout=2_000)
        body_text = " ".join(body_text.split())
        if len(body_text) > MAX_PAGE_TEXT_CHARS:
            body_text = body_text[:MAX_PAGE_TEXT_CHARS] + " ..."

        parts = [f'Page: "{await page.title()}" ({page.url})']
        if lines:
            parts.append("Interactive elements:\n" + "\n".join(lines))
        else:
            parts.append("No interactive elements found.")
        if body_text:
            parts.append("Visible text:\n" + body_text)
        return "\n\n".join(parts)

    async def click(self, ref: int) -> str:
        """Click the element identified by a snapshot ref."""
        el = self._element(ref)
        page = await self.get_page()
        before = page.url
        await el.click(timeout=5_000)
        await self._settle(before)
        self._count = 0
        if page.url != before:
            return (
                f'Clicked [{ref}] and navigated to "{await page.title()}" ({page.url}).'
            )
        return f"Clicked [{ref}]. Take a new snapshot to see the result."

    async def type_text(self, ref: int, text: str, submit: bool) -> str:
        """Fill the field identified by a snapshot ref, optionally submitting."""
        el = self._element(ref)
        page = await self.get_page()
        before = page.url
        await el.fill(text, timeout=5_000)
        if submit:
            await el.press("Enter")
            await self._settle(before, long_wait=True)
            self._count = 0
            if page.url != before:
                return (
                    f'Submitted [{ref}] and navigated to "{await page.title()}" '
                    f"({page.url})."
                )
            return f"Submitted [{ref}]. Take a new snapshot to see the result."
        return f"Filled [{ref}]. Take a new snapshot to see the result."

    async def go_back(self) -> str:
        """Navigate back in history."""
        page = await self.get_page()
        await page.go_back(timeout=10_000)
        self._count = 0
        return f'Back on "{await page.title()}" ({page.url})'

    async def scroll(self, direction: str) -> str:
        """Scroll the page down or up."""
        page = await self.get_page()
        delta = 800 if direction == "down" else -800
        await page.mouse.wheel(0, delta)
        return f"Scrolled {direction}. Take a new snapshot to see the updated page."

    async def _close_browser(self) -> None:
        """Close or disconnect from the browser without touching Playwright."""
        if self._attached:
            # We attached to the user's Edge: disconnect, never close it.
            logger.info("disconnecting from Edge browser")
        else:
            with contextlib.suppress(Exception):
                if self._browser is not None:
                    await self._browser.close()
        self._page = None
        self._browser = None
        self._count = 0
        self._attached = False

    async def close(self) -> None:
        """Release browser resources. Idempotent."""
        await self._close_browser()
        with contextlib.suppress(Exception):
            if self._playwright is not None:
                await self._playwright.stop()
        self._playwright = None
        # Terminate only an Edge process we started ourselves; if we attached to
        # one the user already had open, leave it running.
        if self._edge_proc is not None:
            with contextlib.suppress(Exception):
                self._edge_proc.terminate()
            self._edge_proc = None
        logger.info("browser session closed")

    def _element(self, ref: int) -> Locator:
        if not 1 <= ref <= self._count:
            raise ValueError(
                f"invalid element ref {ref}; take a new snapshot and use a "
                f"ref between 1 and {self._count}"
            )
        assert self._page is not None
        return self._page.locator(f'[data-lk-ref="{ref}"]').first

    async def has_ref(self, ref: int) -> bool:
        """Whether a snapshot ref still exists on the live page."""
        if not 1 <= ref <= self._count:
            return False
        page = self._page
        if page is None:
            return False
        with contextlib.suppress(Exception):
            return await page.locator(f'[data-lk-ref="{ref}"]').count() > 0
        return False

    async def _settle(self, before_url: str, long_wait: bool = False) -> None:
        """Wait for a click or submit to finish navigating, if it navigates.

        Waiting on a fixed timeout either returns too early or wastes time. Watch
        the URL instead, then let the new document settle.
        """
        page = self._page
        if page is None:
            return
        timeout = 8_000 if long_wait else 4_000
        with contextlib.suppress(Exception):
            await page.wait_for_function(
                "(url) => location.href !== url",
                arg=before_url,
                timeout=timeout,
            )
            await page.wait_for_load_state("domcontentloaded", timeout=timeout)


# One session per job. A single process-wide session would be shared by every
# concurrent job in the worker, so one job finishing would close the browser out
# from under the others. Keying by job also survives LiveKit running each tool
# call in its own asyncio task, which a ContextVar cannot: a value set in one
# task is invisible to its siblings, so every tool call would launch a fresh
# browser.
_sessions: dict[object, BrowserSession] = {}


def get_browser_session(session: object) -> BrowserSession:
    """Return the browser session for the given job, creating it if needed.

    `session` is the job's AgentSession. Using the object itself as the key (not
    its id) keeps a strong reference for the job's lifetime, so an id can never
    be reused by a later job and mistake a stale browser for a fresh one.
    """
    browser = _sessions.get(session)
    if browser is None:
        browser = BrowserSession()
        _sessions[session] = browser
    return browser


async def close_browser_session(session: object) -> None:
    """Close the given job's browser session. Safe to call more than once."""
    browser = _sessions.pop(session, None)
    if browser is not None:
        await browser.close()
