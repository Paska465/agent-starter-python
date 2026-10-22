"""Browser control tools for the Iconic agent, backed by Playwright.

The LLM drives the user's own Microsoft Edge through a small, focused tool set: navigate,
snapshot, click, type, back, scroll, wait, and close. Elements are addressed by
the numbered refs returned by browser_snapshot.

Each job gets its own browser session, keyed by the job's AgentSession (see
browser.get_browser_session), so concurrent jobs never share or close each
other's browser.
"""

import asyncio
import logging
from typing import Literal

from livekit.agents import RunContext, ToolError, function_tool

from browser import get_browser_session

logger = logging.getLogger("browser-tools")


@function_tool()
async def browser_navigate(context: RunContext, url: str) -> str:
    """Open a website in the user's browser. Always call this first.

    The page opens in the user's Microsoft Edge window, which they can see on
    screen, so any previous page in that window is replaced. Use a bare domain
    like "livekit.io" or a full address like "https://example.com/news". Do not
    use it for simple lookups; the web search tool is faster for those.

    Args:
        url: The web address to open.
    """
    session = get_browser_session(context.session)
    async with session.lock:
        try:
            result = await session.navigate(url)
            logger.info("browser_navigate %s -> %s", url, result)
            return result
        except Exception as e:
            logger.error("browser_navigate failed for %s: %s", url, e)
            raise ToolError(
                f"Could not open {url}. Check the address and try again."
            ) from e


@function_tool()
async def browser_snapshot(context: RunContext) -> str:
    """Read the current page: numbered interactive elements plus visible text.

    Call this after navigating, clicking, typing, or scrolling to see the
    updated page. Use the element numbers from this snapshot with the click
    and type tools; they become invalid after any page change.
    """
    session = get_browser_session(context.session)
    async with session.lock:
        try:
            return await session.snapshot()
        except Exception as e:
            logger.error("browser_snapshot failed: %s", e)
            raise ToolError(
                "Could not read the page. It may have closed; navigate again."
            ) from e


@function_tool()
async def browser_click(context: RunContext, ref: int) -> str:
    """Click an interactive element on the current page by its number.

    Only use numbers returned by the most recent browser_snapshot. If the
    number is rejected, take a new snapshot first.

    Args:
        ref: The element number from the latest snapshot, such as 3.
    """
    # A click can trigger an irreversible action, so keep it from being
    # interrupted partway through.
    context.disallow_interruptions()
    session = get_browser_session(context.session)
    async with session.lock:
        try:
            if not await session.has_ref(ref):
                snap = await session.snapshot()
                raise ToolError(
                    f"Element [{ref}] is no longer on the page. Here is the "
                    f"current page; pick the right element and try again."
                    f"\n\n{snap}"
                )
            result = await session.click(ref)
            logger.info("browser_click [%s]", ref)
            return result
        except ToolError:
            raise
        except ValueError as e:
            raise ToolError(str(e)) from e
        except Exception as e:
            logger.error("browser_click [%s] failed: %s", ref, e)
            raise ToolError(
                f"Could not click element [{ref}]. Take a new snapshot."
            ) from e


@function_tool()
async def browser_type(
    context: RunContext,
    ref: int,
    text: str,
    submit: bool = False,
) -> str:
    """Type text into a field on the current page by its element number.

    Replaces any existing content of the field. Set submit to true to press
    Enter afterwards, for example to run a search or sign in.

    Args:
        ref: The field number from the latest snapshot.
        text: The text to type.
        submit: Press Enter after typing. Defaults to false.
    """
    if submit:
        # Submitting a form can trigger an irreversible action, so do not let
        # user speech interrupt it halfway through.
        context.disallow_interruptions()
    session = get_browser_session(context.session)
    async with session.lock:
        try:
            if not await session.has_ref(ref):
                snap = await session.snapshot()
                raise ToolError(
                    f"Field [{ref}] is no longer on the page. Here is the "
                    f"current page; pick the right field and try again."
                    f"\n\n{snap}"
                )
            result = await session.type_text(ref, text, submit)
            logger.info("browser_type [%s] submit=%s", ref, submit)
            return result
        except ToolError:
            raise
        except ValueError as e:
            raise ToolError(str(e)) from e
        except Exception as e:
            logger.error("browser_type [%s] failed: %s", ref, e)
            raise ToolError(f"Could not type into [{ref}]. Take a new snapshot.") from e


@function_tool()
async def browser_go_back(context: RunContext) -> str:
    """Go back to the previous page in the browser's history."""
    session = get_browser_session(context.session)
    async with session.lock:
        try:
            return await session.go_back()
        except Exception as e:
            logger.error("browser_go_back failed: %s", e)
            raise ToolError("Could not go back; there may be no previous page.") from e


@function_tool()
async def browser_scroll(
    context: RunContext,
    direction: Literal["down", "up"],
) -> str:
    """Scroll the current page to see content below or above the fold.

    Args:
        direction: Must be "down" or "up".
    """
    session = get_browser_session(context.session)
    async with session.lock:
        try:
            return await session.scroll(direction)
        except Exception as e:
            logger.error("browser_scroll %s failed: %s", direction, e)
            raise ToolError("Could not scroll the page.") from e


@function_tool()
async def browser_wait(context: RunContext, seconds: int) -> str:
    """Wait for a slow page to finish loading or updating.

    Use sparingly; prefer taking a snapshot to check the page state.

    Args:
        seconds: How long to wait, between 1 and 10.
    """
    seconds = max(1, min(seconds, 10))
    session = get_browser_session(context.session)
    async with session.lock:
        await asyncio.sleep(seconds)
        return f"Waited {seconds} seconds. Take a new snapshot."


@function_tool()
async def browser_close(context: RunContext) -> str:
    """Close the browser when the user is finished browsing.

    Ends the browsing session. If the agent is driving a browser it launched
    itself, that window closes; if it attached to an Edge window the user
    already had open, that window is left alone. The browser reopens
    automatically on the next browser_navigate call.
    """
    session = get_browser_session(context.session)
    async with session.lock:
        await session.close()
        return "Closed the browsing session."
