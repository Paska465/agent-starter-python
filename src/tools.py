import asyncio
import logging

from ddgs import DDGS
from livekit.agents import RunContext, ToolError, function_tool

logger = logging.getLogger("tools")


@function_tool()
async def search_web(context: RunContext, query: str) -> str:
    """Search the web for information related to the given query.

    Use this for quick facts, headlines, and simple lookups. It is much faster
    than browsing. Use the browser tools instead when the user wants to visit,
    read, or interact with a specific website.

    Args:
        query: The search query, for example "livekit cloud pricing".
    """
    try:
        # DDGS is a blocking, network-bound library, so run it in a worker
        # thread. Calling it on the event loop stalls audio and turn handling.
        result = await asyncio.to_thread(DDGS().text, query, max_results=5)
    except Exception as e:
        logger.error("Web search failed for query '%s': %s", query, e)
        raise ToolError("The web search failed. Try again or use the browser.") from e

    if not result:
        return f"No search results found for '{query}'."

    lines = []
    for item in result:
        title = item.get("title", "").strip()
        body = " ".join(item.get("body", "").split())
        lines.append(f"{title}: {body}")
    summary = "\n".join(lines)
    logger.info("Web search result for query '%s': %s", query, summary[:200])
    return summary
