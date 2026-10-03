"""Black Box target agent: a controlled LangChain laptop-recommendation agent.

Only the LangChain-free event types are re-exported here so that importing the
package stays cheap. Import the agent itself with `from agent.agent import LaptopAgent`.
"""

from .events import Event, EventLog, EventType, format_timeline

__all__ = ["Event", "EventLog", "EventType", "format_timeline"]
