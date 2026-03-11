"""Checkpointer factory: Redis when REDIS_URL is set, else in-memory."""

import os

from langgraph.checkpoint.memory import MemorySaver


def get_checkpointer():
    """Return a Redis checkpointer if REDIS_URL is configured, else MemorySaver.

    Falls back gracefully to MemorySaver if Redis is unavailable.
    """
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        try:
            from langgraph.checkpoint.redis import RedisSaver

            saver = RedisSaver.from_conn_string(redis_url)
            saver.setup()
            return saver
        except Exception:
            pass
    return MemorySaver()
