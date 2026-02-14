import logging
import threading

import clickhouse_connect

from app.core.config.settings import settings
from app.core.logging.logger import setup_logging_config

setup_logging_config()
logger = logging.getLogger(__name__)

_local = threading.local()


def get_client():
    client = getattr(_local, "client", None)
    if client is None:
        client = clickhouse_connect.get_client(
            host=settings.clickhouse_host,
            port=settings.clickhouse_port,
            username=settings.clickhouse_user,
            password=settings.clickhouse_password.get_secret_value(),
            # We do not rely on ClickHouse HTTP sessions (temp tables, etc).
            # Disabling auto-session avoids "concurrent queries within the same session"
            # when work is fanned out across multiple threads.
            autogenerate_session_id=False,
        )

        # Force connection/auth/protocol negotiation for this thread-local session.
        client.command("SELECT 1")
        _local.client = client

    return client
