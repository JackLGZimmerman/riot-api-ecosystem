import logging

import clickhouse_connect

from app.core.config.settings import settings
from app.core.logging.logger import setup_logging_config

setup_logging_config()
logger = logging.getLogger(__name__)

_client = None


def get_client():
    global _client
    if _client is None:
        _client = clickhouse_connect.get_client(
            host=settings.clickhouse_host,
            port=settings.clickhouse_port,
            username=settings.clickhouse_user,
            password=settings.clickhouse_password.get_secret_value(),
            database=settings.clickhouse_database,
        )

        _client.command("SELECT 1")

    return _client
