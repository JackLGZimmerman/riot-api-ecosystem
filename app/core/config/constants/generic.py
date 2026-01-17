from http import HTTPStatus
from typing import Any, TypeAlias

JSON: TypeAlias = dict[str, Any]
JSONList: TypeAlias = list[JSON]

RETRYABLE = {
    HTTPStatus.TOO_MANY_REQUESTS.value,  # 429
    HTTPStatus.INTERNAL_SERVER_ERROR.value,  # 500
    HTTPStatus.BAD_GATEWAY.value,  # 502
    HTTPStatus.SERVICE_UNAVAILABLE.value,  # 503
    HTTPStatus.GATEWAY_TIMEOUT.value,  # 504
}
