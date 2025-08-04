from typing import Final, Set, Dict

# Standard HTTP status codes
HTTP_OK: Final[int]                     = 200
HTTP_BAD_REQUEST: Final[int]            = 400
HTTP_UNAUTHORIZED: Final[int]           = 401
HTTP_FORBIDDEN: Final[int]              = 403
HTTP_NOT_FOUND: Final[int]              = 404
HTTP_METHOD_NOT_ALLOWED: Final[int]     = 405
HTTP_UNSUPPORTED_MEDIA_TYPE: Final[int] = 415
HTTP_RATE_LIMITED: Final[int]           = 429
HTTP_INTERNAL_SERVER_ERROR: Final[int]  = 500
HTTP_BAD_GATEWAY: Final[int]            = 502
HTTP_SERVICE_UNAVAILABLE: Final[int]    = 503
HTTP_GATEWAY_TIMEOUT: Final[int]        = 504

# Human-friendly error messages
ERROR_MESSAGES: Final[Dict[int, str]] = {
    HTTP_BAD_REQUEST:            "Bad request",
    HTTP_UNAUTHORIZED:           "Invalid API key",
    HTTP_FORBIDDEN:              "Forbidden",
    HTTP_NOT_FOUND:              "Resource not found",
    HTTP_METHOD_NOT_ALLOWED:     "Method not allowed",
    HTTP_UNSUPPORTED_MEDIA_TYPE: "Unsupported media type",
    HTTP_RATE_LIMITED:           "Rate limit exceeded",
    HTTP_INTERNAL_SERVER_ERROR:  "Internal server error",
    HTTP_BAD_GATEWAY:            "Bad gateway",
    HTTP_SERVICE_UNAVAILABLE:    "Service unavailable",
    HTTP_GATEWAY_TIMEOUT:        "Gateway timeout",
}

# Which codes we should retry automatically
RETRYABLE_STATUS_CODES: Final[Set[int]] = {
    HTTP_RATE_LIMITED,
    HTTP_INTERNAL_SERVER_ERROR,
    HTTP_BAD_GATEWAY,
    HTTP_SERVICE_UNAVAILABLE,
    HTTP_GATEWAY_TIMEOUT,
}

# Which codes indicate a client error we should not retry
CLIENT_ERROR_STATUS_CODES: Final[Set[int]] = {
    HTTP_BAD_REQUEST,
    HTTP_UNAUTHORIZED,
    HTTP_FORBIDDEN,
    HTTP_NOT_FOUND,
    HTTP_METHOD_NOT_ALLOWED,
    HTTP_UNSUPPORTED_MEDIA_TYPE,
}