import orjson
from pydantic import BaseModel, ConfigDict

def _orjson_dumps(v, *, default):
    # pydantic expects a str, orjson returns bytes
    return orjson.dumps(v, default=default).decode()

class BaseORJSONModel(BaseModel):
    model_config = ConfigDict(
        extra='ignore', 
    )