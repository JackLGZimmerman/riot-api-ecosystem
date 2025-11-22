# app/core/config/bounds.py
from typing import Any, Dict

from pydantic import BaseModel, TypeAdapter

from app.core.config.constants.parameters import (
    Divisions,
    EliteTiers,
    Queues,
    Tiers,
)

# ---------- Config models ----------


class EliteBoundConfig(BaseModel):
    collect: bool = True
    upper: EliteTiers | None = None
    lower: EliteTiers | None = None


class BasicBoundConfig(BaseModel):
    collect: bool = True
    upper_tier: Tiers | None = None
    upper_division: Divisions | None = None
    lower_tier: Tiers | None = None
    lower_division: Divisions | None = None


EliteBoundsConfig = Dict[Queues, EliteBoundConfig]
BasicBoundsConfig = Dict[Queues, BasicBoundConfig]


# ---------- TypeAdapters + parse helpers ----------

_elite_bounds_adapter = TypeAdapter(EliteBoundsConfig)
_basic_bounds_adapter = TypeAdapter(BasicBoundsConfig)


def parse_elite_bounds(data: Any) -> EliteBoundsConfig:
    """
    Validate and coerce an incoming elite-bounds payload into
    a dict[Queues, EliteBoundConfig].
    """
    return _elite_bounds_adapter.validate_python(data)


def parse_basic_bounds(data: Any) -> BasicBoundsConfig:
    """
    Validate and coerce an incoming basic-bounds payload into
    a dict[Queues, BasicBoundConfig].
    """
    return _basic_bounds_adapter.validate_python(data)
