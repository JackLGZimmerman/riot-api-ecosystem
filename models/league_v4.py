from pydantic import BaseModel, RootModel
from config.constants import EliteTiers, Queues, Tiers, Divisions  # Enums


# ---------- Elite ----------
class EliteBoundConfig(BaseModel):
    collect: bool = True  # <— new
    upper: EliteTiers | None = None  # None = no upper bound
    lower: EliteTiers | None = None  # None = no lower bound


class EliteBoundsConfig(RootModel[dict[Queues, EliteBoundConfig]]):
    """Map Queue -> Elite bounds config"""

    pass


# ---------- Basic ----------
class BasicBoundSubConfig(BaseModel):
    tier: Tiers | None = None
    division: Divisions | None = None


class BasicBoundConfig(BaseModel):
    collect: bool = True  # <— new
    upper: BasicBoundSubConfig
    lower: BasicBoundSubConfig


class BasicBoundsConfig(RootModel[dict[Queues, BasicBoundConfig]]):
    """Map Queue -> Basic bounds config"""

    pass