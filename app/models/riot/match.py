from pydantic import BaseModel
from typing import List


# -----------------------------------------------------------------------------#
#                 /lol/match/v5/matches/by-puuid/{puuid}/ids                  #
# -----------------------------------------------------------------------------#

MatchIds = List[str]

# -----------------------------------------------------------------------------#
#                        /lol/match/v5/matches/{matchId}                      #
# -----------------------------------------------------------------------------#


class MatchDto(BaseModel):
    pass

class MetadataDto(BaseModel):
    pass

class InfoDto(BaseModel):
    pass

class ParticipantDto(BaseModel):
    pass

class ChallengesDto(BaseModel):
    pass

class MissionsDto(BaseModel):
    pass

class PerksDto(BaseModel):
    pass

class PerkStatsDto(BaseModel):
    pass

class PerkStyleDto(BaseModel):
    pass

class PerkStyleSelectionDto(BaseModel):
    pass

class TeamDto(BaseModel):
    pass

class BanDto(BaseModel):
    pass

class ObjectivesDto(BaseModel):
    pass

class ObjectiveDto(BaseModel):
    first: bool
    kills: int
