from ..base import BaseORJSONModel
from pydantic import ConfigDict
from typing import List




#-----------------------------------------------------------------------------#
#                 /lol/match/v5/matches/by-puuid/{puuid}/ids                  #
#-----------------------------------------------------------------------------#

MatchIds = List[str]

#-----------------------------------------------------------------------------#
#                        /lol/match/v5/matches/{matchId}                      #
#-----------------------------------------------------------------------------#

class MatchDto(BaseORJSONModel):
    model_config = ConfigDict(slots=True)


class MetadataDto(BaseORJSONModel):
    model_config = ConfigDict(slots=True)


class InfoDto(BaseORJSONModel):
    model_config = ConfigDict(slots=True)


class ParticipantDto(BaseORJSONModel):
    model_config = ConfigDict(slots=True)


class ChallengesDto(BaseORJSONModel):
    model_config = ConfigDict(slots=True)


class MissionsDto(BaseORJSONModel):
    model_config = ConfigDict(slots=True)


class PerksDto(BaseORJSONModel):
    model_config = ConfigDict(slots=True)


class PerkStatsDto(BaseORJSONModel):
    model_config = ConfigDict(slots=True)


class PerkStyleDto(BaseORJSONModel):
    model_config = ConfigDict(slots=True)


class PerkStyleSelectionDto(BaseORJSONModel):
    model_config = ConfigDict(slots=True)


class TeamDto(BaseORJSONModel):
    model_config = ConfigDict(slots=True)


class BanDto(BaseORJSONModel):
    model_config = ConfigDict(slots=True)


class ObjectivesDto(BaseORJSONModel):
    model_config = ConfigDict(slots=True)


class ObjectiveDto(BaseORJSONModel):
    first: bool
    kills: int