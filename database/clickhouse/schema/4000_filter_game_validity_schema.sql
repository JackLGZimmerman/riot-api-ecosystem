DROP VIEW IF EXISTS game_data.mv_filter_game_validity;
DROP TABLE IF EXISTS game_data.filter_game_validity;

CREATE TABLE game_data.filter_game_validity
(
    matchid String,
    teamid UInt8,
    participantid UInt8,
    player_rule_mask UInt32,
    team_rule_mask UInt32,
    game_rule_mask UInt32,
    rule_mask UInt32,
    is_valid UInt8
)
ENGINE = MergeTree
ORDER BY (matchid, teamid, participantid);
