-- noqa: disable=LT01,LT05,PRS
--
-- Coarsest champion-pair backoff for 6004: same-team 2vx synergy priors keyed
-- on championid only (build and teamposition dropped). Canonicalised
-- smaller-id-first (win_rate is own-team, symmetric).

DROP TABLE IF EXISTS game_data_filtered.synergy_2vx_champ;

CREATE TABLE IF NOT EXISTS game_data_filtered.synergy_2vx_champ
(
    split LowCardinality(String),
    championid_1 Int32,
    championid_2 Int32,
    matchups UInt64,
    win_rate Float32
)
ENGINE = MergeTree
ORDER BY (split, championid_1, championid_2);
