# league_v4.py

if __name__ == "__main__":
    elite_input: dict[str, dict[str, Any]] = {
        "RANKED_SOLO_5x5": {
            "collect": True,
            "upper": None,
            "lower": "GRANDMASTER",
        },
        "RANKED_FLEX_SR": {
            "collect": True,
            "upper": None,
            "lower": "MASTER",
        },
    }

    basic_input: dict[str, dict[str, Any]] = {
        "RANKED_SOLO_5x5": {
            "collect": True,
            "upper_tier": "DIAMOND",
            "upper_division": "I",
            "lower_tier": "EMERALD",
            "lower_division": "II",
        },
        "RANKED_FLEX_SR": {
            "collect": True,
            "upper_tier": None,
            "upper_division": None,
            "lower_tier": "DIAMOND",
            "lower_division": "II",
        },
    }

    elite_bounds: EliteBoundsConfig = parse_elite_bounds(elite_input)
    basic_bounds: BasicBoundsConfig = parse_basic_bounds(basic_input)

    for queue, cfg in elite_bounds.items():
        print("elite", queue, cfg)

    for queue, cfg in basic_bounds.items():
        print("basic", queue, cfg)
