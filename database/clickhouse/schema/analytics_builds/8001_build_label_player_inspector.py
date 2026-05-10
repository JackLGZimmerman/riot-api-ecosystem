"""
Run 8001_build_label_player_inspector.sql and open results as an HTML table
with item images rendered inline. Opens automatically in the default browser.

Usage:
    python 8001_build_label_player_inspector.py [champion_id] [team_position] [build_label]

Examples:
    python 8001_build_label_player_inspector.py
    python 8001_build_label_player_inspector.py 2 JUNGLE mr_tank

The underlying participant_item_value_totals table is built by 5133 against
item_value_map_dict with the composite key (championid, teamposition, itemid).
Specific (championid, teamposition) rows in item_value_map.jsonl are applied
to that exact pair; rows with NULL championid/teamposition act as the general
fallback. The inspector filters directly on those resolved totals.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import os


def configure_local_clickhouse_host():
    host = os.environ.get("CLICKHOUSE_HOST")
    if host in (None, "", "clickhouse"):
        os.environ["CLICKHOUSE_HOST"] = "localhost"


configure_local_clickhouse_host()

from database.clickhouse.client import get_client

ITEM_COLS = ["item0", "item1", "item2", "item3", "item4", "item5", "item6", "roleBoundItem"]

SQL = """
WITH params AS (
    SELECT
        toInt32({champion_id}) AS champion_id,
        '{team_position}' AS team_position,
        '{build_label}' AS build_label
)
SELECT
    ps.matchid AS gameid,
    ps.teamid,
    ps.championid,
    dictGet('game_data.championid_name_map_dict', 'name', toInt32(ps.championid)) AS champion_name,
    dictGet('game_data.item_info_dict', 'image', toUInt32(ps.item0)) AS item0,
    dictGet('game_data.item_info_dict', 'image', toUInt32(ps.item1)) AS item1,
    dictGet('game_data.item_info_dict', 'image', toUInt32(ps.item2)) AS item2,
    dictGet('game_data.item_info_dict', 'image', toUInt32(ps.item3)) AS item3,
    dictGet('game_data.item_info_dict', 'image', toUInt32(ps.item4)) AS item4,
    dictGet('game_data.item_info_dict', 'image', toUInt32(ps.item5)) AS item5,
    dictGet('game_data.item_info_dict', 'image', toUInt32(ps.item6)) AS item6,
    if(
        isNull(ps.rolebounditem),
        '',
        dictGet('game_data.item_info_dict', 'image', toUInt32(assumeNotNull(ps.rolebounditem)))
    ) AS roleBoundItem,
    ps.kills,
    ps.deaths,
    ps.assists,
    (ps.kills + ps.assists) / ps.deaths AS KDA,
    if(ps.win = 1, 'WIN', 'LOSS') AS outcome
FROM game_data_filtered.participant_item_value_totals AS ivt
INNER JOIN game_data_filtered.participant_stats AS ps
    ON ivt.matchid = ps.matchid AND ivt.participantid = ps.participantid
CROSS JOIN params
WHERE
    ivt.championid = params.champion_id
    AND ivt.teamposition = params.team_position
    AND ivt.highest_value_label = params.build_label
ORDER BY ps.matchid, ps.teamid, ps.participantid
"""


def build_html(rows, columns, champion_id, team_position, build_label):
    title = f"Champion {champion_id} · {team_position} · {build_label}"

    header = "".join(f"<th>{c}</th>" for c in columns)

    def render_cell(col, val):
        if col in ITEM_COLS and val:
            return f'<td><img src="{val}" title="{val}" onerror="this.style.display=\'none\'"></td>'
        return f"<td>{val}</td>"

    body_rows = []
    for row in rows:
        cells = "".join(render_cell(col, val) for col, val in zip(columns, row))
        body_rows.append(f"<tr>{cells}</tr>")
    body = "\n".join(body_rows)

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: sans-serif; padding: 1rem; }}
    h2 {{ margin-bottom: 0.5rem; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; vertical-align: middle; }}
    th {{ background: #f4f4f4; }}
    tr:nth-child(even) {{ background: #fafafa; }}
    img {{ width: 40px; height: 40px; }}
  </style>
</head>
<body>
  <h2>{title}</h2>
  <p>{len(rows)} rows</p>
  <table>
    <thead><tr>{header}</tr></thead>
    <tbody>{body}</tbody>
  </table>
</body>
</html>"""


def main():
    champion_id = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    team_position = sys.argv[2] if len(sys.argv) > 2 else "JUNGLE"
    build_label = sys.argv[3] if len(sys.argv) > 3 else "mr_tank"

    sql = SQL.format(
        champion_id=champion_id,
        team_position=team_position,
        build_label=build_label,
    )

    client = get_client()
    result = client.query(sql)

    html = build_html(
        result.result_rows, result.column_names, champion_id, team_position, build_label
    )

    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(html)
        path = f.name

    win_path = subprocess.check_output(["wslpath", "-w", path]).decode().strip()
    subprocess.run(["explorer.exe", win_path], check=False)
    print(f"Opened {path} ({len(result.result_rows)} rows)")


if __name__ == "__main__":
    main()
