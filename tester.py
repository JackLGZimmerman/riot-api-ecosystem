from pathlib import Path

import zstandard as zstd

players = Path("lake/pipelines/players/info/playerinfo.jsonl.zst")
puuids = Path("lake/pipelines/players/puuids/playerpuuids.csv.zst")
match_ids = Path("lake/pipelines/matchids/matchids/matchids_000000.csv.zst")
match_ids_puuids = Path(
    "lake/pipelines/matchids/puuids/puuidmatchids_1768201236.csv.zst"
)

cctx = zstd.ZstdDecompressor()


# with puuids.open("rb") as fb, cctx.stream_reader(fb) as stream:
#     puuids = []
#     text = io.TextIOWrapper(stream, encoding="utf-8", newline="")
#     for line in text:
#         puuids.append(line)

# with match_ids_puuids.open("rb") as fb, cctx.stream_reader(fb) as stream:
#     match_ids_puuids = []
#     text = io.TextIOWrapper(stream, encoding="utf-8", newline="")
#     for line in text:
#         match_ids_puuids.append(line)

# print(",".join(sorted(match_ids_puuids)))
# print(",".join(sorted(puuids)))


# for idx, (puuid, match_ids_puuid) in enumerate(
#     zip(sorted(puuids), sorted(match_ids_puuids))
# ):
#     if puuid.strip() != match_ids_puuid.strip():
#         print(f"Found issue at position: {idx}")
#         print(puuid, "<==>", match_ids_puuid)

# print(len(match_ids_puuids), len(puuids))
# print(set(sorted(match_ids_puuids)) == set(sorted(puuids)))

size_bytes_players = players.stat().st_size
size_bytes_puuids = puuids.stat().st_size
size_bytes_match_ids = match_ids.stat().st_size
size_bytes_match_ids_puuids = match_ids_puuids.stat().st_size

print(f"{players}: {size_bytes_players} bytes")
print(f"{puuids}: {size_bytes_puuids} bytes")
print(f"{match_ids}: {size_bytes_match_ids} bytes")
print(f"{match_ids_puuids}: {size_bytes_match_ids_puuids} bytes")
