from pathlib import Path
import zstandard as zstd
import io
import csv
import time
import aiofiles
import asyncio
import re
# p = Path(
#     r"C:\Users\Jack\Documents\GitHub\fifth-time-lucky-api"
#     r"\data\database\raw\league"
#     r"\league_players.csv.zst"
# )

# start = time.perf_counter()
# with open(p, 'rb') as bytes:
#     with zstd.ZstdDecompressor().stream_reader(bytes) as decompressed:
#         with io.TextIOWrapper(decompressed, encoding="utf-8", newline="") as text:
#             text = csv.reader(text)

#             count = 0
#             for line in text:
#                 count += 1
#             print(count)

# end = time.perf_counter()
# print(f"{end - start} seconds")

# async def open_file_one():
#     base_path = Path(r"C:\Users\Jack\Documents\GitHub\fifth-time-lucky-api")
#     module_dir = "data/database/raw/match"

#     file_names = [
#         file.name
#         for file in (base_path / module_dir).iterdir()
#         if file.is_file() and file.name.startswith("collected_players")
#     ]

#     if not file_names:
#         print("No matching files found")
#         return

#     file_name = file_names[0]
#     p = base_path / module_dir / file_name

#     async with aiofiles.open(p, "rb") as f:
#         compressed = await f.read()

#     decompressed = zstd.ZstdDecompressor().decompress(compressed)
#     text = io.StringIO(decompressed.decode("utf-8"))
#     rows = csv.reader(text)

#     count = 0
#     for row in rows:
#         count += 1
#         print(row)
#         if count == 5:
#             break

# async def open_file_two():

#     base_path = Path(r"C:\Users\Jack\Documents\GitHub\fifth-time-lucky-api")
#     module_dir = r'data/database/raw/match'
#     file_name = r'matchids.csv.zst'

#     file_path = base_path / module_dir / file_name


#     async with aiofiles.open(file_path, 'rb') as f:
#         compressed_bytes = await f.read()
#         decompress = zstd.ZstdDecompressor().decompress(compressed_bytes)
#         text = io.StringIO(decompress.decode("utf-8"))
#         rows = csv.reader(text)
#         count = 0
#         for row in rows:
#             count += 1
#             print(row)
#             if count > 5:
#                 break


# if __name__ == "__main__":
#     # asyncio.run(open_file_one())
#     asyncio.run(open_file_two())
from config import settings
from utils import storages
from typing import List
import zstandard as zstd
# values = [1000, 500, 100, 50, 10, 5, 1]
# num = 1234
# it = iter(values)
# next = next(it)
# print(next)
# print(next)
# digits = [int(digit) for digit in str(num)]
# print(digits)


dir_path = Path(settings.base_project_path) / "data/database/raw/match"

if dir_path.is_dir():
    print(f"[OK] Path is a directory as expected")


file_name = "matchids.csv.zst"
# pattern = re.compile(r"^(collected_players_)(.*)")
# for child in dir_path.iterdir():
#     if pattern.match(str(child.name)):
#         file_name = child.name

if file_name:
    file_path = dir_path / file_name


async def get_data(file_path: Path):
    ztdDecompressor = zstd.ZstdDecompressor()
    batch = []
    with open(file_path, "rb") as fh:
        decompressed_bytes = ztdDecompressor.stream_reader(fh)
        text_stream = io.TextIOWrapper(decompressed_bytes)
        text = csv.reader(text_stream)

        for row in text:
            batch.append(row)
            if len(batch) >= 1000:
                ids = [Id for matchid_list in batch for Id in matchid_list]
                print("batch:", len(batch), "ids:", len(ids))
                yield batch
                batch = []

        if batch:
            yield batch


async def main():
    async for batch in get_data(file_path):
        print("Got batch with", len(batch), "rows")


asyncio.run(main())
