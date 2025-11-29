from pathlib import Path
import io
import zstandard as zstd

PATH = Path("/home/jack/riot-api-ecosystem/data/snapshots/elite_20251128_104339.zst")

dctx = zstd.ZstdDecompressor()

with open(PATH, "rb") as f:
    with dctx.stream_reader(f) as reader:
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")

        for i, line in enumerate(text_stream, start=1):
            print(line.rstrip())
            if i >= 10:
                break
