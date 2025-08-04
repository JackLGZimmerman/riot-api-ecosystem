# file_storage.py
from __future__ import annotations

import csv
import io
from collections import namedtuple
import os
from pathlib import Path
from typing import Iterator, List, Optional, Iterable, Mapping, Any

import zstandard as zstd


def league_v4_load(
    path: str | Path,
    chunk_size: Optional[int] = 1000,
    indexes: Optional[List[int]] = None,  # zero-based positions to keep
) -> Iterator[List[List[Any]]]:
    p = Path(path)
    with open(p, "rb") as raw:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(raw) as decompressed_bin:
            with io.TextIOWrapper(decompressed_bin, encoding="utf-8", newline="") as text_f:
                reader = csv.reader(text_f)
                if chunk_size is None:
                    all_rows: List[List[Any]] = []
                    for row in reader:
                        if indexes is not None:
                            projected = [row[i] for i in indexes if 0 <= i < len(row)]
                        else:
                            projected = row
                        all_rows.append(projected)
                    yield all_rows
                    return

                chunk: List[List[Any]] = []
                for row in reader:
                    if indexes is not None:
                        projected = [row[i] for i in indexes if 0 <= i < len(row)]
                    else:
                        projected = row
                    chunk.append(projected)
                    if len(chunk) >= chunk_size:
                        yield chunk
                        chunk = []
                if chunk:
                    yield chunk

def league_v4_save(
    path: str | Path,
    data: Iterable[List[str]],
    compression_level: int = 15,
):
    p = Path(path)
    cctx = zstd.ZstdCompressor(level=compression_level)

    with open(p, "ab") as raw_out:
        with cctx.stream_writer(raw_out) as compressor_stream:
            with io.TextIOWrapper(compressor_stream, encoding="utf-8", newline="") as text_out:
                try:
                    writer = csv.writer(text_out)
                    for row in data:
                        writer.writerow(row)
                    text_out.flush()
                except:
                    print("There was an error writing the batch for league_v4_save")


league_v4 = {
    "load": league_v4_load,
    "save": league_v4_save
}


def match_v5_load(
    path: str | Path,
    chunk_size: int = 1000,
    indexes: Optional[List[str]] = None,
) -> Iterator[List[dict]]:

    p = Path(path)
    with open(p, "rb") as raw, zstd.ZstdDecompressor().stream_reader(raw) as decompressed_bin:
        with io.TextIOWrapper(decompressed_bin, encoding="utf-8", newline="") as text_f:
            reader = csv.reader(text_f)
            if chunk_size:
                buffer: List[List[str]] = []
                


def match_v5_save(
    path: str | Path,
    data: Iterable[List[str]],
    compression_level: int = 15,
):
    p = Path(path)
    cctx = zstd.ZstdCompressor(level=compression_level)

    with open(p, "ab") as raw_out:
        with cctx.stream_writer(raw_out) as compressor_stream:
            with io.TextIOWrapper(compressor_stream, encoding="utf-8", newline="") as text_out:
                try:
                    writer = csv.writer(text_out)
                    for row in data:
                        writer.writerow(row)
                    text_out.flush()
                except:
                    print("There was an error writing the batch for match_v5_save")


match_v5 = {
    "load": match_v5_load,
    "save": match_v5_save
}
