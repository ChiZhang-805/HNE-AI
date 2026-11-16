"""Read the small metadata and command block before the large telemetry download."""
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/python"))
import zipfile_deflate64 as zipfile
BASE = ROOT / "data/new_cohorts/esa-ad-v2"
SIZE = 3776246073
TAIL_START = 3767890531
URL = "https://zenodo.org/records/15237121/files/ESA-Mission1.zip?download=1"


def part(bounds):
    lo, hi = bounds
    response = requests.get(URL, headers={"Range": f"bytes={lo}-{hi}"}, timeout=(30, 90))
    response.raise_for_status()
    if response.headers.get("Content-Range") != f"bytes {lo}-{hi}/{SIZE}":
        raise ValueError("Metadata range mismatch")
    return response.content


class ArchiveParts(io.RawIOBase):
    def __init__(self, head, tail):
        self.head, self.tail, self.pos = head, tail, 0

    def seek(self, offset, whence=0):
        self.pos = offset + (self.pos if whence == 1 else SIZE if whence == 2 else 0)
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def read(self, n=-1):
        n = SIZE-self.pos if n < 0 else min(n, SIZE-self.pos)
        if self.pos >= TAIL_START:
            data = self.tail[self.pos-TAIL_START:self.pos-TAIL_START+n]
        elif self.pos+n <= len(self.head):
            data = self.head[self.pos:self.pos+n]
        else:
            raise ValueError("Requested telemetry bytes are still downloading")
        self.pos += len(data)
        return data


def main():
    tail_path = BASE / "mission1_commands_block.bin"
    if tail_path.exists():
        tail = tail_path.read_bytes()
    else:
        ranges = [(lo, min(lo+(1 << 20)-1, SIZE-1)) for lo in range(TAIL_START, SIZE, 1 << 20)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            tail = b"".join(pool.map(part, ranges))
        tail_path.write_bytes(tail)
    with (BASE / "ESA-Mission1.zip.part").open("rb") as handle:
        head = handle.read(4096)
    dest = BASE / "metadata"
    with zipfile.ZipFile(ArchiveParts(head, tail)) as archive:
        names = [i.filename for i in archive.infolist() if not i.is_dir() and "/channels/" not in i.filename]
        for name in names:
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    print(json.dumps({"metadata_files": len(names), "telecommand_files": sum('/telecommands/' in n for n in names),
                      "output": str(dest)}), flush=True)


if __name__ == "__main__":
    main()
