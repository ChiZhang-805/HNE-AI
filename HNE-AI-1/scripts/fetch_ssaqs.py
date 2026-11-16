"""Download the official SSAQS release with resumable byte ranges."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://zenodo.org/api/records/18706837/files/SSAQS%20dataset.zip/content'
MD5 = '3a95e68847192c062daf80df25cceb83'


def main():
    folder = ROOT / 'data/new_cohorts/ssaqs-2026'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / 'SSAQS dataset.zip'
    with requests.get(URL, headers={'Range': 'bytes=0-0'}, stream=True, timeout=45) as response:
        response.raise_for_status()
        total = int(response.headers['Content-Range'].split('/')[-1])
    offset = target.stat().st_size if target.exists() else 0
    ranges = [(lo, min(lo+(1 << 20)-1, total-1)) for lo in range(offset, total, 1 << 20)]

    def download(bounds):
        lo, hi = bounds
        path = folder / f'download-{lo}-{hi}.part'
        if path.exists() and path.stat().st_size == hi-lo+1:
            return path
        for attempt in range(3):
            try:
                r = requests.get(URL, headers={'Range': f'bytes={lo}-{hi}'}, timeout=(20, 60))
                r.raise_for_status()
                if r.headers.get('Content-Range') != f'bytes {lo}-{hi}/{total}' or len(r.content) != hi-lo+1:
                    raise ValueError('Server returned a different byte range')
                path.write_bytes(r.content)
                print(f'Downloaded bytes {lo}-{hi}', flush=True)
                return path
            except requests.RequestException:
                if attempt == 2:
                    raise
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(download, ranges))
    with target.open('ab') as handle:
        for path in paths:
            handle.write(path.read_bytes())
    with target.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'md5').hexdigest()
    if digest != MD5:
        raise ValueError('Archive checksum differs from the official release')
    for path in paths:
        path.unlink()
    record = {'url': URL, 'bytes': target.stat().st_size, 'md5': digest,
              'paper': 'https://doi.org/10.1038/s41597-026-07085-7'}
    (folder / 'source.json').write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
