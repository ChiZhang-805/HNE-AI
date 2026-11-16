"""Download the recorded FCM/CATCH modules and apply our compatibility changes."""
from pathlib import Path
import json
import subprocess
import requests

ROOT = Path(__file__).resolve().parents[1]
session = requests.Session()

for method in ['FCM', 'CATCH']:
    folder = ROOT / 'third_party' / method
    metadata = json.loads((folder / 'SOURCE.json').read_text(encoding='utf-8'))
    repository = metadata['repository'].removeprefix('https://github.com/')
    for name in metadata['copied_files']:
        if name in metadata['empty_files']:
            content = b''
        else:
            response = session.get(
                f"https://raw.githubusercontent.com/{repository}/{metadata['commit']}/{name}", timeout=60)
            response.raise_for_status()
            content = response.content
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    patch = ROOT / metadata['patch_file']
    if patch.stat().st_size:
        top = Path(subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], cwd=ROOT, text=True).strip())
        destination = folder.relative_to(top).as_posix()
        subprocess.run(['git', 'apply', '--ignore-whitespace', f'--directory={destination}',
                        str(patch)], cwd=top, check=True)
    print(method, metadata['commit'], 'downloaded and patched', flush=True)
