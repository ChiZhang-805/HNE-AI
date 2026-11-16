"""Fetch the upstream HYDRA implementation used by the public baseline runner."""
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import urlopen

COMMIT = '144bb7aa3186654042f00f36b078370c620acb06'
FILES = {
    'code/hydra.py': '56de99eb1723567ee0a2b3763513ecb4a645d8b1732116c50a07af2b2e3a2d4f',
    'code/hydra_multivariate.py': 'ce7181887d8696f270c979b91f555ae4ed59607ddc5fea2701fe9a021c958c67',
    'LICENSE': '3972dc9744f6499f0f9b2dbf76696f2ae7ad8af9b23dde66d6af86c9dfb36986',
}


def main():
    dest = Path(__file__).resolve().parents[1] / 'cache/hydra'
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest) as scratch:
        for path, sha256 in FILES.items():
            target = dest / Path(path).name
            if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == sha256:
                continue
            url = f'https://raw.githubusercontent.com/angus924/hydra/{COMMIT}/{path}'
            with urlopen(url, timeout=30) as response:
                content = response.read()
            if hashlib.sha256(content).hexdigest() != sha256:
                raise ValueError(f'Unexpected upstream content for {path}')
            (Path(scratch) / target.name).write_bytes(content)
        for file in Path(scratch).iterdir():
            os.replace(file, dest / file.name)
    print('Verified both HYDRA modules and the upstream license.')


if __name__ == '__main__':
    main()
