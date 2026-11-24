"""Download the public Figshare archive; never overwrite a differing local file."""
from pathlib import Path
import hashlib
import json
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
def main():
    raw = ROOT / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen('https://api.figshare.com/v2/articles/21080821', timeout=60) as r:
        meta = json.load(r)
    (raw / 'figshare_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf8')
    for entry in meta['files']:
        path = raw / Path(entry['name']).name
        if not path.exists():
            tmp = path.with_suffix('.part')
            with urllib.request.urlopen(entry['download_url'], timeout=120) as r, tmp.open('wb') as f:
                while chunk := r.read(1024 * 1024):
                    f.write(chunk)
            tmp.rename(path)
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest != entry['computed_md5']:
            raise ValueError(f'Checksum mismatch: {path}')
        if zipfile.is_zipfile(path):
            dest = raw / 'HKU956'
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    target = (dest / member.filename).resolve()
                    if not target.is_relative_to(dest.resolve()):
                        raise ValueError('Unsafe archive path')
                    if not target.exists():
                        archive.extract(member, dest)
        print(f'Verified {path.name}: {path.stat().st_size} bytes')
if __name__ == '__main__':
    main()
