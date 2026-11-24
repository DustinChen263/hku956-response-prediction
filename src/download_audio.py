"""Fetch only published audio URLs, with resumable successes and explicit failures."""
import urllib.request
from pathlib import Path
import pandas as pd
from .common import parser, load_config, folder, read_manifest, digest


def download_audio(cfg, limit=None):
    df = read_manifest(cfg)
    if 'audio_url' not in df:
        raise ValueError('Manifest has no official audio_url column')
    songs = df[['song_id','audio_url']].drop_duplicates('song_id')
    if limit:
        songs = songs.head(limit)
    out, rows = folder(cfg, 'audio'), []
    for song in songs.itertuples():
        if not str(song.song_id).isalnum():
            raise ValueError('Unsafe song identifier')
        path = out / f'{song.song_id}.mp3'
        try:
            if not path.exists():
                part = path.with_suffix('.part')
                with urllib.request.urlopen(song.audio_url,timeout=20) as r, part.open('wb') as f:
                    while chunk := r.read(1024 * 1024):
                        f.write(chunk)
                import soundfile as sf
                if sf.info(part).duration <= 0:
                    raise ValueError('Downloaded file is not audio')
                part.rename(path)
            rows.append({'song_id': song.song_id, 'url': song.audio_url, 'status': 'ok', 'sha256': digest(path), 'error': ''})
        except Exception as exc:
            rows.append({'song_id': song.song_id, 'url': song.audio_url, 'status': 'failed', 'error': str(exc)})
    pd.DataFrame(rows).to_csv(folder(cfg,'reports') / 'audio_download_log.csv',index=False)
    print(f"Downloaded/verified {sum(r['status']=='ok' for r in rows)}/{len(rows)} songs; rerun audit after downloads.")


def main():
    p = parser(__doc__)
    p.add_argument('--limit',type=int)
    args = p.parse_args()
    download_audio(load_config(args.config), args.limit)

if __name__ == '__main__':
    main()
