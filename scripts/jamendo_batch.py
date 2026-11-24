"""Batch lookup and bounded parallel recovery of HKU956 Jamendo candidates."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time
import urllib.parse
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.recover_jamendo import JamendoError, metadata_row, download_allowed
from src.common import digest, folder, load_config, read_manifest, write_json
import pandas as pd


def validate_batch(payload, requested):
    headers = payload.get('headers', {})
    if headers.get('status') != 'success' or str(headers.get('code')) != '0':
        raise JamendoError(f"Jamendo API failed, code={headers.get('code', 'unknown')}")
    results = payload.get('results', [])
    ids = [str(t.get('id')) for t in results]
    if len(ids) != len(set(ids)) or not set(ids).issubset(set(requested)):
        raise JamendoError('Duplicate or unexpected track IDs in batch response')
    return {str(t['id']): t for t in results}


def lookup_batch(ids, client_id):
    found = _lookup_once(ids, client_id)
    missing = [sid for sid in ids if sid not in found]
    # Jamendo can omit valid IDs in large batches. Only single-ID misses are definitive for this run.
    if len(ids) > 1 and missing:
        size = 10 if len(ids) > 10 else 1
        for start in range(0, len(missing), size):
            found.update(lookup_batch(missing[start:start + size], client_id))
            time.sleep(.15)
    return found


def _lookup_once(ids, client_id):
    params = {'client_id': client_id, 'format': 'json', 'id': ' '.join(ids), 'limit': 200,
              'type': 'single albumtrack', 'audioformat': 'mp32', 'audiodlformat': 'mp32', 'include': 'licenses'}
    req = urllib.request.Request('https://api.jamendo.com/v3.0/tracks/?' + urllib.parse.urlencode(params),
                                 headers={'User-Agent': 'HKU956-research-recovery/0.1'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return validate_batch(json.load(response), ids)
        except JamendoError:
            raise
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                retry = exc.headers.get('Retry-After', 'unspecified')
                raise JamendoError(f'Rate limited (HTTP 429); Retry-After={retry}. Stop and resume later.') from None
            if attempt == 2:
                raise JamendoError(f'Jamendo HTTP {exc.code}; metadata batch incomplete') from None
            time.sleep(2 ** (attempt + 1))
        except (OSError, ValueError):
            if attempt == 2:
                raise JamendoError('Jamendo metadata request failed after 3 attempts') from None
            time.sleep(1 + attempt)


def download_candidate(track, directory):
    import soundfile as sf
    sid = str(track['id'])
    row = metadata_row(sid, track)
    path = directory / f'{sid}.mp3'
    if not download_allowed(track):
        row['status'] = 'download_not_allowed'
        return row
    if urllib.parse.urlparse(track['audiodownload']).scheme != 'https':
        row['status'] = 'non_https_download_url'
        return row
    for attempt in range(2):
        try:
            if not path.exists():
                part = path.with_suffix('.part')
                with urllib.request.urlopen(track['audiodownload'], timeout=45) as response, part.open('wb') as f:
                    while block := response.read(1024 * 1024):
                        f.write(block)
                sf.info(part)
                part.rename(path)
            info = sf.info(path)
            if info.duration <= 0:
                raise ValueError('empty audio')
            row.update(status='downloaded_candidate', local_path=str(path), sha256=digest(path),
                       decoded_duration_s=info.duration, sample_rate=info.samplerate, channels=info.channels,
                       file_bytes=path.stat().st_size)
            write_json(path.with_suffix('.json'), row)
            return row
        except (OSError, ValueError, RuntimeError):
            if attempt:
                row['status'] = 'audio_download_or_decode_failed'
            else:
                time.sleep(1)
    return row


def run(cfg, client_id, download=False, workers=4, only_missing=False):
    if not client_id:
        raise JamendoError('Set JAMENDO_CLIENT_ID; no client_secret needed')
    manifest = read_manifest(cfg)
    ids = manifest.song_id.drop_duplicates().tolist()
    prefix = 'jamendo_recheck' if only_missing else 'jamendo'
    if only_missing:
        previous = pd.read_csv(folder(cfg, 'reports') / 'jamendo_catalog.csv', dtype={'song_id': str})
        missing = set(previous.loc[previous.status == 'not_found', 'song_id'])
        ids = [sid for sid in ids if sid in missing]
    if any(not sid.isascii() or not sid.isdigit() for sid in ids):
        raise ValueError('Expected numeric HKU song IDs')
    reports = folder(cfg, 'reports')
    directory = folder(cfg, 'audio') / 'jamendo_candidates'
    directory.mkdir(parents=True, exist_ok=True)
    tracks, rows = {}, {}
    for start in range(0, len(ids), 50):
        batch = ids[start:start + 50]
        tracks.update(lookup_batch(batch, client_id))
        for sid in batch:
            rows[sid] = metadata_row(sid, tracks.get(sid))
        pd.DataFrame(rows.values()).to_csv(reports / f'{prefix}_catalog.csv', index=False)
        print(f'Metadata {min(start+50, len(ids))}/{len(ids)}; found {len(tracks)}', flush=True)
        time.sleep(.3)
    if download:
        selected = [t for t in tracks.values() if download_allowed(t)]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(download_candidate, track, directory): str(track['id']) for track in selected}
            for count, future in enumerate(as_completed(futures), 1):
                row = future.result()
                rows[row['song_id']] = row
                if count % 10 == 0 or count == len(selected):
                    pd.DataFrame(rows.values()).to_csv(reports / f'{prefix}_recovery.csv', index=False)
                    print(f'Download {count}/{len(selected)}; successful {sum(r["status"]=="downloaded_candidate" for r in rows.values())}', flush=True)
        for sid, row in rows.items():
            if sid in tracks and not download_allowed(tracks[sid]):
                row['status'] = 'download_not_allowed'
        pd.DataFrame(rows.values()).to_csv(reports / f'{prefix}_recovery.csv', index=False)
    matched = set(tracks)
    allowed = {sid for sid,t in tracks.items() if download_allowed(t)}
    downloaded = {sid for sid,r in rows.items() if r['status'] == 'downloaded_candidate'}
    summary = {'unique_songs': len(ids), 'matched_ids': len(matched), 'download_allowed': len(allowed),
               'download_requested': download, 'downloaded_candidates': len(downloaded),
               'records_with_matched_id': int(manifest.song_id.isin(matched).sum()),
               'records_with_download_allowed': int(manifest.song_id.isin(allowed).sum()),
               'records_with_downloaded_candidate': int(manifest.song_id.isin(downloaded).sum()),
               'local_bytes': sum(r.get('file_bytes',0) for r in rows.values()),
               'recording_identity_verified': False, 'playback_alignment_verified': False}
    write_json(reports / f'{prefix}_batch_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT/'configs/default.yaml'))
    p.add_argument('--download',action='store_true')
    p.add_argument('--only-missing',action='store_true',help='Recheck missing IDs from the initial catalog, keeping separate logs')
    p.add_argument('--workers',type=int,default=4,choices=range(1,9))
    args = p.parse_args()
    try:
        run(load_config(args.config), os.environ.get('JAMENDO_CLIENT_ID','').strip(), args.download,args.workers,args.only_missing)
    except (JamendoError, ValueError) as exc:
        p.exit(2,str(exc)+'\n')

if __name__=='__main__':
    main()
