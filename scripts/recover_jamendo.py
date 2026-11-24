"""Recover candidate HKU956 recordings via the official Jamendo read API.

Credentials are read only from JAMENDO_CLIENT_ID, never written to config/logs.
Recovered audio stays separate until recording identity and alignment are verified.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.common import digest, folder, load_config, read_manifest, write_json
import pandas as pd


class JamendoError(RuntimeError):
    pass


def validate_response(payload, song_id):
    headers = payload.get('headers', {})
    if headers.get('status') != 'success' or headers.get('code') not in (0, '0'):
        # Do not echo a server error message: it may contain request credentials.
        raise JamendoError(f"Jamendo API failed (code {headers.get('code', 'unknown')}); check application status")
    results = payload.get('results', [])
    if not results:
        return None
    if len(results) != 1 or str(results[0].get('id')) != str(song_id):
        raise JamendoError('API did not return exactly the requested song ID')
    return results[0]


def download_allowed(track):
    return track.get('audiodownload_allowed') is True and bool(track.get('audiodownload'))


def lookup(song_id, client_id):
    query = urllib.parse.urlencode({'client_id': client_id, 'format': 'json', 'id': song_id,
                                   'type': 'single albumtrack', 'audioformat': 'mp32',
                                   'audiodlformat': 'mp32', 'include': 'licenses'})
    request = urllib.request.Request('https://api.jamendo.com/v3.0/tracks/?' + query,
                                     headers={'User-Agent': 'HKU956-research-recovery/0.1'})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        raise JamendoError(f'Jamendo HTTP {exc.code}') from None
    except (OSError, ValueError):
        raise JamendoError('Jamendo connection or JSON decoding failed') from None
    return validate_response(payload, song_id)


def metadata_row(song_id, track):
    if track is None:
        return {'song_id': song_id, 'status': 'not_found'}
    return {'song_id': song_id, 'status': 'matched_id_only', 'track_name': track.get('name', ''),
            'artist_name': track.get('artist_name', ''), 'album_name': track.get('album_name', ''),
            'jamendo_duration_s': track.get('duration'), 'license_url': track.get('license_ccurl', ''),
            'download_allowed': download_allowed(track), 'recording_identity_verified': False,
            'playback_alignment_verified': False}


def recover(cfg, client_id, limit=3, download=False):
    if not client_id:
        raise JamendoError('Missing JAMENDO_CLIENT_ID. Register a Read only app at https://devportal.jamendo.com/; no client_secret needed.')
    manifest = read_manifest(cfg)
    ids = manifest.song_id.drop_duplicates().tolist()
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive')
        ids = ids[:limit]
    out = folder(cfg, 'audio') / 'jamendo_candidates'
    out.mkdir(parents=True, exist_ok=True)
    reports = folder(cfg, 'reports')
    rows = []
    log = reports / 'jamendo_recovery.csv'
    previous = pd.read_csv(log, dtype={'song_id': str}).to_dict('records') if log.exists() else []
    try:
        for song_id in ids:
            if not str(song_id).isascii() or not str(song_id).isdigit():
                raise ValueError('HKU song_id must be a numeric Jamendo candidate ID')
            track = lookup(song_id, client_id)
            row = metadata_row(song_id, track)
            sidecar = out / f'{song_id}.json'
            audio = out / f'{song_id}.mp3'
            if track and download:
                if not download_allowed(track):
                    row['status'] = 'download_not_allowed'
                else:
                    import soundfile as sf
                    try:
                        if not audio.exists():
                            url = track['audiodownload']
                            if urllib.parse.urlparse(url).scheme != 'https':
                                raise JamendoError('Download URL is not HTTPS; no implicit transport change')
                            part = audio.with_suffix('.part')
                            with urllib.request.urlopen(url, timeout=60) as response, part.open('wb') as f:
                                while chunk := response.read(1024 * 1024):
                                    f.write(chunk)
                            info = sf.info(part)
                            if info.duration <= 0:
                                raise JamendoError('Downloaded data is not usable audio')
                            part.rename(audio)
                        info = sf.info(audio)
                        row.update(status='downloaded_candidate', local_path=str(audio), sha256=digest(audio),
                                   decoded_duration_s=info.duration, sample_rate=info.samplerate,
                                   channels=info.channels)
                    except (OSError, RuntimeError, ValueError):
                        row['status'] = 'audio_download_or_decode_failed'
            rows.append(row)
            if track:
                write_json(sidecar, row)
            # Preserve prior runs and write progress after every song.
            current_ids = {r['song_id'] for r in rows}
            pd.DataFrame([r for r in previous if r['song_id'] not in current_ids] + rows).to_csv(log, index=False)
            print(song_id, row['status'], flush=True)
            time.sleep(.3)
    finally:
        write_json(reports / 'jamendo_status.json', {
            'requested_songs': len(ids), 'processed_songs': len(rows), 'download_requested': download,
            'downloaded_candidates': sum(r['status'] == 'downloaded_candidate' for r in rows),
            'candidate_directory': str(out), 'requires_recording_and_alignment_verification': True})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default=str(ROOT / 'configs/default.yaml'))
    p.add_argument('--limit', type=int, default=3)
    p.add_argument('--all', action='store_true', help='Query all unique HKU song IDs')
    p.add_argument('--download', action='store_true', help='Download only when explicitly allowed by track metadata')
    args = p.parse_args()
    try:
        recover(load_config(args.config), os.environ.get('JAMENDO_CLIENT_ID', '').strip(), None if args.all else args.limit, args.download)
    except (JamendoError, ValueError) as exc:
        p.exit(2, str(exc) + '\n')

if __name__ == '__main__':
    main()
