# Running and reproducing the project

## Clean checkout

Install `requirements.txt` and run `python -m pytest -q`. This does not download data or train models. Two real-data integration tests skip without their local files. `requirements.lock` is the historical runtime snapshot, not a solver lock. `tabulate` is added for pandas Markdown reports. Optional BCMI probes use `requirements-bcmi.txt`. MERT requirements are separate and MERT was not used for reported results.

`python -m src.demo --generate-only` creates synthetic data. Running `python -m src.demo` also trains synthetic models. These are software demonstrations, never HKU956 research results.

## Local real-data layout

```
data/raw/HKU956/HKU956/
    README
    1. physiological_signals/<participant>/{HR,IBI,BVP,EDA,TEMP}/<order>_<song_id>.csv
    2. original_song_audio.csv
    3. AV_ratings.csv
    4. participant_personality.csv
data/raw/audio/jamendo_candidates/<song_id>.mp3
data/raw/author_supplement_20261028/
    raw_log.download              # author CSV
    timestamped_ibi.download      # author ZIP
```

These files are not supplied here. `python scripts/download_hku.py` retrieves the public distribution and checks metadata-provided MD5. Confirm the distribution version. The author supplements must be obtained separately. Never substitute unrelated audio for an unavailable song.

From the repository root, core preparation begins with:

```bash
python -m src.audit
python -m src.build_labels
python -m src.make_splits
```

Author alignment additionally requires the supplements and generated project files. Read `scripts/author_alignment.py` and its configuration before running. The whole-track preparation script also reads the generated alignment table. Exact historical audio recovery membership, locked hash manifests, split artifacts, cached features and original model files are not bundled. Consequently a fresh download is not guaranteed to recreate the published 771-row cohort or exact scores.

The whole-track driver stages are:

```bash
python scripts/song_level_baseline.py prepare
python scripts/song_level_baseline.py extract
# Training is opt-in and should use a new, documented experiment directory:
# python scripts/song_level_baseline.py baseline
```

The later MLP driver, final-holdout driver, and reporting scripts depend on artifacts from earlier stages. They enforce local signatures; moving historical pickle/config bundles may invalidate path-dependent provenance. Do not bypass a lock merely to reproduce a score. Preserve historical results and configure a new run for changed data, features, or models.

The default `src` pipeline requires explicitly verified audio segments. It is distinct from the completed whole-track experiment. Do not mark an assumed zero start as verified just to enable feature extraction.

The strict audit (`scripts/audit_hku956_quality.py`) requires the full raw signals, supplements, alignment table, local split, and audio-recovery records. Historical report/figure scripts require their referenced local reports and predictions. They are retained for traceability rather than promised as standalone scripts in an empty checkout.

## Scope of this publication

Core model and preprocessing implementations and research configurations are preserved. Publication-only differences are: two optional integration-test skips when real data are absent, an additional reporting dependency, and removal of a local home-directory string from a BCMI report template. Generated report names may carry historical dates.

Aggregate results live in `docs/results/`; no per-participant records or predictions are published. All reported metrics precede this repository preparation. The final held-out participants are already evaluated, so future model choices cannot use them as a fresh blind confirmation set.
