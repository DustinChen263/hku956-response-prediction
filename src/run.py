"""Execute stages in order; missing audio stops training with an actionable status."""
from .common import parser, load_config, folder
from .audit import audit
from .build_labels import build_labels
from .make_splits import make_splits
from .extract_audio import extract_audio
from .train import train
from .evaluate import evaluate
from .report import report


def main():
    p = parser(__doc__)
    p.add_argument('--cv', action='store_true')
    args = p.parse_args()
    cfg = load_config(args.config)
    audit(cfg)
    build_labels(cfg)
    make_splits(cfg)
    try:
        extract_audio(cfg)
    except RuntimeError as exc:
        report(cfg)
        (folder(cfg,'reports') / 'pipeline_status.md').write_text(f'# Pipeline blocked before training\n\n{exc}\n',encoding='utf8')
        raise SystemExit(str(exc)) from exc
    train(cfg)
    evaluate(cfg)
    if args.cv:
        train(cfg,'cv')
        evaluate(cfg,suite='cv')
    report(cfg)

if __name__ == '__main__':
    main()
