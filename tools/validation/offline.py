"""Saved-endpoint comparison using the same policy as the live runner."""
import hashlib
import json
from pathlib import Path

from .common import load_endpoint
from .policy import prepare_policy
from .precision import mass_model


def receipt_for(endpoint):
    path = Path(endpoint)
    return path.with_name(path.stem.removesuffix('-performance') + '.json')


def compare_saved(args):
    cpu = load_endpoint(args.cpu, args.cpu_prefix)
    previous = load_endpoint(args.baseline, args.baseline_prefix)
    samples = [load_endpoint(path, args.old_sample_prefix) for path in args.old_sample]
    paths = [receipt_for(path) for path in (args.candidate, args.baseline, args.cpu, *args.old_sample)]
    receipts = [json.loads(path.read_text()) if path.is_file() else None for path in paths]
    verified = all(r is not None and r.get('input', {}).get('signature') for r in receipts)
    if args.precision == 'fp32' and not verified:
        raise ValueError('FP32 offline comparison requires matching adjacent JSON input receipts')
    if verified:
        signatures = [r.get('input', {}).get('signature') for r in receipts]
        if not signatures[0] or len(set(signatures)) != 1:
            raise ValueError('saved endpoint input signatures differ')
        if any(r.get('worlds', r.get('input', {}).get('worlds')) != args.worlds for r in receipts):
            raise ValueError('saved endpoint world counts differ')
        if receipts[0].get('precision', 'fp64') != args.precision:
            raise ValueError('candidate receipt precision differs from comparison precision')
        if receipts[1].get('status') != 'recorded_gpu' or receipts[2].get('status') != 'recorded_cpu':
            raise ValueError('comparison requires explicitly recorded frozen GPU/CPU references')
        if any(r.get('status') != 'recorded_gpu' for r in receipts[3:]):
            raise ValueError('additional samples must be explicitly recorded frozen GPU references')
    model = None
    if args.precision == 'fp32':
        if args.model is None:
            raise ValueError('FP32 offline comparison requires --model PATH/model.mjb')
        if args.model.name != 'model.mjb':
            raise ValueError('--model must identify the task model.mjb')
        digest = hashlib.sha256(args.model.read_bytes()).hexdigest()
        if any(r['input'].get('model_sha256') != digest for r in receipts):
            raise ValueError('offline model does not match the saved input identity')
        model = mass_model(dict(directory=str(args.model.parent)))
    policy = prepare_policy(args.precision, previous, args.worlds, model=model)
    actual = load_endpoint(args.candidate, args.candidate_prefix)
    return dict(**policy.evaluate(actual, previous, cpu, baseline_samples=samples),
        comparison_policy=policy.metadata, precision=args.precision,
        candidate=str(args.candidate), baseline=str(args.baseline), cpu=str(args.cpu),
        extra_old_samples=[str(path) for path in args.old_sample],
        input_receipts_verified=verified, physics_executed=False,
        scope='endpoint numerical screen; integer work is reported by trajectory receipts')
