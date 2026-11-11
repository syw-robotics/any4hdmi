from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

from any4hdmi.core.repair import RepairSettings
from any4hdmi.core.repair.dataset import RepairSession, select_motions
from any4hdmi.dataset.loading import resolve_input_paths


def main():
    parser = argparse.ArgumentParser(description='Repair ground penetration and self collisions with temporal pose preservation.')
    parser.add_argument('--motion', required=True, help='Dataset, motion subdirectory or .npz; accepts hf:// references.')
    parser.add_argument('--output-root', type=Path, help='New dataset directory outside the input dataset.')
    parser.add_argument('--dry-run', action='store_true', help='Check only; print findings without writing a dataset.')
    parser.add_argument('--passed-only', action='store_true', help='Export only passed/unchanged clips; otherwise retain original unresolved clips.')
    parser.add_argument('--ground', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--self-collision', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--ground-z', type=float, default=0.)
    args = parser.parse_args()
    if not args.dry_run and args.output_root is None:
        parser.error('--output-root is required unless --dry-run is used')
    source = resolve_input_paths(Path.cwd(), args.motion)[0] if args.motion.startswith("hf://") else Path(args.motion).expanduser().absolute()
    session = RepairSession(select_motions(source), RepairSettings(ground=args.ground, self_collision=args.self_collision, ground_z=args.ground_z))
    try:
        with tqdm(total=len(session.paths), desc='Checking' if args.dry_run else 'Repairing', unit='motion', miniters=1) as progress:
            session.start(check_only=args.dry_run)
            previous_status = None
            while True:
                state = session.snapshot()
                current = state['current']
                status = str(current.relative_to(session.motions_root)) if current else 'loading model' if state['running'] else 'done'
                progress.set_postfix_str(status, refresh=False)
                progress.update(len(state['results']) - progress.n)
                if sys.stderr.isatty() or status != previous_status or not state['running']:
                    progress.refresh()
                previous_status = status
                if not state['running']:
                    break
                time.sleep(.5)
            session.wait()
        state = session.snapshot()
        counts = Counter(item['outcome'] for item in state['results'].values())
        print(f"{'Checked' if args.dry_run else 'Repaired'} {len(state['results'])}/{state['total']} motions: {dict(counts)}")
        if args.dry_run:
            print(json.dumps({str(path.relative_to(session.motions_root)): report for path, report in state['results'].items()}, indent=2))
        else:
            print(f'Exporting to {args.output_root}...', flush=True)
            print(f'Report: {session.export(session.paths, args.output_root, passed_only=args.passed_only)}')
        raise SystemExit(1 if counts['needs_review'] or counts['failed'] else 0)
    except KeyboardInterrupt:
        session.cancel()
        raise SystemExit(130)
    finally:
        session.close()


if __name__ == '__main__':
    main()
