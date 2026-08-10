import argparse
import csv
import re
from pathlib import Path


TEST_EPOCH_RE = re.compile(r'Test Epoch\s+(\d+)')
BEST_RE = re.compile(r'Best Result:([0-9.]+),\s*epoch:([0-9]+)')
SECTION_RE = re.compile(
    r'^(?P<class>\w+)\s+(?P<protocol>AP_R40|AP)@'
    r'(?P<overlap_easy>[0-9.]+),\s*(?P<overlap_moderate>[0-9.]+),\s*(?P<overlap_hard>[0-9.]+):'
)
METRIC_RE = re.compile(r'^(?P<metric>bbox|bev|3d|aos)\s+AP:\s*(?P<easy>[0-9.]+),\s*(?P<moderate>[0-9.]+),\s*(?P<hard>[0-9.]+)')


def parse_log(path):
    rows = []
    current_epoch = None
    current_class = None
    current_protocol = None
    current_overlaps = None
    best_result = None
    best_epoch = None

    for raw_line in path.read_text(errors='ignore').splitlines():
        line = raw_line.strip()

        epoch_match = TEST_EPOCH_RE.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))

        best_match = BEST_RE.search(line)
        if best_match:
            best_result = float(best_match.group(1))
            best_epoch = int(best_match.group(2))

        section_match = SECTION_RE.match(line)
        if section_match:
            current_class = section_match.group('class')
            current_protocol = section_match.group('protocol')
            current_overlaps = (
                float(section_match.group('overlap_easy')),
                float(section_match.group('overlap_moderate')),
                float(section_match.group('overlap_hard')),
            )
            continue

        metric_match = METRIC_RE.match(line)
        if metric_match and current_class and current_protocol and current_overlaps:
            rows.append({
                'log_file': str(path),
                'epoch': current_epoch if current_epoch is not None else '',
                'class': current_class,
                'protocol': current_protocol,
                'overlap_easy': current_overlaps[0],
                'overlap_moderate': current_overlaps[1],
                'overlap_hard': current_overlaps[2],
                'metric': metric_match.group('metric'),
                'easy': float(metric_match.group('easy')),
                'moderate': float(metric_match.group('moderate')),
                'hard': float(metric_match.group('hard')),
                'best_result': best_result if best_result is not None else '',
                'best_epoch': best_epoch if best_epoch is not None else '',
            })

    return rows


def main():
    parser = argparse.ArgumentParser(description='Parse MonoDGP train logs into CSV metrics.')
    parser.add_argument('--log', nargs='+', required=True, help='One or more train.log files.')
    parser.add_argument('--out_csv', required=True, help='Output CSV path.')
    args = parser.parse_args()

    rows = []
    for item in args.log:
        for path in sorted(Path().glob(item) if any(ch in item for ch in '*?[]') else [Path(item)]):
            if path.exists():
                rows.extend(parse_log(path))

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'log_file',
        'epoch',
        'class',
        'protocol',
        'overlap_easy',
        'overlap_moderate',
        'overlap_hard',
        'metric',
        'easy',
        'moderate',
        'hard',
        'best_result',
        'best_epoch',
    ]
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Wrote {len(rows)} metric rows to {out_path}')


if __name__ == '__main__':
    main()
