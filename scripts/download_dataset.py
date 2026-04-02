from __future__ import annotations

import argparse
from pathlib import Path
import sys
import urllib.error
import urllib.request
import zipfile


DATASET_URLS = {
    "apple2orange": "https://efrosgans.eecs.berkeley.edu/cyclegan/datasets/apple2orange.zip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a CycleGAN dataset.")
    parser.add_argument("--dataset-name", default="apple2orange", choices=sorted(DATASET_URLS))
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    dataset_dir = data_root / args.dataset_name
    archive_path = data_root / f"{args.dataset_name}.zip"

    if dataset_dir.exists() and not args.force:
        print(f"Dataset already exists at {dataset_dir}")
        return

    dataset_url = DATASET_URLS[args.dataset_name]
    try:
        urllib.request.urlretrieve(dataset_url, archive_path)
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"Failed to download {args.dataset_name} from {dataset_url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Failed to download {args.dataset_name} from {dataset_url}: {exc.reason}") from exc

    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(data_root)
    archive_path.unlink(missing_ok=True)
    print(f"Dataset ready at {dataset_dir}")


if __name__ == "__main__":
    sys.exit(main())
