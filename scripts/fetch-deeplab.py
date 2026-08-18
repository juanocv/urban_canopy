#!/usr/bin/env python
"""
Fetch just the model definitions the DeepLab backend needs.

VainF's DeepLabV3Plus-Pytorch is research code with no ``setup.py``, so it
cannot be pip-installed, and cloning it pulls ~11 MB -- mostly git history and
demo images -- when the pipeline only ever imports ``network.modeling``. That
package is self-contained: it depends on torch and numpy and on nothing else in
the repository.

This script downloads one pinned commit as a tarball and extracts only
``network/`` plus the upstream ``LICENSE``. No git required, ~65 KB on disk, and
the commit is fixed so two machines get identical code.

    python scripts/fetch-deeplab.py                     # -> ./DeepLabV3Plus-network
    python scripts/fetch-deeplab.py --dest vendor/dl    # somewhere else
    tree-ai --seg deeplab --deeplab-repo ./DeepLabV3Plus-network --ckpt <weights.pth>

Upstream is MIT licensed (Copyright (c) 2020 Gongfan Fang); the LICENSE file is
copied alongside the code so the terms travel with it.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = "VainF/DeepLabV3Plus-Pytorch"
#: Pinned so the fetched code is reproducible. Update deliberately.
COMMIT = "4e1087de98bc49d55b9239ae92810ef7368660db"
TARBALL = f"https://codeload.github.com/{REPO}/tar.gz/{COMMIT}"

#: Everything the pipeline needs, plus the licence it arrives under.
WANTED_DIRS = ("network/",)
WANTED_FILES = ("LICENSE",)


def _wanted(relative: str) -> bool:
    return relative.startswith(WANTED_DIRS) or relative in WANTED_FILES


def fetch(dest: Path, *, timeout: int = 120) -> list[Path]:
    """Download the pinned tarball and extract the needed members into *dest*."""
    print(f"Downloading {REPO} @ {COMMIT[:12]} ...")
    with urllib.request.urlopen(TARBALL, timeout=timeout) as response:
        payload = response.read()
    print(f"  {len(payload) / 1e6:.2f} MB downloaded")

    written: list[Path] = []
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            # Strip the "<repo>-<commit>/" prefix GitHub wraps the archive in.
            relative = member.name.split("/", 1)[1] if "/" in member.name else member.name
            if not _wanted(relative):
                continue

            target = (dest / relative).resolve()
            # Refuse anything that would escape the destination: a tar member is
            # attacker-controlled input, and Python does not check this for you.
            if not str(target).startswith(str(dest.resolve())):
                raise RuntimeError(f"Refusing path traversal in archive: {member.name}")

            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:  # pragma: no cover - directories filtered above
                continue
            with source, open(target, "wb") as handle:
                shutil.copyfileobj(source, handle)
            written.append(target)

    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("DeepLabV3Plus-network"),
        help="Where to write network/ and LICENSE (default: ./DeepLabV3Plus-network)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing destination",
    )
    args = parser.parse_args(argv)

    dest = args.dest
    if dest.exists() and any(dest.iterdir()) and not args.force:
        print(
            f"{dest} already exists and is not empty; pass --force to overwrite.", file=sys.stderr
        )
        return 1
    dest.mkdir(parents=True, exist_ok=True)

    written = fetch(dest)
    if not any(p.name == "modeling.py" for p in written):
        print(
            "network/modeling.py was not in the archive; refusing a partial fetch.", file=sys.stderr
        )
        return 1

    total = sum(p.stat().st_size for p in written)
    print(f"  {len(written)} files, {total / 1024:.0f} KB written to {dest}")
    print("\nUse it with:")
    print(f"  tree-ai --seg deeplab --deeplab-repo {dest} --ckpt <cityscapes-weights.pth>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
