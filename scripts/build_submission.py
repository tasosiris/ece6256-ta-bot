#!/usr/bin/env python3
"""Create an allowlisted code archive with ta_bot.py at its root."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]


def main():
    names = ("README.md", "ta_bot.py", "eval_dev.py", "mock_endpoint.py", "endpoint_config.py", "requirements.txt",
             "requirements-dev.txt", "requirements-lock.txt", "corpus", "dev_eval", "tests",
             "experiments", "scripts", "results", "report")
    files = []
    for name in names:
        path = ROOT / name
        if not path.exists():
            raise SystemExit("Missing required artifact: " + name)
        files.extend([path] if path.is_file() else path.rglob("*"))
    files = sorted(path for path in files if path.is_file() and
                   not any(part.startswith(".") or part in ("__pycache__", "pdf")
                           for part in path.relative_to(ROOT).parts) and
                   path.suffix not in (".pyc", ".sqlite", ".sqlite3"))
    destination = ROOT / "dist"
    destination.mkdir(exist_ok=True)
    output = destination / "ece6256-ta-bot.zip"
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            entry = ZipInfo(path.relative_to(ROOT).as_posix(), date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, path.read_bytes())
    print("Built {} ({:.1f} MiB; {} files)".format(output, output.stat().st_size / 1024**2, len(files)))


if __name__ == "__main__":
    main()
