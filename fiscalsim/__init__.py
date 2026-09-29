"""AI fiscal simulation of civil service pay, workforce and pensions (modules M1 to M6)."""
from pathlib import Path

# Newest modification time of the package's source files when it was imported. The dashboard
# compares it with the files on disk to notice code pulled into a server that is still running.
SOURCE_STAMP = max(p.stat().st_mtime for p in Path(__file__).parent.glob("*.py"))
