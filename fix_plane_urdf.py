"""Patch Genesis 1.4.2 bundled plane URDFs: XML version "0.0" -> "1.0".
Upstream fix: Genesis-Embodied-AI/genesis-world PR #3393.
Must be re-run after any reinstall of genesis-world."""
from importlib.util import find_spec
from pathlib import Path

plane_dir = Path(find_spec("genesis").origin).parent / "assets" / "urdf" / "plane"
for p in sorted(plane_dir.glob("*.urdf")):
    text = p.read_text(encoding="utf-8")
    first = text.splitlines()[0]
    print(f"{p.name}: {first}")
    if 'version="0.0"' in first:
        p.write_text(text.replace('version="0.0"', 'version="1.0"', 1), encoding="utf-8")
        print("  -> patched to 1.0")