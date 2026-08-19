"""Environment verification: import every required module in an isolated subprocess."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

MODS = [
    "numpy", "scipy", "pandas", "sklearn", "statsmodels", "matplotlib", "networkx",
    "torch", "torch_geometric", "toponetx", "topomodelx", "krippendorff", "rbo",
    "datasketch", "gudhi", "hypernetx", "deepsig", "nltk", "transformers",
    "sentence_transformers", "requests", "bs4", "yaml",
]

report = {}
for m in MODS:
    p = subprocess.run(
        [sys.executable, "-c", f"import {m}; print(getattr({m},'__version__','?'))"],
        capture_output=True, text=True, timeout=300,
    )
    report[m] = {
        "ok": p.returncode == 0,
        "version": p.stdout.strip().splitlines()[-1] if p.returncode == 0 and p.stdout.strip() else None,
        "error": (p.stderr.strip().splitlines()[-1][:160] if p.returncode != 0 else None),
    }

out = Path(__file__).resolve().parents[1] / "data" / "logs" / "env_report.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2), encoding="utf-8")

bad = [k for k, v in report.items() if not v["ok"]]
print(f"OK: {len(MODS) - len(bad)}/{len(MODS)}")
for k in bad:
    print(f"  FAIL {k}: {report[k]['error']}")
