"""Run folders <results_dir>/<YYYYmmdd-HHMMSS>_<name>/ with output.txt and config.yaml.
Never overwritten. publish() copies an artefact to <results_dir>/current/ (previous copy kept as
<name>.prev.<ext>)."""
import shutil
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def results_root(cfg):
    p = Path(cfg["results_dir"])
    return p if p.is_absolute() else ROOT / p


def current_path(cfg, fname):
    return results_root(cfg) / "current" / fname


class _Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        for st in self.s:
            st.write(x)
            st.flush()

    def flush(self):
        for st in self.s:
            st.flush()


class Run:
    def __init__(self, cfg, name):
        base = results_root(cfg) / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"
        self.dir, i = base, 1
        while self.dir.exists():
            self.dir = base.with_name(f"{base.name}_{i}")
            i += 1
        self.cfg = cfg

    def path(self, fname):
        return self.dir / fname

    def publish(self, fname):
        dst = current_path(self.cfg, fname)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.copy2(dst, dst.with_name(f"{dst.stem}.prev{dst.suffix}"))
        shutil.copy2(self.path(fname), dst)
        print(f"published {fname} -> {dst}")

    def __enter__(self):
        self.dir.mkdir(parents=True)
        (self.dir / "config.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False), encoding="utf-8")
        self._fh = open(self.dir / "output.txt", "w", encoding="utf-8")
        self._out = sys.stdout
        sys.stdout = _Tee(self._out, self._fh)
        print(f"run folder: {self.dir}")
        return self

    def __exit__(self, *exc):
        sys.stdout = self._out
        self._fh.close()
        return False
