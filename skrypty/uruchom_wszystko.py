"""Runs, one after another, all scripts that reproduce the numbers and figures of the chapter."""
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SCRIPTS = ["stale", "f1_nagroda", "f1_imp", "f1_miary", "f1_rozklady", "f1_heurystyka", "f1_siec_zalu", "f1_ctde",
           "f1_koncowe", "partia", "f0_proby", "f0_licytacja_sztuczna", "f0_cechy_koloru", "f0_dziadek", "f0_audyt",
           "f0_role", "f0_stare_nowe_cechy", "f0_heurystyka_sygnaly", "f2_porownanie", "f2_eksperci", "f1_wyrocznia"]


def main():
    selected = sys.argv[1:] or SCRIPTS
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    for n in selected:
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, n + ".py")], cwd=ROOT, env=env)
        print("%-24s %6.1f min  %s" % (n, (time.time() - t0) / 60, "ok" if r.returncode == 0 else "failed"))


if __name__ == "__main__":
    main()
