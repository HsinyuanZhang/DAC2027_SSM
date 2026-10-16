#!/usr/bin/env python
"""Write / check the pinned integrity manifest of the FALCON M2 held-in npz cache.

  python scripts/m2_npz_manifest.py build [--verify-nwb]   # (re)write the manifest from the cache
  python scripts/m2_npz_manifest.py check                  # load every session through load_m2_recording

``--verify-nwb`` re-hashes the NWB files (about 14 GB on the HDD) and refuses to write a manifest
if a stored sha256 does not match.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ssm_decode import m2_calib as m2                                     # noqa: E402
from ssm_decode.data import sha256_file                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("--verify-nwb", action="store_true")
    args = ap.parse_args()
    if args.cmd == "build":
        man = m2.build_m2_manifest()
        if args.verify_nwb:
            for s, e in man.items():
                ok = Path(e["nwb_path"]).exists() and sha256_file(Path(e["nwb_path"])) == e["nwb_sha256"]
                print(s, "nwb sha256 ok" if ok else "NWB MISMATCH", flush=True)
                if not ok:
                    raise SystemExit(1)
        m2.M2_MANIFEST.write_text(json.dumps(man, indent=1) + "\n")
        print("wrote", m2.M2_MANIFEST, len(man), "sessions")
    else:
        for f in sorted(m2.M2_CACHE.glob("*.npz")):
            r = m2.load_m2_recording(f.stem)
            print(f.stem, "ok", r.neural.shape, len(r.trial_bounds))


if __name__ == "__main__":
    main()
