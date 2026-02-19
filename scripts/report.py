#!/usr/bin/env python3
import argparse, json
from pathlib import Path

def load_json(p: Path):
    try:
        with p.open("r") as f:
            return json.load(f)
    except Exception:
        return None

def find_ci_values(obj, targets={"psnr","time"}):
    """Iterative, case-insensitive search for first hits of target keys."""
    want = {k.lower() for k in targets}
    found = {}
    stack = [obj]
    while stack and want - found.keys():
        x = stack.pop()
        if isinstance(x, dict):
            for k, v in x.items():
                kl = k.lower()
                if kl in want and kl not in found:
                    found[kl] = v
                if isinstance(v, (dict, list, tuple)):
                    stack.append(v)
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
    return found

def to_float(v):
    if isinstance(v, (int, float)): return float(v)
    if isinstance(v, str):
        try: return float(v)
        except: return None
    if isinstance(v, (list, tuple)):
        for e in reversed(v):  # prefer latest-looking entry
            fe = to_float(e)
            if fe is not None: return fe
    return None

def main():
    ap = argparse.ArgumentParser(description="Collect psnr/time from JSONs in dirs.")
    ap.add_argument("dirs", nargs="+", help="Shell-expanded directories")
    ap.add_argument("--method", type=str, default="onthefly")
    args = ap.parse_args()

    rows, psnrs, times = [], [], []
    for d in map(Path, args.dirs):
        d = d / args.method
        if d.exists():
            jp = (d / "metadata.json")
            if not jp.exists():
                jp = d / "stats" / "val_step29999.json"
            psnr = time = None
            if jp.exists():
                data = load_json(jp)
                if data is not None:
                    vals = find_ci_values(data, {"psnr", "time"})
                    if "psnr" in vals: psnr = to_float(vals["psnr"])
                    if "time" in vals: time = to_float(vals["time"])
            if psnr is not None: psnrs.append(psnr)
            if time is not None: times.append(time)
            rows.append((d, psnr, time))
        else:
            rows.append((d, None, None))

    print("dir,psnr,time")
    for n, p, t in rows:
        print(f"{n},{'' if p is None else p},{'' if t is None else t}")

    n = len(rows)
    np, nt = len(psnrs), len(times)
    avg_p = (sum(psnrs)/np) if np else float("nan")
    avg_t = (sum(times)/nt) if nt else float("nan")
    print(f"avg_psnr={avg_p} over {np}/{n} dirs")
    print(f"avg_time={avg_t} over {nt}/{n} dirs")

if __name__ == "__main__":
    main()
