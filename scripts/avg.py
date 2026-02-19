# read_psnr.py
import sys
import numpy as np

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <file.txt>")
    sys.exit(1)

with open(sys.argv[1]) as f:
    values = [float(line.strip().split('    ')[-1].strip(',').split()[-1]) for line in f]

print(f"Average: {np.mean(values):.4f}, {len(values)}")
