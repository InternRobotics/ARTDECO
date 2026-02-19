# Minimal script to organize images by prefix
import os
import shutil

input_dir = 'D:\\Users\\zhengzhewen\\datasets\\l1\\3f\\bedroom-20250611-para-flattened\\images'
output_dir = 'D:\\Users\\zhengzhewen\\datasets\\l1\\3f\\bedroom-20250611-para\\images'

for fname in os.listdir(input_dir):
    if not (fname.lower().endswith('.jpg') or fname.lower().endswith('.png')):
        continue
    parts = fname.split('_', 1)
    if len(parts) < 2:
        continue  # skip files without underscore
    subdir = parts[0]
    new_fname = parts[1]
    out_subdir = os.path.join(output_dir, subdir)
    os.makedirs(out_subdir, exist_ok=True)
    shutil.copy2(os.path.join(input_dir, fname), os.path.join(out_subdir, new_fname))
