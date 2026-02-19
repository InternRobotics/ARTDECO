import shutil
from pathlib import Path
from read_write_model import read_model, write_model, Image

def filter_model(cameras, images, cond):
    new_camera_keys = set()
    new_images = {}
    for i, (k, v) in enumerate(images.items(), start=1):
        if cond(v):
            new_images[i] = v
            new_camera_keys.add(v.camera_id)
    new_cameras = {k: cameras[k] for k in new_camera_keys}
    return new_cameras, new_images

data_dir = Path('D:\\Users\\zhengzhewen\\datasets\\l1\\3f\\bedroom-20250611-para')
input_dir = data_dir / 'rectified'
new_model = data_dir.parent / (data_dir.name + '-new')
output_dir = new_model / 'sparse/0'
output_dir.mkdir(parents=True, exist_ok=True)

cond = lambda x: x.name.startswith('0/')
model_data = read_model(input_dir / 'sparse')
if model_data is None:
    raise ValueError(f"Failed to read model from {input_dir / 'sparse'}")
cameras, images, points = model_data
cameras, images = filter_model(cameras, images, cond)
write_model(cameras, images, points, output_dir)
for k, v in images.items():
    new_image_fp = new_model / 'images' / v.name
    new_image_fp.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying {input_dir / 'images' / v.name} to {new_image_fp}")
    shutil.copyfile(input_dir / 'images' / v.name, new_image_fp)