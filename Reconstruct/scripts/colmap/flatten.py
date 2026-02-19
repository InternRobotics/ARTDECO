import shutil
from pathlib import Path
from read_write_model import read_model, write_model, Image
def flatten_image_directory(input_dir: Path, output_dir: Path) -> None:
    """
    Flattens the directory structure of images by moving all images to the root of the specified directory.
    
    Args:
        directory (Path): The path to the directory containing subdirectories with images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in input_dir.iterdir():
        if subdir.is_dir():
            for image_fp in subdir.glob('*'):
                if image_fp.is_file():
                    relative_path_parts = image_fp.parent.relative_to(input_dir).parts
                    new_image_fp = output_dir / f"{'_'.join(relative_path_parts)}_{image_fp.name}"
                    print(f"Copying {image_fp} to {new_image_fp}")
                    shutil.copyfile(image_fp, new_image_fp)

def flatten_model(images):
    return {k: v._replace(name=v.name.replace('/', '_')) for k, v in images.items()}
    

data_dir = Path('D:\\Users\\zhengzhewen\\datasets\\l1\\3f\\bedroom-20250611-para-new')
new_model = data_dir.parent / (data_dir.name + '-flattened')
output_dir = new_model / 'sparse/0'
output_dir.mkdir(parents=True, exist_ok=True)
cond = lambda x: x.name.startswith('0/')

flatten_image_directory(data_dir / 'images', new_model / 'images')

model_data = read_model(data_dir / 'sparse/0')
if model_data is None:
    raise ValueError(f"Failed to read model from {input_dir / 'sparse'}")
cameras, images, points = model_data
images = flatten_model(images)
write_model(cameras, images, points, output_dir)




