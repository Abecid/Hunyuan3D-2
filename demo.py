import os
import time
from tqdm import tqdm

import torch
from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

# shape
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    'tencent/Hunyuan3D-2mv',
    subfolder='hunyuan3d-dit-v2-mv-turbo',
    variant='fp16'
)
pipeline.enable_flashvdm()

# paint
paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
    'tencent/Hunyuan3D-2',
    subfolder='hunyuan3d-paint-v2-0-turbo'
)

dataset_path = "/home/fai/workspace/adam/3d_rendering/assets/Welstory"
output_folder = "output"

def generate_mesh(images, dir_name="demo"):
    print("Starting shape generation...")
    start_time = time.time()
    mesh = pipeline(
        image=images,
        num_inference_steps=5,
        octree_resolution=380,
        num_chunks=20000,
        generator=torch.manual_seed(12345),
        output_type='trimesh'
    )[0]
    print(f"Shape generation time: --- {time.time() - start_time} seconds ---")
    
    output_path = os.path.join(output_folder, dir_name)
    os.makedirs(output_path, exist_ok=True)
    shape_output_path = os.path.join(output_path, f'shape.glb')
    mesh.export(shape_output_path)

    output_mesh_path = os.path.join(output_path, f'textured.glb')
    
    print("Starting texture generation...")
    start_time = time.time()
    mesh = paint_pipeline(mesh, image=[images[k] for k in images])
    print(f"Texture generation time: --- {time.time() - start_time} seconds ---")
    mesh.export(output_mesh_path)

def image_preprocess(image_path):
    image = Image.open(image_path).convert("RGBA")
    if image.mode == 'RGB':
        rembg = BackgroundRemover()
        image = rembg(image)
    return image

def main():
    # Iteratve through dataset directories
    for dir_name in tqdm(os.listdir(dataset_path)):
        # Check if it's a directory
        dir_path = os.path.join(dataset_path, dir_name)
        if os.path.isdir(dir_path):
            # image_path = [f for f in os.listdir(dir_path) if (f.endswith(".jpg") or f.endswith(".png") ) and f.split("_")[-2] == "TC"][0]
            images = [str(os.path.join(dir_path, f)) for f in os.listdir(dir_path) if (f.endswith(".jpg") or f.endswith(".png") ) and (f.split("_")[-2] == "TC" or f.split("_")[-2] == "LW")]
            tc_index = 0 if "TC" in images[0] else 1
            images = {
                "front": image_preprocess(images[tc_index]),
                "left": image_preprocess(images[1 - tc_index]),
            }
            generate_mesh(images, dir_name)

    

if __name__ == "__main__":
    main()
