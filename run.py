import os
import time
from tqdm import tqdm

import trimesh
import torch

from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.rembg import BackgroundRemover

image_path = "/home/fai/workspace/adam/Hunyuan3D-Omni/assets/canon_views/332/910_1_5_TC_L.jpg"

dataset_path = "/home/fai/workspace/jhkim/vco_train_vol/dataset/images_od/train"

def image_gen(image_path=image_path, name="mesh", pipeline=None, paint_pipeline=None):
    start_time = time.time()
    mesh = pipeline(image=image_path)[0]
    print(f"Shape generation took {time.time() - start_time:.2f} seconds")
    start_time = time.time()
    mesh = paint_pipeline(mesh, image=image_path)
    end_time = time.time()
    print(f"Painting took {end_time - start_time:.2f} seconds")

    mesh_path = f"output/{name}.glb"
    os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
    mesh.export(mesh_path, file_type='glb')

def texture():
    paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')

    mesh_path = "/home/fai/workspace/adam/Hunyuan3D-Omni/omni_inference_results/3domni_point/0_0_TC_L.glb"
    image_path = "/home/fai/workspace/adam/Hunyuan3D-Omni/assets/welstory1/9/0_0_TC_L.png"
    mesh = trimesh.load(mesh_path, process=False)
    start_time = time.time()
    mesh = paint_pipeline(mesh, image=image_path)
    end_time = time.time()
    print(f"Painting took {end_time - start_time:.2f} seconds")

    mesh_path = "output/mesh.glb"
    os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
    mesh.export(mesh_path, file_type='glb')

def images_gen(images=None, remove_bg=False, name="mesh", pipeline=None, paint_pipeline=None):
    from PIL import Image

    if images is None:
        images = {
            "front": "assets/example_mv_images/1/front.png",
            "left": "assets/example_mv_images/1/left.png",
            "back": "assets/example_mv_images/1/back.png"
        }

    for key in images:
        image = Image.open(images[key]).convert("RGBA")
        if remove_bg and image.mode == 'RGB':
            rembg = BackgroundRemover()
            image = rembg(image)
        images[key] = image

    start_time = time.time()
    mesh = pipeline(
        image=images,
        num_inference_steps=50,
        octree_resolution=380,
        num_chunks=20000,
        generator=torch.manual_seed(12345),
        output_type='trimesh'
    )[0]
    print(f"Shape generation took {time.time() - start_time:.2f} seconds")
    start_time = time.time()

    mesh = paint_pipeline(mesh, image=images["front"])
    print(f"Painting took {time.time() - start_time:.2f} seconds")

    mesh_path = f"output/{name}.glb"
    os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
    mesh.export(mesh_path)

def dataset_gen(num_samples=10, type="sv"):
    assets = os.listdir(dataset_path)[:num_samples]

    if type == "sv":
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2')
        paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
    elif type == "mv":
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            'tencent/Hunyuan3D-2mv',
            subfolder='hunyuan3d-dit-v2-mv',
            variant='fp16'
        )
        paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained('tencent/Hunyuan3D-2')
    
    for asset_dir in tqdm(assets):
        if not os.path.isdir(os.path.join(dataset_path, asset_dir)):
            continue
        images = {}
        for img_file in os.listdir(os.path.join(dataset_path, asset_dir)):
            if "1_5_TC_L" in img_file:
                images["front"] = os.path.join(dataset_path, asset_dir, img_file)
            elif "1_5_RW_L" in img_file:
                images["left"] = os.path.join(dataset_path, asset_dir, img_file)
            elif "0_5_TC_L" in img_file:
                images["back"] = os.path.join(dataset_path, asset_dir, img_file)
        
        print(f"Processing asset: {asset_dir}")
        if type == "sv" and "front" in images:
            image_gen(images["front"], name=f"{asset_dir}/sv", pipeline=pipeline, paint_pipeline=paint_pipeline)
        elif type == "mv":
            images_gen(images, name=f"{asset_dir}/mv", pipeline=pipeline, paint_pipeline=paint_pipeline)

    
    # relesase GPU memory
    pipeline.to("cpu")
    paint_pipeline.to("cpu")
    del pipeline
    del paint_pipeline
    torch.cuda.empty_cache()


if __name__ == '__main__':
    dataset_gen()
    dataset_gen(type="mv")
