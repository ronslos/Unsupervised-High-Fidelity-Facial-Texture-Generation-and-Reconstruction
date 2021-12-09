import glob
# from PIL import Image
import shutil
import os.path as osp

OUT_DIR = '/Users/ronslos/Downloads/recon_figures'

recon_files = glob.glob('/Users/ronslos/Downloads/reconstruction5/*')
for f in recon_files:
    crop_path = osp.join(f, 'target_crop.png')
    rendered_path = osp.join(f, 'rendered_rot.png')
    dir = f.split('/')[-1]

    crop_out_path = osp.join(OUT_DIR, f'{dir}_crop.png')
    rendered_out_path = osp.join(OUT_DIR, f'{dir}_rendered.png')

    print(f'copying {rendered_path} to {rendered_out_path}')
    shutil.copy2(crop_path, crop_out_path)
    shutil.copy2(rendered_path, rendered_out_path)

    # target = np.array(Image.open(crop_path)) / 255
    # rendered = np.array(Image.open(rendered_path)) / 255

gen_files1 = glob.glob('/Users/ronslos/Downloads/out_3d3/*')
gen_files2 = glob.glob('/Users/ronslos/Downloads/out_3d4/*')

OUT_DIR = '/Users/ronslos/Downloads/gen_figures'
for f1, f2 in zip(gen_files1, gen_files2):
    tex_path = osp.join(f1, 'model.png')
    rendered_path1 = osp.join(f1, 'rendered_lighting.png')
    rendered_path2 = osp.join(f2, 'rendered_lighting.png')
    dir1 = f1.split('/')[-1]
    dir2 = f2.split('/')[-1]

    tex_out_path = osp.join(OUT_DIR, f'{dir1}_model.png')
    rendered_out_path1 = osp.join(OUT_DIR, f'{dir1}_rendered_lighting1.png')
    rendered_out_path2 = osp.join(OUT_DIR, f'{dir2}_rendered_lighting2.png')

    print(f'copying {rendered_path1} to {rendered_out_path1}')
    shutil.copy2(tex_path, tex_out_path)
    shutil.copy2(rendered_path1, rendered_out_path1)
    shutil.copy2(rendered_path2, rendered_out_path2)
