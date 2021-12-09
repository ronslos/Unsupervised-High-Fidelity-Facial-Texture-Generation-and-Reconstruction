import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from texture_projector import project
from tqdm import tqdm
import os
import itertools
import imageio
import numpy as np
import PIL.Image
import torch
from torch_utils.misc import load_lm3d, POS, resize_n_crop_img
import torch.nn.functional as F
from torch_utils import misc
from training.deepfacemodel import ReconNetWrapper
from facenet_pytorch import MTCNN
import pytorch3d.io as py3dio
from scipy.io import loadmat, savemat
from training.models import ReconModelOrig

import dnnlib
import legacy

run_celeba = False

if run_celeba:
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Pad(padding=[20, 0], fill=0, padding_mode='constant')])
    dataset = datasets.CelebA('./datasets', 'test', transform=transform)
    OUT_DIR = 'celeba_test'
else:
    transform = transforms.Compose(
        [transforms.ToTensor()])
    dataset = datasets.ImageFolder('./examples_celebs', transform=transform)
    print(len(dataset))
    REG_WEIGHT = 10000
    OUT_DIR = 'reconstruction_'+str(REG_WEIGHT)


data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False)

FACE_MODEL_PATH = 'BFM/BFM_model_front_msft.mat'


print(len(dataset))
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# Load networks.
network_pkl = 'saved_models/flip_hide_mouth_narrow_msft-128.pkl'
print('Loading networks from "%s"...' % network_pkl)
device = torch.device('cuda')
with dnnlib.util.open_url(network_pkl) as fp:
    G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)  # type: ignore

facemodel = loadmat(FACE_MODEL_PATH)
face_model = ReconModelOrig(facemodel, img_size=[1024, 1024], focal=[1015 * 1024 / 224, 1015 * 1024 / 224],
                            device=device).to(device)

# Load target image.
# target_pil = PIL.Image.open(target_fname).convert('RGB')
# save_name = target_fname.split('.')[0].split('/')[-1]
# w, h = target_pil.size
# s = min(w, h)
# target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
# target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
# target_uint8 = np.array(target_pil, dtype=np.uint8)
start = 0
data_loader = itertools.islice(data_loader, start, start+10)
num_steps = 1000
for i, (images, labels) in enumerate(tqdm(data_loader)):
    target = images.to(device).squeeze() * 255

    # Optimize projection.
    projected_w_steps, composed, tex, composed_crop, mask_full, mask, target_crop, rendered_img_rot, scores = project(
        G,
        regularize_noise_weight=REG_WEIGHT,
        target=target.type(torch.uint8),  # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        verbose=False
    )
    if composed is None:
        continue

    if not os.path.isdir(f'{OUT_DIR}/{start + i}'):
        os.mkdir(f'./{OUT_DIR}/{start + i}')
    composed = composed.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    PIL.Image.fromarray(composed, 'RGB').save(f'{OUT_DIR}/{start + i}/proj_composed.png')

    composed_crop = composed_crop.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    PIL.Image.fromarray(composed_crop, 'RGB').save(f'{OUT_DIR}/{start + i}/proj_composed_crop.png')

    tex_save = tex.squeeze().clamp(0, 255).to(torch.uint8).cpu().numpy()
    PIL.Image.fromarray(tex_save, 'RGB').save(f'{OUT_DIR}/{start + i}/proj_tex.png')

    target = target.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    PIL.Image.fromarray(target, 'RGB').save(f'{OUT_DIR}/{start + i}/target.png')

    target_crop = target_crop.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    PIL.Image.fromarray(target_crop, 'RGB').save(f'{OUT_DIR}/{start + i}/target_crop.png')

    mask_full = mask_full.clamp(0, 255).to(torch.uint8).cpu().numpy()
    PIL.Image.fromarray(mask_full, 'L').save(f'{OUT_DIR}/{start + i}/mask.png')

    mask = mask.clamp(0, 255).to(torch.uint8).cpu().numpy()
    PIL.Image.fromarray(mask, 'L').save(f'{OUT_DIR}/{start + i}/mask_crop.png')

    rendered_img_rot = rendered_img_rot.permute(1, 2, 0).clamp(0, 255).to(torch.uint8).cpu().numpy()
    PIL.Image.fromarray(rendered_img_rot, 'RGB').save(f'{OUT_DIR}/{start + i}/rendered_rot.png')

    savemat(f'{OUT_DIR}/{start + i}/scores.mat', scores)

    params = torch.from_numpy(scores['params']).to(device)
    rendered_img, pred_lms, face_texture, mask, vert, tri, uv, rendered_img_white = face_model.inference(params,
                                                                                                         texture_map=tex)
    py3dio.save_obj(f'{OUT_DIR}/{start + i}/proj_model.obj', verts=vert.squeeze(), faces=tri, verts_uvs=uv,
                    faces_uvs=tri,
                    texture_map=(tex.squeeze() / 255).clamp(0, 1))
