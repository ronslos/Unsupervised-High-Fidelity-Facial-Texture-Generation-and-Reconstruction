# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from facenet_pytorch import MTCNN
from torch_utils.misc import load_lm3d

import legacy
from training.models import ReconModelOrig, DFRModel, RigNet, ResNet, BasicBlock
from training.deepfacemodel import ReconNetWrapper
from scipy.io import loadmat
from plyfile import PlyData, PlyElement
from torch_utils.misc import crop_img, POS, resize_n_crop_img
import pytorch3d.io as py3dio
import os.path as osp

DFR_MODEL_PATH = 'BFM/LSFM_model_front.mat'
FACE_MODEL_PATH = 'BFM/BFM_model_front_msft.mat'
TAR_SIZE = 1024  # size for rendering window
make_figures = True


# ----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2)) + 1))
    vals = s.split(',')
    return [int(x) for x in vals]


# ----------------------------------------------------------------------------
def save_ply(verts, faces, uv, f_name):
    vert_uv = np.concatenate((verts, uv), axis=1)
    verts = np.array([tuple(vert_uv[i]) for i in range(vert_uv.shape[0])],
                     dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('s', 'f4'), ('t', 'f4')])
    tris = np.array([(list(faces[i, :]), 255, 255, 255) for i in range(faces.shape[0])],
                    dtype=[('vertex_indices', 'i4', (3,)), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el1 = PlyElement.describe(verts, 'vertex')
    el2 = PlyElement.describe(tris, 'face')
    PlyData([el1, el2]).write(f'{f_name}.ply')


def normalize_img(img):
    norm_img = (img * 127.5 + 128).clamp(0, 255)
    norm_img = norm_img.permute(0, 2, 3, 1)
    return norm_img.to(torch.uint8)


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const',
              show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--num_images', help='Number of images to generate', type=int, required=True)
def generate_images(
        ctx: click.Context,
        network_pkl: str,
        seeds: Optional[List[int]],
        truncation_psi: float,
        noise_mode: str,
            outdir: str,
        num_images: int,
        class_idx: Optional[int],
        projected_w: Optional[str]
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    network_pkl_2d = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl'
    with dnnlib.util.open_url(network_pkl_2d) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device).eval()  # type: ignore
    with dnnlib.util.open_url(network_pkl) as f:
        G_3d = legacy.load_network_pkl(f)['G_ema'].to(device).eval()  # type: ignore

    mtcnn = MTCNN(device=device, keep_all=False, selection_method='probability')
    lm3d_std = load_lm3d('BFM')
    net_recon_path = 'saved_models/epoch_20.pth'
    net_recon = ReconNetWrapper(
        net_recon='resnet50', use_last_fc=False, init_path=None).to(device)
    net_recon.load_state_dict(
        torch.load(net_recon_path, map_location=device)['net_recon'])
    net_recon = net_recon.eval()

    dfr_model = DFRModel().to(device)
    resume_path = 'saved_models/params_lsfm_465000.pth'
    print(f'loading model {resume_path}')
    dfr_model.load_state_dict(
        torch.load(resume_path, map_location=device))

    rignet_model = RigNet().to(device)
    resume_path = 'saved_models/rignet_no_trans2_84000.pth'
    print(f'loading model {resume_path}')
    rignet_model.load_state_dict(
        torch.load(resume_path, map_location=device))

    try:
        facemodel = loadmat(FACE_MODEL_PATH)
        dfr_facemodel = loadmat(DFR_MODEL_PATH)
    except Exception as e:
        print('failed to load %s' % FACE_MODEL_PATH)

    face_model = ReconModelOrig(facemodel, img_size=[TAR_SIZE, TAR_SIZE], device=device).to(device)
    dfr_face_model = ReconModelOrig(dfr_facemodel, img_size=[TAR_SIZE, TAR_SIZE], focal=[1015 * 4, 1015 * 4],
                                    device=device).to(device)

    os.makedirs(outdir, exist_ok=True)

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device)  # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
        return

    if seeds is None:
        ctx.fail('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    style_mixing_prob = 0.9
    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        np.random.seed(seed)

        for i in range(num_images):
            print('Generating image for seed %d (%d/%d) ...' % (seed, i, len(seeds)))
            z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
            ws_2d = G.mapping(z.to(device), label, truncation_psi=truncation_psi)
            ws_3d = G_3d.mapping(z.to(device), label, truncation_psi=truncation_psi)

            p = dfr_model(ws_2d)

            # d_tex = rignet_model(ws_3d, p)
            # ws_rot = ws_2d + d_rot

            p_edit_rot = p.clone()
            p_edit_rot[:, 225] = 2
            d_rot = rignet_model(ws_2d, p_edit_rot)
            ws_rot_2d = ws_2d #+ d_rot
            tex = G_3d.synthesis(ws_3d, noise_mode=noise_mode)
            tex = ((tex.clone() + 1) / 2).clamp(0, 1).permute(0, 2, 3, 1)
            rendered_img, pred_lms, face_texture, mask, vert, tri, uv, rendered_img_white = dfr_face_model.inference(p,
                                                                                                                     texture_map=tex * 255)

            rendered_img = (255 * rendered_img[:, :, :, :3] / rendered_img[:, :, :, :3].max()).to(torch.uint8).squeeze()
            rendered_img_white = (255 * rendered_img_white[:, :, :, :3] / rendered_img_white[:, :, :, :3].max()).to(
                torch.uint8).squeeze()

            if not osp.isdir(f'{outdir}/{i:06d}'):
                os.makedirs(f'{outdir}/{i:06d}')
            py3dio.save_obj(f'{outdir}/{i:06d}/model.obj', verts=vert.squeeze(), faces=tri, verts_uvs=uv, faces_uvs=tri,
                            texture_map=tex.squeeze())

            if make_figures:
                bg = normalize_img(G.synthesis(ws_2d, noise_mode=noise_mode))
                bg_rot = G.synthesis(ws_rot_2d, noise_mode=noise_mode)
                bg_img_normalized = ((bg_rot + 1) * (255 / 2)).clamp(0, 255)
                bg_rot = normalize_img(bg_rot)
                boxes, probs, points = mtcnn.detect(bg_img_normalized.permute(0, 2, 3, 1), landmarks=True)
                print(points.shape)
                if points[0] is None:
                    continue
                if points.shape[1] > 1:
                    points = points[:, 0, ...]
                print(points.shape)
                t, s = POS(points.squeeze().copy(), lm3d_std)
                cropped, left, up = resize_n_crop_img(bg_img_normalized.squeeze(), t, s, target_size=224.)
                box = torch.Tensor([left, up, s])[None, ...]

                params = net_recon(cropped.type(torch.uint8).type(torch.float32) / 255)
                params[:, 225] = -0.5
                rendered_img, _, face_texture, _, mask = face_model(params, texture_map=tex * 255, bboxs=box)
                rendered_img_lighting, _, face_texture, _, mask = face_model(params, texture_map=tex * 255, bboxs=box,
                                                                             use_lighting=True)
                box_new = torch.Tensor([TAR_SIZE / 2 - 112, TAR_SIZE / 2 - 112, 102])[None, ...]
                p_edit_rot[:, 225] = -0.5
                rendered_img_dfr, _, face_texture, _, mask = dfr_face_model(p_edit_rot, texture_map=tex * 255,
                                                                            bboxs=box_new,
                                                                            use_lighting=True)

                rendered_img_white, _, _, _, _ = face_model(params, texture_map=torch.ones_like(tex) * 200, bboxs=box)
                rendered_img = (255 * rendered_img[:3, :, :] / rendered_img[:3, :, :].max()).to(
                    torch.uint8).squeeze()
                rendered_img_lighting = (
                        255 * rendered_img_lighting[:3, :, :] / rendered_img_lighting[:3, :, :].max()).to(
                    torch.uint8).squeeze()

                rendered_img_dfr = (
                        255 * rendered_img_dfr[:3, :, :] / rendered_img_dfr[:3, :, :].max()).to(
                    torch.uint8).squeeze()

                rendered_img_white = (255 * rendered_img_white[:3, :, :] / rendered_img_white[:3, :, :].max()).to(
                    torch.uint8).squeeze()

                rendered_img = rendered_img.permute((1, 2, 0))
                rendered_img_lighting = rendered_img_lighting.permute((1, 2, 0))
                rendered_img_dfr = rendered_img_dfr.permute((1, 2, 0))
                rendered_img_white = rendered_img_white.permute((1, 2, 0))
                composed = bg_rot[0] * (~mask[:, :, None]) + mask[:, :, None] * rendered_img_lighting
                PIL.Image.fromarray(rendered_img.cpu().detach().numpy(), 'RGB').save(f'{outdir}/{i:06d}/rendered.png')
                PIL.Image.fromarray(rendered_img_lighting.cpu().detach().numpy(), 'RGB').save(
                    f'{outdir}/{i:06d}/rendered_lighting.png')
                PIL.Image.fromarray(rendered_img_dfr.cpu().detach().numpy(), 'RGB').save(
                    f'{outdir}/{i:06d}/rendered_dfr.png')
                PIL.Image.fromarray(rendered_img_white.cpu().detach().numpy(), 'RGB').save(
                    f'{outdir}/{i:06d}/rendered_white.png')
                PIL.Image.fromarray(bg[0].cpu().detach().numpy(), 'RGB').save(f'{outdir}/{i:06d}/bg.png')
                PIL.Image.fromarray(bg_rot[0].cpu().detach().numpy(), 'RGB').save(f'{outdir}/{i:06d}/bg_rot_.png')
                PIL.Image.fromarray(composed.cpu().detach().numpy(), 'RGB').save(f'{outdir}/{i:06d}/composed.png')

            # save_ply(vert, tri, uv, f'{outdir}/{i:06d}')
            # np.savez_compressed(f'{outdir}/{i:06d}.npz', ws=ws.cpu().detach().numpy())


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
