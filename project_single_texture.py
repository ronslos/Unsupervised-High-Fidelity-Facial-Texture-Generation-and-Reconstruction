# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from time import perf_counter

import click
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
from scipy.io import loadmat
from training.models import ReconModelOrig

import dnnlib
import legacy

FACE_MODEL_PATH = 'BFM/BFM_model_front_msft.mat'
TARGET_SIZE = 512


def project(
        G,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.1,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1000,
        verbose=False,
        device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)  # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None, truncation_psi=0.7)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)
    mtcnn = MTCNN(device=device, keep_all=False, selection_method='probability')
    lm3d_std = load_lm3d('BFM')
    net_recon_path = 'saved_models/epoch_20.pth'
    net_recon = ReconNetWrapper(
        net_recon='resnet50', use_last_fc=False, init_path=None).to(device)
    net_recon.load_state_dict(
        torch.load(net_recon_path, map_location=device)['net_recon'])
    net_recon = net_recon.eval()
    facemodel = loadmat(FACE_MODEL_PATH)
    face_model = ReconModelOrig(facemodel, img_size=[TARGET_SIZE, TARGET_SIZE],
                                focal=[1015 * TARGET_SIZE / 224, 1015 * TARGET_SIZE / 224], device=device).to(
        device)

    with torch.no_grad():
        boxes, probs, points = mtcnn.detect(target.permute(1, 2, 0), landmarks=True)
        t, s = POS(points[0].squeeze().copy(), lm3d_std)
        cropped_for_model, left, up = resize_n_crop_img(target.squeeze(), t, s, target_size=224.)
        cropped_target, left, up = resize_n_crop_img(target.squeeze(), t, s * 224. / TARGET_SIZE,
                                                     target_size=TARGET_SIZE)

        box = torch.Tensor([left, up, s])[None, ...]

        params = net_recon(cropped_for_model.type(torch.uint8).type(torch.float32) / 255)

    # Features for target image.
    # target_images = target.unsqueeze(0).to(device).to(torch.float32)
    # if cropped_target.shape[2] > 256:
    # cropped_target = F.interpolate(cropped_target, size=(TARGET_SIZE, TARGET_SIZE), mode='area')
    bg_img = cropped_target.clone()
    target_features = vgg16(cropped_target, resize_images=False, return_lpips=True)

    avg_w = torch.tensor(w_avg, dtype=torch.float32, device=device)  # pylint: disable=not-callable
    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)  # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_tex = (G.synthesis(ws, noise_mode='const') + 1) * (255 / 2)
        synth_tex = synth_tex.permute((0, 2, 3, 1))
        rendered_img, _, face_texture, _, mask = face_model.forward_recon(params, texture_map=synth_tex, bboxs=box)

        synth_images = bg_img * (~mask[None, ...]) + mask[None, ...] * rendered_img[:, 0:3, :, :]
        synth_output = synth_images.clone()
        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight  # + 0.001*(w_opt-avg_w).square().sum()

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1]), synth_output, params, synth_tex


# ----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--num-steps', help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed', help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video', help='Save an mp4 video of optimization progress', type=bool, default=False,
              show_default=True)
@click.option('--outdir', help='Where to save the output images', required=True, metavar='DIR')
def run_projection(
        network_pkl: str,
        target_fname: str,
        outdir: str,
        save_video: bool,
        seed: int,
        num_steps: int
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device)  # type: ignore
    facemodel = loadmat(FACE_MODEL_PATH)
    face_model = ReconModelOrig(facemodel, img_size=[1024, 1024], focal=[1015 * 1024 / 224, 1015 * 1024 / 224],
                                device=device).to(device)
    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps, composed, params, tex = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device),  # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print(f'Elapsed: {(perf_counter() - start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        print(f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps:
            synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
            synth_image = (synth_image + 1) * (255 / 2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(synth_image)
        video.close()

    # Save final projected frame and W vector .
    target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    # synth_tex = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    # tex_save = synth_tex.clone().permute(0, 2, 3, 1)
    # tex_save = ((tex_save + 1) / 2).clamp(0, 1)
    # synth_tex = ((synth_tex + 1) * (255 / 2))

    rendered_img, pred_lms, face_texture, mask, vert, tri, uv, rendered_img_white = face_model.inference(params,
                                                                                                         texture_map=tex)
    composed = composed.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    rendered_img = rendered_img[:, :, :, 0:3].clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    rendered_img_white = rendered_img_white[:, :, :, 0:3].clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(composed, 'RGB').save(f'{outdir}/proj_composed.png')
    PIL.Image.fromarray(rendered_img, 'RGB').save(f'{outdir}/proj_render.png')
    PIL.Image.fromarray(rendered_img_white, 'RGB').save(f'{outdir}/proj_render_geom.png')
    py3dio.save_obj(f'{outdir}/proj_model.obj', verts=vert.squeeze(), faces=tri, verts_uvs=uv, faces_uvs=tri,
                    texture_map=(tex.squeeze() / 255).clamp(0, 1))
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection()  # pylint: disable=no-value-for-parameter

# ----------------------------------------------------------------------------
