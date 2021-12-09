# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
from facenet_pytorch import MTCNN
import mediapipe as mp
import pickle
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from torch_utils.misc import load_lm3d
from training.models import ReconModel, ReconModelMP, ReconModelOrig, DFRModel, RigNet, ResNet, BasicBlock
from training.deepfacemodel import ReconNetWrapper
from scipy.io import loadmat
import os.path as osp
import dnnlib
import legacy
import yaml
import matplotlib.pyplot as plt
# import onnx
import numpy as np
# from onnx import numpy_helper
from FaceBoxes import FaceBoxesPytorch
from torch_utils.misc import crop_img, POS, resize_n_crop_img
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
# from TDDFA import TDDFA
# from models.resnet22 import Model as tddfa_model

FACE_MODEL_PATH = 'BFM/BFM_model_front_msft.mat'
TAR_SIZE = 1024


# ----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):  # to be overridden by subclass
        raise NotImplementedError()


# ----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9, r1_gamma=10,
                 pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

        self.dfr_model = DFRModel().to(device)
        # resume_path_dfr = 'saved_models/params_160000.pth'
        # resume_path_rignet = 'saved_models/rignet_320000.pth'

        # resume_path_dfr = 'saved_models/params_137000.pth'
        # resume_path_rignet = 'saved_models/rignet_410000.pth'

        resume_path_dfr = 'saved_models/params_1407000.pth'
        resume_path_rignet_rot = 'saved_models/rignet_no_trans2_84000.pth'
        resume_path_rignet_exp = 'saved_models/rignet_exp_only_70000.pth'

        print(f'loading model {resume_path_dfr}')
        self.dfr_model.load_state_dict(
            torch.load(resume_path_dfr, map_location=device))

        self.RigNetRot = RigNet().to(device)
        self.RigNetExp = RigNet().to(device)

        print(f'loading model {resume_path_rignet_rot}')
        self.RigNetRot.load_state_dict(
            torch.load(resume_path_rignet_rot, map_location=device))

        print(f'loading model {resume_path_rignet_exp}')
        self.RigNetExp.load_state_dict(
            torch.load(resume_path_rignet_exp, map_location=device))

        try:
            facemodel = loadmat(FACE_MODEL_PATH)
        except Exception as e:
            print('failed to load %s' % FACE_MODEL_PATH)

        # self.face_model = ReconModel(facemodel, img_size=TAR_SIZE, device=device).to(device)
        self.face_model = ReconModelOrig(facemodel, img_size=[TAR_SIZE,TAR_SIZE], device=device).to(device)
        self.face_model_mp = ReconModelMP(device=device).to(device)
        network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl'
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore
        self.trained_G = G

        # self.tddfa = ResNet(BasicBlock, [3, 4, 3], num_landmarks=136, input_channel=3, fc_flg=False)
        # onnx_model = onnx.load('weights/resnet22.onnx')

        # graph = onnx_model.graph
        # initalizers = dict()
        # for init in graph.initializer:
        #     initalizers[init.name] = numpy_helper.to_array(init)
        #
        # for name, p in self.tddfa.named_parameters():
        #     p.data = (torch.from_numpy(initalizers[name])).data.clone()
        # self.tddfa.to(self.device)
        # self.face_boxes = FaceBoxesPytorch(device=self.device)

        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.99)
        self.mtcnn = MTCNN(device=self.device, keep_all=False, selection_method='probability')
        self.lm3d_std = self.load_lm3d('BFM')
        net_recon_path = 'saved_models/epoch_20.pth'
        self.net_recon = ReconNetWrapper(
            net_recon='resnet50', use_last_fc=False, init_path=None).to(self.device)
        self.net_recon.load_state_dict(
            torch.load(net_recon_path, map_location=self.device)['net_recon'])
        self.net_recon = self.net_recon.eval()

    def load_lm3d(self, bfm_folder):

        Lm3D = loadmat(osp.join(bfm_folder, 'similarity_Lm3D_all.mat'))
        Lm3D = Lm3D['lm']

        # calculate 5 facial landmarks using 68 landmarks
        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(
            Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
        Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

        return Lm3D

    def run_G_3D(self, z, c, sync):
        ws_trained = self.trained_G.mapping(z, c)
        ws = self.G_mapping(z, c)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                                     torch.full_like(cutoff, ws.shape[1]))
                mixing_z = torch.randn_like(z)
                ws[:, cutoff:] = self.G_mapping(mixing_z, c, skip_w_avg_update=True)[:, cutoff:]
                ws_trained[:, cutoff:] = self.trained_G.mapping(mixing_z, c, skip_w_avg_update=True)[:, cutoff:]

        p = self.dfr_model(ws_trained)
        ones = torch.ones([ws.shape[0], 1], device=ws.device)
        rot_sign = torch.where(torch.rand([ws.shape[0], 1], device=ws.device) > 0.5, ones, -ones)

        p_edit_rot = p.clone()
        # p_edit_exp = p.clone()
        p_edit_rot[:, 224:227] = rot_sign * (
            p_edit_rot[torch.randperm(p_edit_rot.shape[0]), 224:227])  # change rotation within batch
        p_edit_rot[:, 225] = (6 * torch.rand(1) - 3).to(self.device) - p_edit_rot[:, 225]

        # p_edit_exp[:, 80: 144] = p_edit_exp[torch.randperm(p_edit_exp.shape[0]),
        #                          80: 144]  # - change expression within batch

        d_rot = self.RigNetRot(ws_trained, p_edit_rot)
        # d_exp = self.RigNetExp(ws_trained, p_edit_exp)
        # d_tex = self.RigNetRot(ws, p_edit_rot)
        ws_modified = ws_trained + d_rot  # + d_exp
        # cutoff = 6
        # ws_tex = torch.empty_like(ws)
        # ws_tex[:, cutoff:] = (ws + d_tex)[:, cutoff:]
        # ws_tex[:, :cutoff] = ws[:, :cutoff]
        # ws_tex = ws + d_tex
        ws_tex = ws
        with misc.ddp_sync(self.G_synthesis, sync):
            tex_image_orig = self.G_synthesis(ws_tex)

        bg_img = self.trained_G.synthesis(ws_modified)
        flip = torch.rand([bg_img.shape[0]], device=ws.device) < 0.2
        bg_img[flip, :] = bg_img[flip, :].flip(3)

        bg_img_normalized = ((bg_img + 1) * (255 / 2)).clamp(0, 255)
        bg_img_mp = bg_img_normalized.permute((0, 2, 3, 1)).detach().cpu().numpy().astype(np.uint8)

        boxes, probs, points = self.mtcnn.detect(bg_img_normalized.permute(0, 2, 3, 1), landmarks=True)
        cropped_all = []
        lms = []
        bboxes = []
        for im_batch, pt_batch, mp_img in zip(bg_img_normalized, points, bg_img_mp):
            if pt_batch is not None:
                t, s = POS(pt_batch[0].squeeze().copy(), self.lm3d_std)
                cropped, left, up = resize_n_crop_img(im_batch, t, s, target_size=224.)
                cropped_all.append(cropped.squeeze())
                bboxes.append(torch.Tensor([left, up, s]).to(self.device))
                for i in range(2):
                    result = self.mp_face_mesh.process(mp_img.squeeze())
                if not result.multi_face_landmarks:
                    lms.append(-np.ones([468, 3]))
                else:
                    lms.append(np.array([[lm.x, lm.y, lm.z] for lm in result.multi_face_landmarks[0].landmark]))
            else:
                cropped_all.append(torch.zeros((3, 224, 224)).to(self.device))
                bboxes.append(torch.Tensor([2000, 2000, 1]).to(self.device))
                lms.append(-np.ones([468, 3]))
        lms = np.stack(lms)
        lms = torch.from_numpy(lms).to(self.device)
        cropped = torch.stack(cropped_all)
        bboxes = torch.stack(bboxes)

        params = self.net_recon(cropped.type(torch.uint8).type(torch.float32) / 255)
        tex_image = tex_image_orig.permute(0, 2, 3, 1)

        mask_mouth = self.face_model_mp(landmarks=lms)
        rendered_img, pred_lms, face_texture, _, mask = self.face_model(params,
                                                                        texture_map=tex_image, bboxs=bboxes)

        # mask[mask_mouth] = 0
        composed_image = bg_img * (~mask[:, None, :, :]) + mask[:, None, :, :] * (rendered_img[:, :3, :, :])
        return composed_image, tex_image_orig, ws_tex

    # def run_G_3D(self, z, c, sync):
    #
    #     ws_trained = self.trained_G.mapping(z, c)
    #     ws = self.G_mapping(z, c)
    #     if self.style_mixing_prob > 0:
    #         with torch.autograd.profiler.record_function('style_mixing'):
    #             cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
    #             cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
    #                                  torch.full_like(cutoff, ws.shape[1]))
    #             mixing_z = torch.randn_like(z)
    #             ws[:, cutoff:] = self.G_mapping(mixing_z, c, skip_w_avg_update=True)[:, cutoff:]
    #             ws_trained[:, cutoff:] = self.trained_G.mapping(mixing_z, c, skip_w_avg_update=True)[:, cutoff:]
    #
    #     p = self.dfr_model(ws_trained)
    #     ones = torch.ones([ws.shape[0], 1], device=ws.device)
    #     rot_sign = torch.where(torch.rand([ws.shape[0], 1], device=ws.device) > 0.5, ones, -ones)
    #
    #     p_edit_rot = p.clone()
    #     p_edit_exp = p.clone()
    #     p_edit_rot[:, 224:227] = rot_sign * (
    #         p_edit_rot[torch.randperm(p_edit_rot.shape[0]), 224:227])  # change rotation within batch
    #     p_edit_rot[:, 225] = (6 * torch.rand(1) - 3).to(self.device) - p[:, 225]
    #
    #     p_edit_exp[:, 80: 144] = p_edit_exp[torch.randperm(p_edit_exp.shape[0]),
    #                              80: 144]  # - change expression within batch
    #
    #     d_rot = self.RigNetRot(ws_trained, p_edit_rot)
    #     d_exp = self.RigNetExp(ws_trained, p_edit_exp)
    #     # d_tex = self.RigNetRot(ws, p)
    #     ws_modified = ws_trained + d_rot + d_exp
    #
    #     # p_rot = self.dfr_model(ws_modified)
    #
    #     # cutoff = 6
    #     # ws_tex = torch.empty_like(ws)
    #     # ws_tex[:, cutoff:] = (ws + d_tex)[:, cutoff:]
    #     # ws_tex[:, :cutoff] = ws[:, :cutoff]
    #
    #     ws_tex = ws
    #     with misc.ddp_sync(self.G_synthesis, sync):
    #         tex_image_orig = self.G_synthesis(ws_tex)
    #
    #     bg_img = self.trained_G.synthesis(ws_modified)
    #     flip = torch.rand([bg_img.shape[0]], device=ws.device) < 0.2
    #     bg_img[flip, :] = bg_img[flip, :].flip(3)
    #     bg_img_normalized = (bg_img * 128 + 127.5).clamp(0, 255).permute((0, 2, 3, 1)).detach().cpu().numpy().astype(
    #         np.uint8)
    #     lms = []
    #     for img in bg_img_normalized:
    #         for i in range(2):
    #             result = self.mp_face_mesh.process(img.squeeze())
    #
    #         if not result.multi_face_landmarks:
    #             lms.append(-np.ones([468, 3]))
    #             continue
    #         lms.append(np.array([[lm.x, lm.y, lm.z] for lm in result.multi_face_landmarks[0].landmark]))
    #         # mean_x = 0.5 * (np.min(lms[-1][:, 0]) + np.max(lms[-1][:, 0]))
    #         # lms[-1][:, 0] = 0.97 * (lms[-1][:, 0] - mean_x) + mean_x
    #     lms = np.stack(lms)
    #     lms = torch.from_numpy(lms).to(self.device)
    #
    #     tex_image = tex_image_orig.permute(0, 2, 3, 1)
    #
    #     rendered_img, face_texture, mask = self.face_model_mp(landmarks=lms, tex=tex_image)
    #     rendered_img = rendered_img.permute(0, 3, 1, 2)
    #
    #     composed_image = bg_img * (~mask[:, None, :, :]) + mask[:, None, :, :] * (rendered_img[:, :3, :, :])
    #
    #     return composed_image, tex_image_orig, ws_tex

    def run_G(self, z, c, sync):
        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                                         torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(ws)
        return img, ws

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Gpl = (phase in ['Greg', 'Gboth']) and (self.pl_weight != 0)
        do_Dr1 = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _, _gen_ws = self.run_G_3D(gen_z, gen_c, sync=(sync and not do_Gpl))  # May get synced by Gpl.
                # gen_img, gen_ws = self.run_G(gen_z, gen_c, sync=(sync and not do_Gpl))
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                _, tex_img, gen_ws = self.run_G_3D(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                # tex_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size], sync=sync)
                pl_noise = torch.randn_like(tex_img) / np.sqrt(tex_img.shape[2] * tex_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = \
                        torch.autograd.grad(outputs=[(tex_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True,
                                            only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (tex_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _, _gen_ws = self.run_G_3D(gen_z, gen_c, sync=False)
                # gen_img, _gen_ws = self.run_G(gen_z, gen_c, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False)  # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = \
                            torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True,
                                                only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

# ----------------------------------------------------------------------------
