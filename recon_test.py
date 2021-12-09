import glob
import numpy as np
import os.path as osp
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from light_cnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2
import torchvision.transforms as transforms

input_path = 'celeba_test/*'
use_crop = True
sample_dirs = glob.glob(input_path)
ssim_all, psnr_all, L1_all, feat_sim_all = [], [], [], []
light_cnn_model_path = 'saved_models/LightCNN_29Layers_V2_checkpoint.pth'

model = LightCNN_29Layers_v2(num_classes=80013)
model.eval()
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load(light_cnn_model_path)
model.load_state_dict(checkpoint['state_dict'])
transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([128, 128])])

cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
def process_for_model(img):
    img = img.mean(2)[:, :, None]
    return img


for dir in tqdm(sample_dirs):

    if use_crop:
        target_path = osp.join(dir, 'target_crop.png')
        mask_path = osp.join(dir, 'mask_crop.png')
        recon_path = osp.join(dir, 'proj_composed_crop.png')
        target = np.array(Image.open(target_path)) / 255
        mask = np.array(Image.open(mask_path)) / 255
        recon = np.array(Image.open(recon_path)) / 255

    else:
        target_path = osp.join(dir, 'target.png')
        mask_path = osp.join(dir, 'mask.png')
        recon_path = osp.join(dir, 'proj_composed.png')
        target = np.array(Image.open(target_path))[20:-20:, :] / 255
        mask = np.array(Image.open(mask_path))[20:-20, :] / 255
        recon = np.array(Image.open(recon_path))[20:-20, :, :] / 255

    target_for_model = transform(process_for_model(target)).cuda()[None, ...].float()
    recon_for_model = transform(process_for_model(recon)).cuda()[None, ...].float()

    _, target_features = model(target_for_model)
    _, recon_features = model(recon_for_model)

    sim = cos_sim(target_features, recon_features)

    target = target * mask[:, :, None]
    recon = recon * mask[:, :, None]

    ssim_score = ssim(target, recon, data_range=recon.max() - recon.min(), multichannel=True)
    psnr_score = psnr(target, recon, data_range=recon.max() - recon.min())
    L1_score = np.abs(target - recon).sum()/(mask.sum()*3)

    ssim_all.append(ssim_score)
    psnr_all.append(psnr_score)
    L1_all.append(L1_score)
    feat_sim_all.append(sim.item())

ssim_all = np.array(ssim_all)
psnr_all = np.array(psnr_all)
L1_all = np.array(L1_all)
cos_sim = np.array(feat_sim_all)


print(f'SSIM total: {ssim_all.mean()} PSNR total: {psnr_all.mean()} L1 total: {L1_all.mean()} COS sim feat: {cos_sim.mean()}')
