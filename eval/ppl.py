import argparse

import torch
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm

import lpips
# from model import Generator


def normalize(x):
    return x / torch.sqrt(x.pow(2).sum(-1, keepdim=True))


def slerp(a, b, t):
    a = normalize(a)
    b = normalize(b)
    d = (a * b).sum(-1, keepdim=True)
    p = t * torch.acos(d)
    c = normalize(b - d * a)
    d = a * torch.cos(p) + c * torch.sin(p)

    return normalize(d)


def lerp(a, b, t):
    return a + (b - a) * t


def compute_ppl(g, num_samples=50000, epsilon=1e-4, space='w', sampling='end', crop=False, batch_size=25, device='cuda'):
    percept = lpips.LPIPS(net='vgg')
    g.to(device)

    distances = []

    n_batch = num_samples // batch_size
    resid = num_samples - (n_batch * batch_size)
    batch_sizes = [batch_size] * n_batch + [resid] if resid != 0 else [batch_size] * n_batch

    with torch.no_grad():
        for batch in tqdm(batch_sizes):
            noise = g.make_noise()
            latent_dim = 512
            inputs = torch.randn([batch * 2, latent_dim], device=device)
            if sampling == 'full':
                lerp_t = torch.rand(batch, device=device)
            else:
                lerp_t = torch.zeros(batch, device=device)

            if space == 'w':
                latent = g.get_latent(inputs)
                latent_t0, latent_t1 = latent[::2], latent[1::2]
                latent_e0 = lerp(latent_t0, latent_t1, lerp_t[:, None])
                latent_e1 = lerp(latent_t0, latent_t1, lerp_t[:, None] + epsilon)
                latent_e = torch.stack([latent_e0, latent_e1], 1).view(*latent.shape)

            image, _ = g([latent_e], input_is_latent=True, noise=noise)

            if crop:
                c = image.shape[2] // 8
                image = image[:, :, c * 3 : c * 7, c * 2 : c * 6]

            factor = image.shape[2] // 256

            if factor > 1:
                image = F.interpolate(
                    image, size=(256, 256), mode='bilinear', align_corners=False
                )

            dist = percept(image[::2], image[1::2]).view(image.shape[0] // 2) / (
                epsilon ** 2
            )
            distances.append(dist.to('cpu').numpy())
    g.to('cpu')

    distances = np.concatenate(distances, 0)

    lo = np.percentile(distances, 1, interpolation='lower')
    hi = np.percentile(distances, 99, interpolation='higher')
    filtered_dist = np.extract(
        np.logical_and(lo <= distances, distances <= hi), distances
    )

    ppl = filtered_dist.mean()
    return ppl


# if __name__ == '__main__':
#     device = 'cuda'

#     parser = argparse.ArgumentParser()

#     parser.add_argument('--space', choices=['z', 'w'])
#     parser.add_argument('--batch', type=int, default=64)
#     parser.add_argument('--n_sample', type=int, default=5000)
#     parser.add_argument('--size', type=int, default=256)
#     parser.add_argument('--eps', type=float, default=1e-4)
#     parser.add_argument('--crop', action='store_true')
#     parser.add_argument('--sampling', default='end', choices=['end', 'full'])
#     parser.add_argument('ckpt', metavar='CHECKPOINT')

#     args = parser.parse_args()

#     latent_dim = 512

#     ckpt = torch.load(args.ckpt)

#     g = Generator(args.size, latent_dim, 8).to(device)
#     g.load_state_dict(ckpt['g_ema'])
#     g.eval()


