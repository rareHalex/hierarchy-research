import numpy as np
import torch
from torch import nn
import clip


class MLP(nn.Module):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)


clip_model, preprocess = clip.load("ViT-L/14", device="cuda")
aesthetic_model = MLP(768)


def aesthetic_model_normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


def reward_fn(imgs, device):
    clip_model.to(device)
    aesthetic_model.to(device)
    rewards = aesthetic_scoring(imgs, preprocess, clip_model, aesthetic_model_normalize, aesthetic_model)
    clip_model.to('cpu')
    aesthetic_model.to('cpu')
    return rewards


def aesthetic_scoring(img, preprocess, clip_model, aesthetic_model_normalize, aesthetic_model):
    image = preprocess(img).unsqueeze(0).cuda()
    with torch.no_grad(): image_features = clip_model.encode_image(image)
    im_emb_arr = aesthetic_model_normalize(image_features.cpu().detach().numpy())
    prediction = aesthetic_model(torch.from_numpy(im_emb_arr).float().cuda())
    return prediction