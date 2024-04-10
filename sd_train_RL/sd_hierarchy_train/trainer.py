import os
from metrics.hierarchy import Hierarchy, SubtreeInProb, SubtreeIS
from sd_train_RL.sd_aesthetics_train.train import calculate_log_probs, compute_loss
from sd_train_RL.data_loaders.loader import PromptDataset, PerPromptStatTracker
from sd_train_RL.utils import imagenet_animal_prompts
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from fastprogress import master_bar, progress_bar
from PIL import Image
import torchvision
from diffusers import StableDiffusionPipeline

dataset_classes_path = Path("imagenet.txt")
dataset_hierarchy = Hierarchy(dataset_classes_path.resolve())
model_name = 'resnet101'
model = getattr(torchvision.models, model_name)(weights="IMAGENET1K_V1")
model.eval()
transform = torchvision.models.get_model_weights(model_name).IMAGENET1K_V1.transforms()


torch.backends.cuda.matmul.allow_tf32 = True
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")
pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()
pipe.text_encoder.requires_grad_(False)
pipe.vae.requires_grad_(False)


num_samples_per_epoch = 128
num_epochs = 5
num_inner_epochs = 1
num_timesteps = 50
batch_size = 1
img_size = 32
lr = 5e-6
clip_advantages = 10.0
clip_ratio = 1e-4
cfg = 5.

train_set = PromptDataset(imagenet_animal_prompts, num_samples_per_epoch)
train_dl = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
per_prompt_stat_tracker = PerPromptStatTracker(buffer_size=32, min_count=16)


def sd_sample(prompts, pipe, height, width, guidance_scale, num_inference_steps, eta, device):
    scheduler = pipe.scheduler
    unet = pipe.unet
    text_embeddings = pipe._encode_prompt(
        prompts, device, 1, do_classifier_free_guidance=guidance_scale > 1.0
    )

    scheduler.set_timesteps(num_inference_steps, device=device)
    latents = torch.randn((len(prompts), unet.in_channels, height // 8, width // 8)).to(
        device
    )

    all_step_preds, log_probs = [latents], []

    for i, t in enumerate(progress_bar(scheduler.timesteps)):
        input = torch.cat([latents] * 2)
        input = scheduler.scale_model_input(input, t)

        pred = unet(input, t, encoder_hidden_states=text_embeddings).sample

        pred_uncond, pred_text = pred.chunk(2)
        pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)

        scheduler_output = scheduler.step(pred, t, latents, eta, variance_noise=0)
        t_1 = t - scheduler.config.num_train_timesteps // num_inference_steps
        variance = scheduler._get_variance(t, t_1)
        std_dev_t = eta * variance ** (0.5)
        prev_sample_mean = (scheduler_output.prev_sample)
        prev_sample = (prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t)
        log_probs.append(calculate_log_probs(prev_sample, prev_sample_mean, std_dev_t).mean(
            dim=tuple(range(1, prev_sample_mean.ndim))))
        all_step_preds.append(prev_sample)
        latents = prev_sample

    return latents, torch.stack(all_step_preds), torch.stack(log_probs)


def decoding_fn(latents, pipe):
    images = pipe.vae.decode(1 / 0.18215 * latents.cuda()).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    return images


def generation_by_prompt(images, prompt='dog', save_path='/gen-sd-test'):
    prompt_counter = 1
    save_path = save_path + prompt
    for img in images:
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            os.mkdir(save_path + f'/{prompt}')
        elif not os.path.exists(save_path + f'/{prompt}'):
            os.mkdir(save_path + f'/{prompt}')
        img = Image.fromarray(img)
        img.save(save_path + f"/{prompt}/{i}.jpg")
        prompt_counter += 1


def reward_fn(imgs, prompt, save_path='/gen-sd-test'):
    rewards = scs_scoring(imgs, prompt=prompt, save_path=save_path)
    return rewards


def sample_and_calculate_rewards(prompts, pipe, image_size, cfg, num_timesteps, decoding_fn, reward_fn, device):
    preds, all_step_preds, log_probs = sd_sample(prompts, pipe, image_size, image_size, cfg, num_timesteps, 1, device)
    imgs = decoding_fn(preds, pipe)
    rewards = reward_fn(imgs, prompts[0])
    return imgs, rewards, all_step_preds, log_probs


def scs_scoring(images, prompt='dog', save_path='/gen-sd-test'):
    generation_by_prompt(images, prompt, save_path=save_path)
    path_gen_img = save_path + prompt
    dataset = ImageFolder(path_gen_img, transform=transform)
    classes_list, _ = dataset.find_classes(path_gen_img)
    dataloader = DataLoader(dataset, batch_size=4)
    logits_list_dict = defaultdict(list)
    for images_batch, classes_batch in tqdm(dataloader):
        logits_batch = model(images_batch).cpu().detach().numpy()

    for logits, class_id in zip(logits_batch, classes_batch):
        logits_list_dict[classes_list[class_id]].append(logits)

    logits_dict = {f'{prompt}.n.01': np.stack(v) for k, v in logits_list_dict.items()}
    save_path = Path('/project/')
    save_path.mkdir(parents=True, exist_ok=True)
    save_path = save_path / f"{'metric'}_logits.npz"
    np.savez(save_path, **logits_dict)
    logits_path = {"model": save_path}
    synset_num_children = {}
    in_prob_metric = SubtreeInProb(dataset_hierarchy, 2)
    is_metric = SubtreeIS(dataset_hierarchy, 2)
    for synset in dataset_hierarchy.get_all_synsets(remove_leaves=True):
        n_ch = len(dataset_hierarchy.get_classifiable_subtree(synset))
        synset_num_children[synset.name()] = n_ch
    more_than_one_child_fn = lambda x: synset_num_children[x] > 1
    sorted_keys = np.array([k for k, v in sorted(synset_num_children.items(), key=lambda x: x[0])])
    n_ch_array = np.array([v for k, v in sorted(synset_num_children.items(), key=lambda x: x[0])])
    max_is_unnormed = np.log(n_ch_array[n_ch_array != 1]).mean()

    synset_num_children = {}
    in_prob_metric = SubtreeInProb(dataset_hierarchy, 2)
    is_metric = SubtreeIS(dataset_hierarchy, 2)
    for synset in dataset_hierarchy.get_all_synsets(remove_leaves=True):
        n_ch = len(dataset_hierarchy.get_classifiable_subtree(synset))
        synset_num_children[synset.name()] = n_ch
    more_than_one_child_fn = lambda x: synset_num_children[x] > 1
    sorted_keys = np.array([k for k, v in sorted(synset_num_children.items(), key=lambda x: x[0])])
    n_ch_array = np.array([v for k, v in sorted(synset_num_children.items(), key=lambda x: x[0])])
    max_is_unnormed = np.log(n_ch_array[n_ch_array != 1]).mean()
    model_results = {k: (in_prob_metric.compute_metric(v), is_metric.compute_metric(v)) for k, v in logits_path.items()}
    prediction = model_results['model'][1]['average']
    return prediction


pipe.unet.enable_gradient_checkpointing()
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=lr, weight_decay=1e-4)


all_step_preds_mean, log_probs_mean, advantages_mean, all_prompts_mean, all_rewards_mean = [], [], [], [], []
all_step_preds_iter, log_probs_iter, advantages_iter, all_prompts_iter, all_rewards_iter = [], [], [], [], []
mean_loss = []
lossess = []
for epoch in master_bar(range(5)):
    all_step_preds, log_probs, advantages, all_prompts, all_rewards = [], [], [], [], []

    for i, prompts in enumerate(list(range(1000))):
        if i % 2 == 0:
            prompts = ['cat'] * 500
        else:
            prompts = ['dog'] * 500
        batch_imgs, rewards, batch_all_step_preds, batch_log_probs = sample_and_calculate_rewards(prompts, pipe,
                                                                                                  img_size, cfg,
                                                                                                  num_timesteps,
                                                                                                  decoding_fn,
                                                                                                  reward_fn, 'cuda')
        batch_advantages = batch_advantages = torch.from_numpy(
            per_prompt_stat_tracker.update(np.array(prompts), [rewards.squeeze()] * 4)).float().to('cuda')
        all_step_preds.append(batch_all_step_preds)
        log_probs.append(batch_log_probs)
        advantages.append(batch_advantages)
        all_prompts += prompts
        all_rewards.append(rewards)
        all_step_preds_iter.append(batch_all_step_preds)
        log_probs_iter.append(batch_log_probs)
        advantages_iter.append(batch_advantages)
        all_rewards_iter.append(rewards)

    all_step_preds = torch.cat(all_step_preds, dim=1)
    log_probs = torch.cat(log_probs, dim=1)
    advantages = torch.cat(advantages)
    all_step_preds_mean.append(torch.mean(all_step_preds).item())
    log_probs_mean.append(torch.mean(log_probs).item())
    advantages_mean.append(torch.mean(advantages).item())
    all_rewards_mean.append(np.mean(all_rewards))

    for inner_epoch in progress_bar(range(num_inner_epochs)):
        all_step_preds_chunked = torch.chunk(all_step_preds, num_samples_per_epoch // batch_size, dim=1)
        log_probs_chunked = torch.chunk(log_probs, num_samples_per_epoch // batch_size, dim=1)
        advantages_chunked = torch.chunk(advantages, num_samples_per_epoch // batch_size, dim=0)

        all_prompts_chunked = [all_prompts[i:i + batch_size] for i in range(0, len(all_prompts), batch_size)]

        for i in progress_bar(range(len(all_step_preds_chunked))):
            optimizer.zero_grad()

            loss = compute_loss(all_step_preds_chunked[i], log_probs_chunked[i],
                                advantages_chunked[i], clip_advantages, clip_ratio, all_prompts_chunked[i], pipe,
                                num_timesteps, cfg, 1, 'cuda'
                                )
            lossess.append(loss)
            torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 1.0)
            optimizer.step()
        mean_loss.append(np.mean(lossess).item())
        print(mean_loss[-1])