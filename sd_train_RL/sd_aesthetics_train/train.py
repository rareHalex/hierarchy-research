import math
import numpy as np
import torch
from fastprogress import master_bar, progress_bar
from diffusers import StableDiffusionPipeline, DDIMScheduler
from sd_train_RL.data_loaders import loader
from sd_train_RL.networks import rl_models
from sd_train_RL import utils

torch.backends.cuda.matmul.allow_tf32 = True
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cuda")
pipe.enable_attention_slicing()
pipe.text_encoder.requires_grad_(False)
pipe.vae.requires_grad_(False)


def calculate_log_probs(prev_sample, prev_sample_mean, std_dev_t):
    std_dev_t = torch.clip(std_dev_t, 1e-6)
    log_probs = -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * std_dev_t ** 2) - torch.log(std_dev_t) - math.log(math.sqrt(2 * math.pi))
    return log_probs


def compute_loss(x_t, original_log_probs, advantages, clip_advantages, clip_ratio, prompts, pipe, num_inference_steps,
                 guidance_scale, eta, device):
    scheduler = pipe.scheduler
    unet = pipe.unet
    text_embeddings = pipe._encode_prompt(prompts, device, 1, do_classifier_free_guidance=guidance_scale > 1.0).detach()
    scheduler.set_timesteps(num_inference_steps, device=device)
    loss_value = 0.
    for i, t in enumerate(progress_bar(scheduler.timesteps)):
        clipped_advantages = torch.clip(advantages, -clip_advantages, clip_advantages).detach()

        input = torch.cat([x_t[i].detach()] * 2)
        input = scheduler.scale_model_input(input, t)

        pred = unet(input, t, encoder_hidden_states=text_embeddings).sample

        pred_uncond, pred_text = pred.chunk(2)
        pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)

        scheduler_output = scheduler.step(pred, t, x_t[i].detach(), eta, variance_noise=0)
        t_1 = t - scheduler.config.num_train_timesteps // num_inference_steps
        variance = scheduler._get_variance(t, t_1)
        std_dev_t = eta * variance ** (0.5)
        prev_sample_mean = scheduler_output.prev_sample
        current_log_probs = calculate_log_probs(x_t[i + 1].detach(), prev_sample_mean, std_dev_t).mean(
            dim=tuple(range(1, prev_sample_mean.ndim)))

        ratio = torch.exp(current_log_probs - original_log_probs[
            i].detach())
        unclipped_loss = -clipped_advantages * ratio
        clipped_loss = -clipped_advantages * torch.clip(ratio, 1. - clip_ratio,
                                                        1. + clip_ratio)
        loss = torch.max(unclipped_loss,
                         clipped_loss).mean()
        loss.backward()

        loss_value += loss.item()
    return loss_value


def sample_and_calculate_rewards(prompts, pipe, image_size, cfg, num_timesteps, decoding_fn, reward_fn, device):
    preds, all_step_preds, log_probs = loader.sd_sample(prompts, pipe, image_size, image_size, cfg, num_timesteps, 1, device)
    imgs = decoding_fn(preds,pipe)
    rewards = reward_fn(imgs, device)
    return imgs, rewards, all_step_preds, log_probs


def train(num_epochs=10):
    train_set = loader.PromptDataset(utils.imagenet_animal_prompts, 128)
    train_dl = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=0)
    per_prompt_stat_tracker = loader.PerPromptStatTracker(buffer_size=32, min_count=16)
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in master_bar(range(num_epochs)):
        all_step_preds, log_probs, advantages, all_prompts, all_rewards = [], [], [], [], []

        for i, prompts in enumerate(progress_bar(train_dl)):
            batch_imgs, rewards, batch_all_step_preds, batch_log_probs = sample_and_calculate_rewards(prompts, pipe,
                                                                                                      512, 5.0,
                                                                                                      50,
                                                                                                      utils.decoding_fn,
                                                                                                      rl_models.reward_fn, 'cuda')
            batch_advantages = torch.from_numpy(
                per_prompt_stat_tracker.update(np.array(prompts), rewards.squeeze().cpu().detach().numpy())).float().to(
                'cuda')
            all_step_preds.append(batch_all_step_preds)
            log_probs.append(batch_log_probs)
            advantages.append(batch_advantages)
            all_prompts += prompts
            all_rewards.append(rewards)

        all_step_preds = torch.cat(all_step_preds, dim=1)
        log_probs = torch.cat(log_probs, dim=1)
        advantages = torch.cat(advantages)
        all_rewards = torch.cat(all_rewards)

        for _ in progress_bar(range(1)):
            all_step_preds_chunked = torch.chunk(all_step_preds, 128 // 2, dim=1)
            log_probs_chunked = torch.chunk(log_probs, 128 // 2, dim=1)
            advantages_chunked = torch.chunk(advantages, 128 // 2, dim=0)

            all_prompts_chunked = [all_prompts[i:i + 2] for i in range(0, len(all_prompts), 2)]

            for i in progress_bar(range(len(all_step_preds_chunked))):
                optimizer.zero_grad()

                loss = compute_loss(all_step_preds_chunked[i], log_probs_chunked[i],
                                    advantages_chunked[i], 10, 1e-4, all_prompts_chunked[i], pipe,
                                    50, 5., 1, 'cuda'
                                    )

                torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 1.0)
                optimizer.step()
                print(loss)