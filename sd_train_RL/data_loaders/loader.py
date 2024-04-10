from collections import deque
import torch
import numpy as np
from sd_train_RL.sd_aesthetics_train.train import calculate_log_probs
from fastprogress import master_bar, progress_bar


class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, prompt_fn, num):
        super().__init__()
        self.prompt_fn = prompt_fn
        self.num = num

    def __len__(self): return self.num

    def __getitem__(self, x): return self.prompt_fn()


class PerPromptStatTracker:
    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size
        self.min_count = min_count
        self.stats = {}

    def update(self, prompts, rewards):
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)
        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = deque(maxlen=self.buffer_size)
            self.stats[prompt].extend(prompt_rewards)

            if len(self.stats[prompt]) < self.min_count:
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-6
            else:
                mean = np.mean(self.stats[prompt])
                std = np.std(self.stats[prompt]) + 1e-6
            advantages[prompts == prompt] = (prompt_rewards - mean) / std

        return advantages


def sd_sample(prompts, pipe, height, width, guidance_scale, num_inference_steps, eta, device):
    scheduler = pipe.scheduler
    unet = pipe.unet
    text_embeddings = pipe._encode_prompt(prompts, device, 1, do_classifier_free_guidance=guidance_scale > 1.0)

    scheduler.set_timesteps(num_inference_steps, device=device)
    latents = torch.randn((len(prompts), unet.in_channels, height // 8, width // 8)).to(device)

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
        prev_sample_mean = scheduler_output.prev_sample
        prev_sample = prev_sample_mean + torch.randn_like(
            prev_sample_mean) * std_dev_t
        log_probs.append(calculate_log_probs(prev_sample, prev_sample_mean, std_dev_t).mean(
            dim=tuple(range(1, prev_sample_mean.ndim))))

        all_step_preds.append(prev_sample)
        latents = prev_sample

    return latents, torch.stack(all_step_preds), torch.stack(log_probs)