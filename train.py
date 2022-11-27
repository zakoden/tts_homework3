import collections
import warnings

import numpy as np
import torch
import sys
sys.path.append('.')

from torch.utils.data import DataLoader
import waveglow
import text
import audio
import utils

from .configs import mel_config, model_config, train_config
from . import datasets
from . import dataloader
from .model import FastSpeech
from . import losses
from . import synthesis

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main():
    buffer, variance_stats = datasets.get_data_to_buffer(train_config)
    dataset = datasets.BufferDataset(buffer)
    
    training_loader = DataLoader(
        dataset,
        batch_size=train_config.batch_expand_size * train_config.batch_size,
        shuffle=True,
        collate_fn=dataloader.collate_fn_tensor,
        drop_last=True,
        num_workers=0
    )
    
    model = FastSpeech(model_config, variance_stats)
    model = model.to(train_config.device)

    fastspeech_loss = losses.FastSpeechLoss()
    current_step = 0

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(0.9, 0.98),
        eps=1e-9)

    scheduler = OneCycleLR(optimizer, **{
        "steps_per_epoch": len(training_loader) * train_config.batch_expand_size,
        "epochs": train_config.epochs,
        "anneal_strategy": "cos",
        "max_lr": train_config.learning_rate,
        "pct_start": 0.1
    })
    
    logger = WanDBWriter(train_config)
    
    WaveGlow = utils.get_WaveGlow()
    WaveGlow = WaveGlow.cuda()
    
    for epoch in range(train_config.epochs):
        for i, batchs in enumerate(training_loader):
            # real batch start here
            for j, db in enumerate(batchs):
                current_step += 1
                #tqdm_bar.update(1)

                logger.set_step(current_step)

                # Get Data
                character = db["text"].long().to(train_config.device)
                mel_target = db["mel_target"].float().to(train_config.device)
                pitch = db["pitch"].float().to(train_config.device)
                energy = db["energy"].float().to(train_config.device)
                duration = db["duration"].int().to(train_config.device)
                mel_pos = db["mel_pos"].long().to(train_config.device)
                src_pos = db["src_pos"].long().to(train_config.device)
                max_mel_len = db["mel_max_len"]

                # Forward
                mel_output, duration_predictor_output, pitch_predictor_output, energy_predictor_output = model(
                                                              character,
                                                              src_pos,
                                                              mel_pos=mel_pos,
                                                              mel_max_length=max_mel_len,
                                                              length_target=duration,
                                                              pitch_target=pitch, 
                                                              energy_target=energy, 
                                                              alpha_duration=1.0, 
                                                              alpha_pitch=1.0, 
                                                              alpha_energy=1.0)

                # Calc Loss
                mel_loss, duration_loss, pitch_loss, energy_loss = fastspeech_loss(mel_output,
                                                                                   duration_predictor_output,
                                                                                   pitch_predictor_output, 
                                                                                   energy_predictor_output,
                                                                                   mel_target,
                                                                                   duration,
                                                                                   pitch,
                                                                                   energy)
                total_loss = mel_loss + duration_loss + pitch_loss + energy_loss

                # Logger
                t_l = total_loss.detach().cpu().numpy()
                m_l = mel_loss.detach().cpu().numpy()
                d_l = duration_loss.detach().cpu().numpy()
                p_l = pitch_loss.detach().cpu().numpy()
                e_l = energy_loss.detach().cpu().numpy()

                logger.add_scalar("energy_loss", e_l)
                logger.add_scalar("pitch_loss", p_l)
                logger.add_scalar("duration_loss", d_l)
                logger.add_scalar("mel_loss", m_l)
                logger.add_scalar("total_loss", t_l)

                # Backward
                total_loss.backward()

                # Clipping gradients to avoid gradient explosion
                nn.utils.clip_grad_norm_(
                    model.parameters(), train_config.grad_clip_thresh)

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                if current_step % train_config.save_step == 0:
                    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                    )}, os.path.join(train_config.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                    print("save model at step %d ..." % current_step)

                    synthesis.log_synthesis(model)


if __name__ == "__main__":
    main()
