import sys
sys.path.append('.')

import os
import numpy as np
import torch
import collections
import warnings

import numpy as np
import torch
from torch import nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler  import OneCycleLR

import waveglow
import text
import audio
import utils

from tts_homework3.configs import mel_config, model_config, train_config
from tts_homework3 import datasets
from tts_homework3 import dataloader
from tts_homework3.model import FastSpeech
from tts_homework3 import losses
from tts_homework3 import synthesis
from tts_homework3.logger import logger
import argparse

warnings.filterwarnings("ignore", category=UserWarning)

SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(args):
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
    
    WaveGlow = utils.get_WaveGlow()
    WaveGlow = WaveGlow.cuda()
    
    os.makedirs(args.output, exist_ok=True)

    model.eval()

    tests = []
    log_data = []
    column_names = ["text", "step", "duration_alpha", "pitch_alpha", "energy_alpha", "audio"]

    f = open(args.input, "r")
    for line in f:
      line_lst = line.split(" ")
      alpha_d = int(line_lst[0])
      alpha_p = int(line_lst[1])
      alpha_e = int(line_lst[2])
      cur_text = str(line_lst[-1])
      log_data.append([cur_text, logger.step, alpha_d, alpha_p, alpha_e, None])
      tests.append(cur_text)
    f.close()
    
    data_list = list(text.text_to_sequence(test, train_config.text_cleaners) for test in tests)
    
    for i, phn in enumerate(data_list):
      alpha_d = log_data[i][1]
      alpha_p = log_data[i][2]
      alpha_e = log_data[i][3]
      mel_cuda = one_synthesis(model, phn, alpha_duration=alpha_d, alpha_pitch=alpha_p, alpha_energy=alpha_e)
      cur_path = f"results/d=({alpha_d})_p=({alpha_p})_e=({alpha_e})_{i}_waveglow.wav"
      waveglow.inference.inference(mel_cuda, WaveGlow, cur_path)
      cur_audio = wandb.Audio(cur_path)
      log_data[i][-1] = cur_audio

    table = wandb.Table(data=log_data, columns=column_names)
    logger.wandb.log({"gen_audio": table}, step=logger.step)


if __name__ == "__main__":
    args.add_argument(
        "-i",
        "--input",
        default="input.txt",
        type=str,
        help="Path to read text.",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output",
        type=str,
        help="Path to write results.",
    )
  
    args = args.parse_args()
    main(args)
