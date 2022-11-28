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
import wandb

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
    variance_stats = {
      'pitch_min': 0.0,
      'pitch_max': 861.0652680130653,
      'energy_min': 0.013677556999027729,
      'energy_max': 246.5960235595703
    }

    model = FastSpeech(model_config, variance_stats)
    model_weights = (torch.load(args.checkpoint))['model']
    model.load_state_dict(model_weights)
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
      alpha_d = float(line_lst[0])
      alpha_p = float(line_lst[1])
      alpha_e = float(line_lst[2])
      cur_text = str(" ".join(line_lst[3:]))[:-1]
      log_data.append([cur_text, logger.step, alpha_d, alpha_p, alpha_e, None])
      tests.append(cur_text)
    f.close()
    
    data_list = list(text.text_to_sequence(test, train_config.text_cleaners) for test in tests)
    
    for i, phn in enumerate(data_list):
      alpha_d = log_data[i][2]
      alpha_p = log_data[i][3]
      alpha_e = log_data[i][4]
      print("|" + tests[i] + "|")
      print("alpha_duration=", alpha_d, "alpha_pitch=", alpha_p, "alpha_energy=", alpha_e)
      mel_cuda = synthesis.one_synthesis(model, phn, alpha_duration=alpha_d, alpha_pitch=alpha_p, alpha_energy=alpha_e)
      cur_path = args.output + f"/d=({alpha_d})_p=({alpha_p})_e=({alpha_e})_{i}_waveglow.wav"
      waveglow.inference.inference(mel_cuda, WaveGlow, cur_path)
      cur_audio = wandb.Audio(cur_path)
      log_data[i][-1] = cur_audio

    table = wandb.Table(data=log_data, columns=column_names)
    logger.wandb.log({"gen_audio": table}, step=logger.step)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="test TTS")
    args.add_argument(
        "-c",
        "--checkpoint",
        default="checkpoint",
        type=str,
        help="Path to model checkpoint.",
    )
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
