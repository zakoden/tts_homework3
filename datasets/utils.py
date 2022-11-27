from text import text_to_sequence
import audio

import torch
import numpy as np
import os
import librosa
import time
from tqdm import tqdm
import pyworld as pw
from scipy.interpolate import interp1d

def process_text(train_text_path):
    with open(train_text_path, "r", encoding="utf-8") as f:
        txt = []
        for line in f.readlines():
            txt.append(line)

        return txt


def variance_preprocess(buffer, variance_name):
    min_value = None
    max_value = None
    for elem in buffer:
      cur_min = min(elem[variance_name])
      cur_max = max(elem[variance_name])
      if ((min_value is None) or (min_value > cur_min)):
        min_value = cur_min
      if ((max_value is None) or (max_value < cur_max)):
        max_value = cur_max

    return buffer, min_value, max_value


def get_data_to_buffer(train_config):
    stft_fn = audio.stft.STFT(hop_length=train_config.hop_length)

    wav_names = []
    for wav_name in os.listdir(train_config.wav_path):
      wav_names.append(wav_name)
    wav_names = sorted(wav_names)

    buffer = list()
    text = process_text(train_config.data_path)

    start = time.perf_counter()
    for i in tqdm(range(len(text))):
        # https://github.com/ming024/FastSpeech2/blob/master/preprocessor/preprocessor.py
        wav_file_name = os.path.join(train_config.wav_path, wav_names[i])
        wav, _ = librosa.load(wav_file_name)
        pitch, t = pw.dio(
                    wav.astype(np.float64),
                    train_config.sampling_rate,
                    frame_period=train_config.hop_length / train_config.sampling_rate * 1000,
                )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, train_config.sampling_rate)

        mel_gt_name = os.path.join(
            train_config.mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1))
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(os.path.join(
            train_config.alignment_path, str(i)+".npy"))
        pitch = pitch[:sum(duration)] # now it is pitch contour with zeros for non-voice
        character = text[i][0:len(text[i])-1]
        character = np.array(
            text_to_sequence(character, train_config.text_cleaners))
        
        # pitch interpolation 
        #nonzero_ids = np.where(pitch != 0)[0]
        #interp_fn = interp1d(
        #    nonzero_ids,
        #    pitch[nonzero_ids],
        #    fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
        #    bounds_error=False,
        #)
        #pitch = interp_fn(np.arange(0, len(pitch)))

        # energy
        magnitudes, phases = stft_fn.transform(torch.from_numpy(wav)[None, :])
        energy = torch.norm(magnitudes, dim=1)
        energy = energy[0,:sum(duration)]

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        pitch = torch.from_numpy(pitch)
        mel_gt_target = torch.from_numpy(mel_gt_target)

        buffer.append({"text": character, "duration": duration,
                       "mel_target": mel_gt_target, "pitch":pitch, "energy":energy})
        
    buffer, pitch_min, pitch_max = variance_preprocess(buffer, "pitch")
    buffer, energy_min, energy_max = variance_preprocess(buffer, "energy")
        
    variance_stats = {
        "pitch_min": pitch_min.item(),
        "pitch_max": pitch_max.item(),
        "energy_min": energy_min.item(),
        "energy_max": energy_max.item()
    }

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer, variance_stats
