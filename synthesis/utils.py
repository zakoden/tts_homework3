import wandb

def one_synthesis(model, phn, alpha_duration=1.0, alpha_pitch=1.0, alpha_energy=1.0):
    text = np.array(phn)
    text = np.stack([text])
    src_pos = np.array([i+1 for i in range(text.shape[1])])
    src_pos = np.stack([src_pos])
    sequence = torch.from_numpy(text).long().to(train_config.device)
    src_pos = torch.from_numpy(src_pos).long().to(train_config.device)
    
    with torch.no_grad():
        mel = model.forward(sequence, src_pos, 
                            alpha_duration=alpha_duration, 
                            alpha_pitch=alpha_pitch, 
                            alpha_energy=alpha_energy)
    return mel.contiguous().transpose(1, 2)

def log_synthesis(model):
    tests = [ 
        "A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
        "Massachusetts Institute of Technology may be best known for its math, science and engineering education",
        "Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space"
    ]
    data_list = list(text.text_to_sequence(test, train_config.text_cleaners) for test in tests)
    
    os.makedirs("results", exist_ok=True)

    model.eval()

    log_data = []
    column_names = ["text", "step", "duration_alpha", "pitch_alpha", "energy_alpha", "audio"]

    for alpha_d in [1.0, 0.8, 1.2]:
      for alpha_p in [1.0, 0.8, 1.2]:
        for alpha_e in [1.0, 0.8, 1.2]:
          for i, phn in enumerate(data_list):
              mel_cuda = one_synthesis(model, phn, alpha_duration=alpha_d, alpha_pitch=alpha_p, alpha_energy=alpha_e)
              cur_path = f"results/d=({alpha_d})_p=({alpha_p})_e=({alpha_e})_{i}_waveglow.wav"
              waveglow.inference.inference(mel_cuda, WaveGlow, cur_path)
              cur_audio = wandb.Audio(cur_path)
              log_data.append([tests[i], logger.step, alpha_d, alpha_p, alpha_e, cur_audio])

    table = wandb.Table(data=log_data, columns=column_names)
    logger.wandb.log({"gen_audio": table}, step=logger.step)

    model.train()
