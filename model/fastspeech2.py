from .modules import FFTBlock
from .variance_adaptor import LengthRegulator, VariancePredictor

import torch


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(model_config.PAD).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(model_config.PAD)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


class Encoder(nn.Module):
    def __init__(self, model_config):
        super(Encoder, self).__init__()
        
        len_max_seq=model_config.max_seq_len
        n_position = len_max_seq + 1
        n_layers = model_config.encoder_n_layer

        self.src_word_emb = nn.Embedding(
            model_config.vocab_size,
            model_config.encoder_dim,
            padding_idx=model_config.PAD
        )

        self.position_enc = nn.Embedding(
            n_position,
            model_config.encoder_dim,
            padding_idx=model_config.PAD
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config.encoder_dim,
            model_config.encoder_conv1d_filter_size,
            model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            dropout=model_config.dropout
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)
        
        # -- Forward
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        

        return enc_output, non_pad_mask


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, model_config):

        super(Decoder, self).__init__()

        len_max_seq=model_config.max_seq_len
        n_position = len_max_seq + 1
        n_layers = model_config.decoder_n_layer

        self.position_enc = nn.Embedding(
            n_position,
            model_config.encoder_dim,
            padding_idx=model_config.PAD,
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            model_config.encoder_dim,
            model_config.encoder_conv1d_filter_size,
            model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            model_config.encoder_dim // model_config.encoder_head,
            dropout=model_config.dropout
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):

        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos)
        non_pad_mask = get_non_pad_mask(enc_pos)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output


def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


class FastSpeech(nn.Module):
    """ FastSpeech """

    def __init__(self, model_config, variance_stats):
        super(FastSpeech, self).__init__()

        self.encoder = Encoder(model_config)
        self.length_regulator = LengthRegulator(model_config)
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)
        self.decoder = Decoder(model_config)

        self.mel_linear = nn.Linear(model_config.decoder_dim, mel_config.num_mels)

        # pitch quantization
        pitch_min = np.log1p(variance_stats["pitch_min"])
        pitch_max = np.log1p(variance_stats["pitch_max"])
        self.pitch_bins = nn.Parameter(torch.expm1(torch.linspace(pitch_min, pitch_max, model_config.n_bins - 1)), 
                                       requires_grad=False)
        self.pitch_embedding = nn.Embedding(model_config.n_bins, model_config.encoder_dim)
        # energy quantization
        energy_min = np.log1p(variance_stats["energy_min"])
        energy_max = np.log1p(variance_stats["energy_max"])
        self.energy_bins = nn.Parameter(torch.expm1(torch.linspace(energy_min, energy_max, model_config.n_bins - 1)), 
                                       requires_grad=False)
        self.energy_embedding = nn.Embedding(model_config.n_bins, model_config.encoder_dim)


    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def get_variance_embedding(self, values_seq, variance_name):
        if variance_name == "pitch":
          return self.pitch_embedding(torch.bucketize(values_seq, self.pitch_bins))
        elif variance_name == "energy":
          return self.energy_embedding(torch.bucketize(values_seq, self.energy_bins))
        else:
          return None

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None,
                pitch_target=None, energy_target=None, 
                alpha_duration=1.0, alpha_pitch=1.0, alpha_energy=1.0):
        x, non_pad_mask = self.encoder(src_seq, src_pos)
        
        if self.training:
          # duration
          output, log_duration_predictor_output = self.length_regulator(x, alpha_duration, length_target, mel_max_length)
          # pitch
          log_pitch_predictor_output = self.pitch_predictor(output) # only for loss
          pitch_embedding = self.get_variance_embedding(pitch_target, "pitch")
          output = output + pitch_embedding
          # energy
          log_energy_predictor_output = self.energy_predictor(output) # only for loss
          energy_embedding = self.get_variance_embedding(energy_target, "energy")
          output = output + energy_embedding
          # decoder
          output = self.decoder(output, mel_pos)
          output = self.mask_tensor(output, mel_pos, mel_max_length)
          output = self.mel_linear(output)
          return output, log_duration_predictor_output, log_pitch_predictor_output, log_energy_predictor_output
        else:
          # duration
          output, mel_pos = self.length_regulator(x, alpha_duration)
          # pitch
          log_pitch_predictor_output = self.pitch_predictor(output)
          pitch_embedding = self.get_variance_embedding(torch.expm1(log_pitch_predictor_output) * alpha_pitch, "pitch")
          output = output + pitch_embedding
          # energy
          log_energy_predictor_output = self.energy_predictor(output)
          energy_embedding = self.get_variance_embedding(torch.expm1(log_energy_predictor_output) * alpha_energy, "energy")
          output = output + energy_embedding
          # decoder
          output = self.decoder(output, mel_pos)
          output = self.mel_linear(output)
          return output
