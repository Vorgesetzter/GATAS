import torch
from torch import Tensor

import phonemizer
from phonemizer.backend import EspeakBackend

from nltk.tokenize import word_tokenize

from models import *
from utils import *
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

from helper import addNumbersPattern


class StyleTTS2:

    h_text: Tensor
    h_aligned: Tensor
    f0_pred: Tensor
    a_pred: Tensor
    n_pred: Tensor
    style_vector_prosodic: Tensor

    def __init__(self):

        # Splits words into phonemes (symbols that represent how words are pronounced)
        self.model = None
        self.params = None
        self.sampler = None

        self.global_phonemizer = phonemizer.backend.EspeakBackend(
            language='en-us',
            preserve_punctuation=True,  # Keeps Punctuation such as , . ? !
            with_stress=True  # Adds stress marks to vowels
        )

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.textcleaner = TextCleaner()  # Lowercasing & trimming, expanding numbers & symbols, handling punctuation, phoneme conversion, tokenization

    def load_models(self, yml_path="Audio/LJSpeech/config.yml"):
        config = yaml.safe_load(open(yml_path))  # YAML File with model settings and pretrained checkpoints (ASR, F0, PL-BERT)

        # load pretrained ASR (Automatic Speech Recognition) model
        ASR_config = config.get('ASR_config', False)  # YAML config that describes the model’s structure
        ASR_path = config.get('ASR_path', False)  # Checkpoint File
        text_aligner = load_ASR_models(ASR_path, ASR_config)  # Load PyTorch model

        # load pretrained F0 model (Extracts Pitch Features from Audio, How Pitch Changes over time)
        F0_path = config.get('F0_path', False)  # YAML config that describes the model’s structure
        pitch_extractor = load_F0_models(F0_path)

        # load BERT model (encodes input text with prosodic cues)
        BERT_path = config.get('PLBERT_dir', False)  # YAML config that describes the model’s structure
        plbert = load_plbert(BERT_path)

        self.model = build_model(
            recursive_munch(config['model_params']),  # Allows attribute-style access to keys of model_params,
            text_aligner,  # Automatic Speech Recognition model
            pitch_extractor,  # F0 model
            plbert  # BERT model
        )

        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(self.device) for key in self.model]

        params_whole = torch.load("Audio/LJSpeech/epoch_2nd_00100.pth", map_location='cpu', weights_only=False)
        self.params = params_whole['net']

    def load_checkpoints(self):
        for key in self.model:
            if key in self.params:
                try:
                    self.model[key].load_state_dict(self.params[key])
                except:
                    from collections import OrderedDict
                    state_dict = self.params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        #             except:
        #                 _load(params[key], model[key])
        _ = [self.model[key].eval() for key in self.model]

    def sample_diffusion(self):
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),  # empirical parameters
            clamp=False
        )

    # Turns text to tensor with token ID
    def preprocessText(self, text: str) -> Tensor:
        # 1. Preprocessing Text
        text = text.strip()  # Removes whitespaces from beginning and end of string
        text = text.replace('"', '')  # removes " to prevent unpredictable behavior

        # 2. Text -> Phoneme
        phonemes = self.global_phonemizer.phonemize([text])  # text -> list of phoneme
        phonemes = word_tokenize(phonemes[0])  # Split into individual tokens
        phonemes = ' '.join(phonemes)  # Join tokens together, split by a empty space

        # 3. Phoneme -> Token ID
        tokens = self.textcleaner(phonemes)  # Look up numeric ID per phoneme
        tokens.insert(0, 0)  # Insert leading 0 to mark start

        # 4. Token ID -> PyTorch Tensor
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)  # Converts numeric ID to PyTorch Tensor

        return tokens

    def predictDurationIterative(self, bert_encoder_with_style: Tensor, input_lengths: Tensor) -> Tensor:

        # Duration Predictor, frames per phoneme
        d_pred, _ = self.model.predictor.lstm(bert_encoder_with_style)  # Model temporal dependencies between phonemes, LSTM = RNN
        d_pred = self.model.predictor.duration_proj(d_pred)  # Predict how long each phoneme lasts

        # d_pred = torch.nan_to_num(d_pred, nan=0.0, posinf=20.0, neginf=-20.0)

        d_pred = torch.sigmoid(d_pred).sum(axis=-1)  # Sum of duration prediction -> Result: Prediction of frame duration
        d_pred = torch.round(d_pred.squeeze()).clamp(min=1)  # Convert duration prediction into integers, add clamp to ensure that each phoneme has at least one frame
        d_pred[-1] += 5  # Makes last phoneme last 5 frames longer, to ensure it not being cut off too fast

        # Creates predicted alignment matrix between text (phonemes) and audio frames
        a_pred = torch.zeros(input_lengths.item(), int(d_pred.sum().data))  # Initializes a matrix with sizes: [# of Phonemes (input_lengths)] x [Sum of total predicted frames]
        current_frame = 0
        for i in range(a_pred.size(0)):  # Iterates over phoneme
            a_pred[i, current_frame:current_frame + int(d_pred[i].data)] = 1  # Changes for row-i (the i-th phoneme) all the values from current_frame to current_frame + int(d_pred[i].data) to 1
            current_frame += int(d_pred[i].data)  # Move current_frame to new first start

        return a_pred

    import torch
    from torch import Tensor

    def predictDuration(self, bert_encoder_with_style: Tensor, input_lengths: Tensor) -> Tensor:
        # 1. Transpose Input for LSTM
        # Input: (Batch, Channels, Phonemes) -> (4, 512, 53)
        # Target: (Batch, Phonemes, Channels) -> (4, 53, 512)
        # x = bert_encoder_with_style.transpose(1, 2)

        # 2. Duration Predictor
        # d_pred shape: (Batch, Phonemes, Hidden_Size)
        d_pred, _ = self.model.predictor.lstm(bert_encoder_with_style)

        # Projection
        d_pred = self.model.predictor.duration_proj(d_pred)

        # 3. Process Durations (Preserving Batch Dim)
        # Sum over the projection dimension (axis=-1) to get scalar duration
        # Shape becomes: (Batch, Phonemes)
        d_pred = torch.sigmoid(d_pred).sum(axis=-1)

        # Round and Clamp
        # Do NOT squeeze here, or you risk losing the batch dim if batch_size=1
        d_pred = torch.round(d_pred).clamp(min=1)

        # Add padding to the last phoneme of EVERY batch item
        d_pred[:, -1] += 5

        # 4. Create Alignment Matrix
        batch_size = d_pred.size(0)
        num_phonemes = d_pred.size(1)

        # Calculate total frames for each sample in the batch
        total_durations = d_pred.sum(dim=1).long()
        # Find the max duration to set the matrix width (padding)
        max_frames = total_durations.max().item()

        # Initialize batch alignment matrix: (Batch, Phonemes, Max_Frames)
        a_pred = torch.zeros(batch_size, num_phonemes, max_frames, device=d_pred.device)

        # Fill matrix (Iterate over batch)
        for b in range(batch_size):
            current_frame = 0
            durations = d_pred[b]  # Durations for this specific sample

            # We limit the range to the actual input length of this sample
            # (Assuming input_lengths is valid, otherwise use num_phonemes)
            valid_phonemes = input_lengths[b].item() if input_lengths is not None else num_phonemes

            for i in range(valid_phonemes):
                dur = int(durations[i].item())
                # Set the ones for the duration of this phoneme
                a_pred[b, i, current_frame: current_frame + dur] = 1
                current_frame += dur

        return a_pred

    @torch.no_grad()
    def computeStyleVector(self, noise: Tensor, h_bert: Tensor, embedding_scale: int, diffusion_steps: int) -> tuple[Tensor, Tensor]:

        style_vector = self.sampler(
            noise,
            embedding=h_bert[0].unsqueeze(0),
            embedding_scale=embedding_scale,
            num_steps=diffusion_steps
        ).squeeze(0)

        # Split Style Vector
        style_vector_acoustic = style_vector[:, 128:]  # Right Half = Acoustic Style Vector
        style_vector_prosodic = style_vector[:, :128]  # Left Half = Prosodic Style Vector

        return style_vector_acoustic, style_vector_prosodic

    @torch.no_grad()
    def extract_mixed_embeddings(self, text_gt: str, text_target: str, noise: Tensor, embedding_scale=1, diffusion_steps=5):
        tokens_gt = self.preprocessText(text_gt)
        tokens_target = self.preprocessText(text_target)

        tokens_gt = addNumbersPattern(tokens_gt, tokens_target, [16, 4])
        assert tokens_gt.shape == tokens_target.shape, "Padding didn't work, ground truth and target are of different dimensions"

        h_text_gt, h_bert_raw_gt, h_bert_gt, input_lengths, text_mask = self.extract_embeddings(tokens_gt)
        h_text_target, h_bert_raw_target, h_bert_target, _, _ = self.extract_embeddings(tokens_target)

        style_vector_acoustic, style_vector_prosodic = self.computeStyleVector(noise, h_bert_raw_target, embedding_scale, diffusion_steps)

        return h_text_gt, h_bert_raw_gt, h_bert_gt, h_text_target, h_bert_raw_target, h_bert_target, input_lengths, text_mask, style_vector_acoustic, style_vector_prosodic

    @torch.no_grad()
    def extract_embeddings(self, tokens: Tensor):

        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        # Acoustic text encoder
        h_text = self.model.text_encoder(tokens, input_lengths, text_mask)

        # Prosodic BERT encoders
        h_bert_raw = self.model.bert(tokens, attention_mask=(~text_mask).int())
        h_bert = self.model.bert_encoder(h_bert_raw).transpose(-1, -2)

        return h_text, h_bert_raw, h_bert, input_lengths, text_mask

    @torch.no_grad()
    def inference_after_interpolation_iterative(self, input_lengths: Tensor, text_mask: Tensor, h_bert: Tensor, h_text: Tensor, style_vector_acoustic: Tensor, style_vector_prosodic: Tensor) -> Tensor:

        # AdaIN, Adding information of style vector to phoneme
        h_bert_with_style = self.model.predictor.text_encoder(h_bert, style_vector_acoustic, input_lengths, text_mask)

        # Function Call
        a_pred = self.predictDurationIterative( h_bert_with_style, input_lengths)

        # Multiply alignment matrix with h_text
        h_aligned = h_text @ a_pred.unsqueeze(0).to(self.device)

        # Multiply per-phoneme embedding (bert_encoder_with_style) with frame-per-phoneme matrix -> per-frame text embedding
        h_bert_with_style_per_frame = (h_bert_with_style.transpose(-1, -2) @ a_pred.unsqueeze(0).to(self.device))

        f0_pred, n_pred = self.model.predictor.F0Ntrain(h_bert_with_style_per_frame, style_vector_acoustic)

        out = self.model.decoder(
            h_aligned,
            f0_pred,
            n_pred,
            style_vector_prosodic.squeeze().unsqueeze(0)
        )

        return out.squeeze().cpu().numpy()

    @torch.no_grad()
    def inference_after_interpolation_batches(self, input_lengths: Tensor, text_mask: Tensor, h_bert: Tensor, h_text: Tensor, style_vector_acoustic: Tensor, style_vector_prosodic: Tensor) -> Tensor:

        h_bert_with_style = self.model.predictor.text_encoder(h_bert, style_vector_acoustic, input_lengths, text_mask)

        a_pred = self.predictDuration(h_bert_with_style, input_lengths)
        a_pred = a_pred.to(self.device)

        h_aligned = h_text @ a_pred

        h_bert_with_style_per_frame = h_bert_with_style.transpose(-1, -2) @ a_pred

        f0_pred, n_pred = self.model.predictor.F0Ntrain(h_bert_with_style_per_frame, style_vector_acoustic)

        out = self.model.decoder(
            h_aligned,
            f0_pred,
            n_pred,
            style_vector_prosodic
        )

        return out.squeeze(1).cpu().numpy()

    @torch.no_grad()
    def inference(self, text: str, noise: Tensor, embedding_scale=1, diffusion_steps=5):

        tokens = self.preprocessText(text)

        h_text, h_bert_raw, h_bert, input_lengths, text_mask = self.extract_embeddings(tokens)

        style_vector_acoustic, style_vector_prosodic = self.computeStyleVector(noise, h_bert_raw, embedding_scale, diffusion_steps)

        return self.inference_after_interpolation_iterative(input_lengths, text_mask, h_bert, h_text, style_vector_acoustic, style_vector_prosodic)