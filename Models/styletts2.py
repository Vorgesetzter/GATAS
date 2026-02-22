from torch import Tensor

import phonemizer
from phonemizer.backend import EspeakBackend

from nltk.tokenize import word_tokenize

from models import *
from utils import *
from text_utils import TextCleaner
from Utils.PLBERT.util import load_plbert
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

from Datastructures.dataclass import AudioEmbeddingData

class StyleTTS2:

    def __init__(self, config_path="Audio/LJSpeech/config.yml", checkpoint_path="Audio/LJSpeech/epoch_2nd_00100.pth", device=None):

        self.params = None
        self.model = None
        self.sampler = None
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')

        self.textcleaner = TextCleaner()  # Lowercasing & trimming, expanding numbers & symbols, handling punctuation, phoneme conversion, tokenization

        self.global_phonemizer = phonemizer.backend.EspeakBackend(
            language='en-us',
            preserve_punctuation=True,  # Keeps Punctuation such as , . ? !
            with_stress=True  # Adds stress marks to vowels
        )

        # We pass the paths to the loading functions
        self._load_models(config_path, checkpoint_path)
        self._load_checkpoints()
        self._sample_diffusion()

        if torch.cuda.device_count() > 1 and hasattr(self.model, 'decoder'):
            self.model.decoder = nn.DataParallel(self.model.decoder)

    def _load_models(self, config_path, checkpoint_path):
        with open(config_path) as f:
            config = yaml.safe_load(f)  # YAML File with model settings and pretrained checkpoints (ASR, F0, PL-BERT)

        # load pretrained ASR (Automatic Speech Recognition) model
        asr_config = config.get('ASR_config', False)  # YAML config that describes the model’s structure
        asr_path = config.get('ASR_path', False)  # Checkpoint File
        text_aligner = load_ASR_models(asr_path, asr_config)  # Load PyTorch model

        # load pretrained F0 model (Extracts Pitch Features from Audio, How Pitch Changes over time)
        f0_path = config.get('F0_path', False)  # YAML config that describes the model’s structure
        pitch_extractor = load_F0_models(f0_path)

        # load BERT model (encodes input text with prosodic cues)
        bert_path = config.get('PLBERT_dir', False)  # YAML config that describes the model’s structure
        plbert = load_plbert(bert_path)

        self.model = build_model(
            recursive_munch(config['model_params']),  # Allows attribute-style access to keys of model_params,
            text_aligner,  # Automatic Speech Recognition model
            pitch_extractor,  # F0 model
            plbert  # BERT model
        )

        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(self.device) for key in self.model]

        params_whole = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.params = params_whole['net']

    def _load_checkpoints(self):
        for key in self.model:
            if key in self.params:
                try:
                    self.model[key].load_state_dict(self.params[key])
                except RuntimeError:
                    from collections import OrderedDict
                    state_dict = self.params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    # load params
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        _ = [self.model[key].eval() for key in self.model]

    def _sample_diffusion(self):
        self.sampler = DiffusionSampler(
            self.model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),  # empirical parameters
            clamp=False
        )

    # Turns text to tensor with token ID
    def preprocess_text(self, text: str) -> Tensor:
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

    def _predict_duration(self, bert_encoder_with_style: Tensor, input_length: Tensor) -> Tensor:
        # 1. Duration Predictor
        # d_pred shape: (Batch, Phonemes, Hidden_Size)
        d_pred, _ = self.model.predictor.lstm(bert_encoder_with_style)

        # Projection
        d_pred = self.model.predictor.duration_proj(d_pred)

        # 32 Process Durations (Preserving Batch Dim)
        # Sum over the projection dimension (axis=-1) to get scalar duration
        # Shape becomes: (Batch, Phonemes)
        d_pred = torch.sigmoid(d_pred).sum(axis=-1)

        # Round and Clamp
        # Do NOT squeeze here, or you risk losing the batch dim if batch_size=1
        d_pred = torch.round(d_pred).clamp(min=1)

        # Add padding to the last phoneme of EVERY batch item
        d_pred[:, -1] += 5

        # 3. Create Alignment Matrix
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
            valid_phonemes = input_length[b].item() if input_length is not None else num_phonemes

            for i in range(valid_phonemes):
                dur = int(durations[i].item())
                # Set the ones for the duration of this phoneme
                a_pred[b, i, current_frame: current_frame + dur] = 1
                current_frame += dur

        return a_pred

    @torch.no_grad()
    def _compute_style_vector(self, noise: Tensor, h_bert: Tensor, embedding_scale: int, diffusion_steps: int) -> tuple[Tensor, Tensor]:

        noise = noise.expand(h_bert.shape[0], -1, -1)

        style_vector = self.sampler(
            noise,
            embedding=h_bert,
            embedding_scale=embedding_scale,
            num_steps=diffusion_steps
        ).squeeze(1)

        # Split Style Vector
        style_vector_acoustic = style_vector[:, 128:]  # Right Half = Acoustic Style Vector
        style_vector_prosodic = style_vector[:, :128]  # Left Half = Prosodic Style Vector

        return style_vector_acoustic, style_vector_prosodic

    @torch.no_grad()
    def extract_embeddings(self, tokens: Tensor, noise: Tensor, embedding_scale=1, diffusion_steps=5) -> AudioEmbeddingData:

        input_lengths = torch.LongTensor([tokens.shape[-1]] * tokens.shape[0]).to(tokens.device)
        text_mask = length_to_mask(input_lengths).to(tokens.device)

        # Acoustic text encoder
        h_text = self.model.text_encoder(tokens, input_lengths, text_mask)

        # Prosodic BERT encoders
        h_bert_raw = self.model.bert(tokens, attention_mask=(~text_mask).int())
        h_bert = self.model.bert_encoder(h_bert_raw).transpose(-1, -2)

        style_vector_acoustic, style_vector_prosodic = self._compute_style_vector(noise, h_bert_raw, embedding_scale, diffusion_steps)

        return AudioEmbeddingData(input_lengths, text_mask, h_bert, h_text, style_vector_acoustic, style_vector_prosodic)

    @torch.no_grad()
    def inference_on_embedding(self, audio_embedding_data: AudioEmbeddingData) -> Tensor:

        h_bert_with_style = self.model.predictor.text_encoder(audio_embedding_data.h_bert, audio_embedding_data.style_vector_acoustic, audio_embedding_data.input_length, audio_embedding_data.text_mask)

        a_pred = self._predict_duration(h_bert_with_style, audio_embedding_data.input_length)
        a_pred = a_pred.to(self.device)

        h_aligned = audio_embedding_data.h_text @ a_pred

        h_bert_with_style_per_frame = h_bert_with_style.transpose(-1, -2) @ a_pred

        f0_pred, n_pred = self.model.predictor.F0Ntrain(h_bert_with_style_per_frame, audio_embedding_data.style_vector_acoustic)

        out = self.model.decoder(
            h_aligned,
            f0_pred,
            n_pred,
            audio_embedding_data.style_vector_prosodic
        )

        return out.squeeze(1)

    @torch.no_grad()
    def inference(self, text: str, noise: Tensor, embedding_scale=1, diffusion_steps=5):
        return self.inference_on_token(self.preprocess_text(text), noise, embedding_scale, diffusion_steps)

    @torch.no_grad()
    def inference_on_token(self, tokens: Tensor, noise: Tensor, embedding_scale=1, diffusion_steps=5):
        return self.inference_on_embedding(self.extract_embeddings(tokens, noise, embedding_scale, diffusion_steps))
