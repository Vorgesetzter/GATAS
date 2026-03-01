from dataclasses import dataclass
from typing import List, Dict, Optional, Any
import torch
from Datastructures.enum import AttackMode

@dataclass
class ConfigData:
    """
    Central storage for all static run settings.
    Pass this object instead of the 'data' dictionary.
    """
    # --- User Inputs ---
    text_gt: str
    text_target: str

    # --- Runtime Settings ---
    num_generations: int
    pop_size: int
    loop_count: int
    iv_scalar: float
    size_per_phoneme: int
    batch_size: int
    notify: bool

    # --- Logic Constants ---
    mode: AttackMode
    active_objectives: List["FitnessObjective"]
    thresholds: Dict["FitnessObjective", float]

    # --- Subspace Optimization ---
    subspace_optimization: bool
    random_matrix: torch.Tensor

    # --- Target Selection ---
    num_rms_candidates: int = 20

    def print_summary(self):
        """Prints the formatted configuration to console."""
        print("=== Configuration ===")
        print(f"GT Text:               {self.text_gt}")
        print(f"Target Text:           {self.text_target}")
        print(f"Generations:           {self.num_generations}")
        print(f"Population Size:       {self.pop_size}")
        print(f"Loop Count:            {self.loop_count}")
        print(f"IV Scalar:             {self.iv_scalar}")
        print(f"Size Per Phoneme:      {self.size_per_phoneme}")
        print(f"Batch Size:            {self.batch_size}")
        print(f"Subspace Optimization: {self.subspace_optimization}")
        print(f"Notify (WhatsApp):     {self.notify}")
        print(f"Mode:                  {self.mode.name}")

        obj_names = [obj.name for obj in self.active_objectives]
        print(f"Objectives:            {obj_names}")

        if self.thresholds:
            thresh_str = ", ".join(f"{obj.name}<={val}" for obj, val in self.thresholds.items())
            print(f"Thresholds:            {thresh_str}")
        else:
            print(f"Thresholds:            None (run all generations)")

        if torch.cuda.is_available():
            print(f"Cuda on ({torch.cuda.device_count()} GPUs)")

        print("=====================")

@dataclass
class ModelData:
    # Required Audio
    tts_model: Any  # StyleTTS2
    asr_model: Any  # AutomaticSpeechRecognitionModel

    # Conditional Audio (Default to None)
    embedding_model: Optional[Any] = None  # SentenceTransformer (MPNet)
    sbert_model: Optional[Any] = None  # SentenceTransformer (MiniLM)
    utmos_model: Optional[Any] = None  # TorchScript Module

    # JIWER Transformation Object
    wer_transformations: Optional[Any] = None

    # GPT-2 / Perplexity
    perplexity_model: Optional[Any] = None  # GPT2LMHeadModel
    perplexity_tokenizer: Optional[Any] = None  # GPT2Tokenizer

    # Wav2Vec2
    wav2vec_model: Optional[Any] = None  # Wav2Vec2Model
    wav2vec_processor: Optional[Any] = None  # Wav2Vec2Processor

@dataclass
class LimitedConfigData:
    text_gt: str
    text_target: str

    mode: AttackMode

@dataclass
class AudioEmbeddingData:
    input_length: torch.Tensor
    text_mask: torch.Tensor

    h_bert: torch.Tensor
    h_text: torch.Tensor

    style_vector_acoustic: torch.Tensor
    style_vector_prosodic: torch.Tensor

@dataclass
class AudioData:
    audio_gt: torch.Tensor
    audio_target: torch.Tensor

    audio_embedding_gt: AudioEmbeddingData
    audio_embedding_target: AudioEmbeddingData

@dataclass
class BestMixedAudio:
    audio: torch.Tensor
    text: str
    h_text: torch.Tensor
    h_bert: torch.Tensor

@dataclass
class ModelEmbeddingData:
    """
    Dynamic container for conditional reference embeddings.
    Initialize with EmbeddingData() to start empty.
    """
    # Sentence Transformer Embeddings (MPNet)
    text_embedding_gt: Optional[torch.Tensor] = None
    text_embedding_target: Optional[torch.Tensor] = None

    # SBERT Embeddings (MiniLM)
    s_bert_embedding_gt: Optional[torch.Tensor] = None
    s_bert_embedding_target: Optional[torch.Tensor] = None

    # Wav2Vec2 Embeddings
    wav2vec_embedding_gt: Optional[torch.Tensor] = None
    wav2vec_embedding_target: Optional[torch.Tensor] = None

@dataclass
class FitnessData:
    pareto_fitness: list[float]
    metric_fitness: list[float]

@dataclass
class StepContext:
    audio_mixed: torch.Tensor  # [Batch, Time]
    asr_text: list[str]
    interpolation_vector: torch.Tensor  # [Batch, Dim]
    mel_batch: Optional[torch.Tensor] = None  # [Batch, n_mels, Time] - for objectives needing mel spectrogram

    def get_item(self, index: int):
        """Helper to extract a single-item context from a batch."""
        return StepContext(
            audio_mixed=self.audio_mixed[index],
            asr_text=self.asr_text[index],
            interpolation_vector=self.interpolation_vector[index],
            mel_batch=self.mel_batch[index:index+1] if self.mel_batch is not None else None
        )

    def __len__(self):
        return len(self.asr_text)

@dataclass
class ObjectiveContext:
    """
    Simplified context for objective evaluation.
    Contains only the data needed by objectives.
    """
    audio_mixed_batch: torch.Tensor  # [Batch, Time]
    asr_texts: list[str]
    interpolation_vectors: torch.Tensor  # [Batch, Dim]
    mel_batch: Optional[torch.Tensor] = None  # [Batch, n_mels, Time]

    def get_item(self, index: int) -> "ObjectiveContext":
        """Extract a single-item context from a batch."""
        return ObjectiveContext(
            audio_mixed_batch=self.audio_mixed_batch[index],
            asr_texts=self.asr_texts[index],
            interpolation_vectors=self.interpolation_vectors[index],
            mel_batch=self.mel_batch[index:index+1] if self.mel_batch is not None else None
        )

    def __len__(self) -> int:
        return len(self.asr_texts)
