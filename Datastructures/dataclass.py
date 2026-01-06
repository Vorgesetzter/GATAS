from dataclasses import dataclass
from typing import List, Set, Dict, Optional, Any
import torch
from Datastructures.enum import AttackMode, FitnessObjective

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
    active_objectives: List[FitnessObjective]
    thresholds: Dict[FitnessObjective, float]
    objective_order: List[FitnessObjective]

    # --- Model Constants ---
    diffusion_steps: int
    embedding_scale: float

    # --- Subspace Optimization ---
    subspace_optimization: bool
    random_matrix: torch.Tensor

    def print_summary(self):
        """Prints the formatted configuration to console."""
        print("=== Configuration ===")
        print(f"Mode:              {self.mode.name}")
        print(f"GT Text:           {self.text_gt}")
        print(f"Target Text:       {self.text_target}")
        print(f"Generations:       {self.num_generations}")
        print(f"Population Size:   {self.pop_size}")
        print(f"Loop Count:        {self.loop_count}")
        print(f"IV Scalar:         {self.iv_scalar}")
        print(f"Size Per Phoneme:  {self.size_per_phoneme}")
        print(f"Notify (WhatsApp): {self.notify}")

        obj_names = [o.name for o in self.active_objectives]
        print(f"Objectives:        {obj_names}")

        if self.thresholds:
            t_list = [f"{obj.name}<={val}" for obj, val in self.thresholds.items()]
            print(f"Thresholds:        {', '.join(t_list)}")
        else:
            print("Thresholds:        None (Running full generations)")

        print("=====================")

@dataclass
class ModelData:
    # Required Audio
    tts_model: Any  # StyleTTS2
    asr_model: Any  # AutomaticSpeechRecognitionModel
    optimizer: Any  # PymooOptimizer

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
class AudioData:
    audio_gt: torch.Tensor
    audio_target: torch.Tensor

    h_text_gt: torch.Tensor
    h_text_target: torch.Tensor

    h_bert_raw_gt: torch.Tensor
    h_bert_raw_target: torch.Tensor

    h_bert_gt: torch.Tensor
    h_bert_target: torch.Tensor

    input_lengths: torch.Tensor
    text_mask: torch.Tensor

    style_vector_acoustic: torch.Tensor
    style_vector_prosodic: torch.Tensor

    noise: torch.Tensor

@dataclass
class BestMixedAudio:
    audio: torch.Tensor
    text: str
    h_text: torch.Tensor
    h_bert: torch.Tensor

@dataclass
class EmbeddingData:
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
    mean_fitness: list[float]
    pareto_fitness: list[float]
    total_fitness: list[float]
