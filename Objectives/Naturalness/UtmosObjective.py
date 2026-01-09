import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from Objectives.base import BaseObjective
from Datastructures.dataclass import ModelData, StepContext, AudioData
from Datastructures.enum import FitnessObjective


class UtmosObjective(BaseObjective):
    objective_type = FitnessObjective.UTMOS

    def __init__(self, config, model_data: ModelData, device: str = None):
        super().__init__(config, model_data)

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # --- 1. Load Model (Lazy Loading) ---
        if self.model_data.utmos_model is None:
            print(f"[INFO] Loading UTMOS Model on {self.device}...")
            try:
                model_path = hf_hub_download(repo_id="balacoon/utmos", filename="utmos.jit", repo_type="model")
                loaded_model = torch.jit.load(model_path, map_location=self.device)

                # --- Multi-GPU Support ---
                if self.device == 'cuda' and torch.cuda.device_count() > 1:
                    print(f"[INFO] UTMOS using {torch.cuda.device_count()} GPUs.")
                    loaded_model = nn.DataParallel(loaded_model)

                loaded_model.eval()

                # Update the shared container
                self.model_data.utmos_model = loaded_model

                # FIX 2: Assign to self.model here too!
                self.model = loaded_model

            except Exception as e:
                raise RuntimeError(f"Failed to load UTMOS model: {e}")
        else:
            # Reuse existing model
            self.model = self.model_data.utmos_model

    @property
    def supports_batching(self):
        return True

    def _calculate_logic(self, context: StepContext, audio_data: AudioData):
        """
        Returns a LIST of scores (one for each item in the batch).
        """
        # Prepare Tensor
        audio_tensor = torch.as_tensor(context.audio_mixed, dtype=torch.float32, device=self.device)

        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)

        if audio_tensor.requires_grad:
            audio_tensor = audio_tensor.detach()

        # Inference (Batched)
        with torch.no_grad():
            predicted_mos = self.model(audio_tensor)

            # Ensure it is a flat vector [Batch_Size]
            if predicted_mos.dim() > 1:
                predicted_mos = predicted_mos.squeeze()

            # Handle edge case of single-item batch resulting in 0-d tensor
            if predicted_mos.dim() == 0:
                predicted_mos = predicted_mos.unsqueeze(0)

        # Values: [1, 5]
        # 1 = bad audio, 5 = perfect audio
        val_0_to_1 = (predicted_mos - 1.0) / 4.0
        fitness_scores = 1.0 - val_0_to_1
        fitness_scores = torch.clamp(fitness_scores, 0.0, 1.0)

        # 6. Return as List of Floats
        return fitness_scores.cpu().tolist()