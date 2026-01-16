import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from Objectives.base.BaseObjective import BaseObjective
from Datastructures.dataclass import ModelData, StepContext, ModelEmbeddingData
from Datastructures.enum import FitnessObjective


class PPLObjective(BaseObjective):
    objective_type = FitnessObjective.PPL

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Lazy Loading (Check if Model AND Tokenizer exist)
        if self.model_data.perplexity_model is None or self.model_data.perplexity_tokenizer is None:
            print(f"[INFO] Loading GPT-2 (PPL) on {self.device}...")
            try:
                model_id = "gpt2"

                # Load Tokenizer
                tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
                tokenizer.pad_token = tokenizer.eos_token

                # Load Model
                model = GPT2LMHeadModel.from_pretrained(model_id).to(self.device)
                model.eval()

                # --- Multi-GPU Support ---
                if self.device == 'cuda' and torch.cuda.device_count() > 1:
                    print(f"[INFO] PPL using {torch.cuda.device_count()} GPUs.")
                    model = nn.DataParallel(model)

                # Store in Shared Container
                self.model_data.perplexity_tokenizer = tokenizer
                self.model_data.perplexity_model = model

                # Local References
                self.tokenizer = tokenizer
                self.model = model

            except Exception as e:
                raise RuntimeError(f"Failed to load PPL model: {e}")
        else:
            # Reuse existing
            self.tokenizer = self.model_data.perplexity_tokenizer
            self.model = self.model_data.perplexity_model

    @property
    def supports_batching(self):
        return True

    def _calculate_logic(self, context: StepContext):
        # 1. Prepare Inputs
        texts = context.asr_text
        if isinstance(texts, str): texts = [texts]

        inputs = self.tokenizer(
            texts,
            return_tensors='pt',
            padding=True,  # Essential for batching
            truncation=True,
            max_length=1024
        ).to(self.device)

        # 2. Inference
        with torch.no_grad():
            outputs = self.model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
            logits = outputs.logits  # [Batch, Seq_Len, Vocab_Size]

            # 3. The "Simple" Math (No weird reshaping)
            # Goal: Predict NEXT token.
            # We compare Preds[0:-1] vs Real_Words[1:]

            shift_logits = logits[:, :-1, :]  # Cut last prediction
            shift_labels = inputs.input_ids[:, 1:]  # Cut first word
            shift_mask = inputs.attention_mask[:, 1:]

            # TRICK: Flip dimensions so PyTorch handles the batching for us
            # From: [Batch, Seq, Vocab] -> To: [Batch, Vocab, Seq]
            loss_per_token = F.cross_entropy(
                shift_logits.transpose(1, 2),
                shift_labels,
                reduction='none'  # Give me error for every single word
            )

            # 4. Average per sentence (ignoring Padding)
            # Multiply by mask (0.0 for pads) -> Sum -> Divide by real word count
            sequence_loss = (loss_per_token * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)
            ppl = torch.exp(sequence_loss)

        # 5. Normalize [0, 1] for Genetic Algorithm
        # Soft cap at PPL=100 (Anything above 100 is "Bad")
        val = (ppl - 1.0) / 100.0
        return torch.clamp(val, 0.0, 1.0).cpu().tolist()