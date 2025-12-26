from typing import Optional, Any

import jiwer
import numpy as np
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Wav2Vec2Model, Wav2Vec2Processor

# Local imports
from _styletts2 import StyleTTS2
from _asr_model import AutomaticSpeechRecognitionModel
from _helper import addNumbersPattern

# Import Pymoo components
from _pymoo_optimizer import PymooOptimizer
from pymoo.algorithms.moo.nsga2 import NSGA2


from dataclasses import dataclass
from typing import List, Set, Dict
import torch

# Import your Enums
from _helper import AttackMode, FitnessObjective
from _entities import *

def initialize_environment(args, device):

    config_data = _load_configuration(args, device)
    config_data.print_summary()

    tts_model, asr_model = _load_required_models(device)

    audio_data = _generate_audio_data(config_data, tts_model, device)

    optimizer = _load_optimizer(audio_data, config_data)

    model_data = ModelData(tts_model=tts_model, asr_model=asr_model, optimizer=optimizer)

    embedding_data = _load_conditional_assets(model_data, config_data, audio_data, device)

    return config_data, model_data, audio_data, embedding_data

def _load_configuration(args, device):

    # 1. Set Constants & Configuration
    objective_order: list[FitnessObjective] = [
        FitnessObjective.PHONEME_COUNT,
        FitnessObjective.AVG_LOGPROB,
        FitnessObjective.UTMOS,
        FitnessObjective.PPL,
        FitnessObjective.PESQ,
        FitnessObjective.L1,
        FitnessObjective.L2,
        FitnessObjective.WER_TARGET,
        FitnessObjective.SBERT_TARGET,
        FitnessObjective.TEXT_EMB_TARGET,
        FitnessObjective.WER_GT,
        FitnessObjective.SBERT_GT,
        FitnessObjective.TEXT_EMB_GT,
        FitnessObjective.WAV2VEC_SIMILAR,
        FitnessObjective.WAV2VEC_DIFFERENT,
        FitnessObjective.WAV2VEC_ASR,
    ]

    subspace_optimization = args.subspace_optimization
    random_matrix = torch.from_numpy(np.random.rand(args.size_per_phoneme, 512)).to(device).float()

    # 2. Process Enums and Thresholds
    try:
        mode = AttackMode[args.mode]
    except KeyError:
        print(f"Invalid mode '{args.mode}'. Available modes: {[m.name for m in AttackMode]}")
        return None

    active_objectives = set()
    for obj_name in args.ACTIVE_OBJECTIVES:
        try:
            active_objectives.add(FitnessObjective[obj_name])
        except KeyError:
            print(f"Warning: '{obj_name}' is not a valid FitnessObjective. Skipping.")

    if not active_objectives:
        print("Error: No valid active_objectives selected.")
        return None

    active_objectives = [obj for obj in objective_order if obj in active_objectives]

    thresholds = {}
    if args.thresholds:
        for t in args.thresholds:
            try:
                key_str, val_str = t.split("=")
                obj_enum = FitnessObjective[key_str]
                thresholds[obj_enum] = float(val_str)
            except Exception as e:
                print(f"Error parsing threshold '{t}': {e}")
                return None

    # === CREATE CONFIG OBJECT ===
    return ConfigData(
        text_gt=args.ground_truth_text,
        text_target=args.target_text,
        num_generations=args.num_generations,
        pop_size=args.pop_size,
        loop_count=args.loop_count,
        iv_scalar=args.iv_scalar,
        size_per_phoneme=args.size_per_phoneme,
        notify=args.notify,
        mode=mode,
        active_objectives=active_objectives,
        thresholds=thresholds,
        objective_order=objective_order,
        diffusion_steps=5,
        embedding_scale=1,
        subspace_optimization=subspace_optimization,
        random_matrix=random_matrix,
    )

def _load_required_models(device):
    print("Loading StyleTTS2...")
    tts = StyleTTS2()
    tts.load_models()
    tts.load_checkpoints()
    tts.sample_diffusion()

    print("Loading ASR Model...")
    asr = AutomaticSpeechRecognitionModel("tiny", device=device)

    return tts, asr

def _generate_audio_data(config, tts, device):

    noise = torch.randn(1, 1, 256).to(device)

    # Handle Text Embeddings
    if config.mode is AttackMode.TARGETED:
        # Text -> Tokens, while adding Tokens if necessary
        tokens_gt, tokens_target = addNumbersPattern(
            tts.preprocessText(config.text_gt),
            tts.preprocessText(config.text_target),
            [16, 4]
        )
        h_text_gt, h_bert_raw_gt, h_bert_gt, input_lengths, text_mask = tts.extract_embeddings(tokens_gt)
        h_text_target, h_bert_raw_target, h_bert_target, _, _ = tts.extract_embeddings(tokens_target)
    else:
        tokens_gt = tts.preprocessText(config.text_gt)
        h_text_gt, h_bert_raw_gt, h_bert_gt, input_lengths, text_mask = tts.extract_embeddings(tokens_gt)

        h_text_target = torch.randn_like(h_text_gt)
        h_text_target /= h_text_target.norm()

        h_bert_raw_target = torch.randn_like(h_bert_raw_gt)
        h_bert_raw_target /= h_bert_raw_target.norm()

        h_bert_target = torch.randn_like(h_bert_gt)
        h_bert_target /= h_bert_target.norm()

    # Generate Style Vector
    style_ac_gt, style_pro_gt = tts.computeStyleVector(noise, h_bert_raw_gt, config.embedding_scale, config.diffusion_steps)
    style_ac_target, style_pro_target = tts.computeStyleVector(noise, h_bert_raw_target, config.embedding_scale, config.diffusion_steps)

    # Run rest of inference for ground-truth and target
    audio_gt = tts.inference_after_interpolation(input_lengths, text_mask, h_bert_gt, h_text_gt, style_ac_gt, style_pro_gt)
    audio_target = tts.inference_after_interpolation(input_lengths, text_mask, h_bert_target, h_text_target, style_ac_target, style_pro_target)

    return AudioData(audio_gt, audio_target, h_text_gt, h_text_target, h_bert_raw_gt, h_bert_raw_target, h_bert_gt, h_bert_target, input_lengths, text_mask, style_ac_gt, style_pro_gt, noise)

def _load_optimizer(audio_data, config_data):
    print("Initializing Optimizer...")
    phoneme_count = audio_data.input_lengths.detach().cpu().item()

    return PymooOptimizer(
        bounds=(0, 1),
        algorithm=NSGA2,
        algo_params={"pop_size": config_data.pop_size},
        num_objectives=len(config_data.active_objectives),
        solution_shape=(phoneme_count, config_data.size_per_phoneme),
    )

def _load_conditional_assets(model_data, config_data, audio_data, device):

    active_objectives = config_data.active_objectives
    mode = config_data.mode

    embedding_data = EmbeddingData()

    # Sentence Transformers (MPNet)
    if FitnessObjective.TEXT_EMB_TARGET in active_objectives or FitnessObjective.TEXT_EMB_GT in active_objectives:
        print("Loading SentenceTransformer (all-mpnet-base-v2)...")
        embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
        embedding_model.eval()
        model_data.embedding_model = embedding_model

        text_embedding_gt = embedding_model.encode(config_data.text_gt, convert_to_tensor=True, normalize_embeddings=True)
        embedding_data.text_embedding_gt = text_embedding_gt

        if mode is AttackMode.TARGETED:
            text_embedding_target = embedding_model.encode(config_data.text_target, convert_to_tensor=True, normalize_embeddings=True)
        elif mode is AttackMode.NOISE_UNTARGETED:
            # Create random noise embedding of same dimension
            text_embedding_target = torch.randn_like(text_embedding_gt)
            text_embedding_target /= text_embedding_target.norm()
        else:
            text_embedding_target = None

        embedding_data.text_embedding_target = text_embedding_target

    # UTMOS
    if FitnessObjective.UTMOS in active_objectives:
        print("Loading UTMOS Model...")
        utmos_model = torch.jit.load(
            hf_hub_download(
                repo_id="balacoon/utmos",
                filename="utmos.jit",
                repo_type="model",
                local_dir="./"
            ),
            map_location=device
        )
        utmos_model.eval()
        model_data.utmos_model = utmos_model

    # SBERT (MiniLM)
    if FitnessObjective.SBERT_GT in active_objectives or FitnessObjective.SBERT_TARGET in active_objectives:
        print("Loading SBERT Model (all-MiniLM-L6-v2)...")
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        sbert_model.eval()
        model_data.sbert_model = sbert_model

        s_bert_embedding_gt = sbert_model.encode(config_data.text_gt, convert_to_tensor=True, normalize_embeddings=True)
        embedding_data.s_bert_embedding_gt = s_bert_embedding_gt

        if mode is AttackMode.TARGETED:
            s_bert_embedding_target = sbert_model.encode(config_data.text_target, convert_to_tensor=True, normalize_embeddings=True)

        elif mode is AttackMode.NOISE_UNTARGETED:
            # Create random noise embedding of same dimension
            s_bert_embedding_target = torch.randn_like(s_bert_embedding_gt)
            s_bert_embedding_target /= s_bert_embedding_target.norm()

        else:
            # No target direction needed
            s_bert_embedding_target = None

        embedding_data.s_bert_embedding_target = s_bert_embedding_target

    # JIWER
    if FitnessObjective.WER_TARGET in active_objectives or FitnessObjective.WER_GT in active_objectives:
        # Note: No specific print needed for jiwer as it's a lightweight library, not a model weight file
        model_data.wer_transformations = jiwer.Compose([
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveEmptyStrings(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ])

    # GPT-2 (Perplexity)
    if FitnessObjective.PPL in active_objectives:
        print("Loading GPT-2 (Perplexity Model)...")
        perplexity_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        perplexity_model.eval()
        perplexity_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        model_data.perplexity_model = perplexity_model
        model_data.perplexity_tokenizer = perplexity_tokenizer

    # Wav2Vec2
    if any(x in active_objectives for x in [FitnessObjective.WAV2VEC_SIMILAR, FitnessObjective.WAV2VEC_DIFFERENT, FitnessObjective.WAV2VEC_ASR]):
        print("Loading Wav2Vec2 Model...")
        wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(device)
        wav2vec_model.eval()

        model_data.wav2vec_processor = wav2vec_processor
        model_data.wav2vec_model = wav2vec_model

        with torch.no_grad():
            wav2vec_embedding_gt = torch.mean(
                wav2vec_model(
                    **wav2vec_processor(
                        audio_data.audio_gt,
                        return_tensors="pt",
                        sampling_rate=16000
                    ).to(device)
                ).last_hidden_state,
                dim=1
            )

            embedding_data.wav2vec_embedding_gt = wav2vec_embedding_gt

            if mode is AttackMode.TARGETED:
                wav2vec_embedding_target = torch.mean(
                    wav2vec_model(
                        **wav2vec_processor(
                            audio_data.audio_target,
                            return_tensors="pt",
                            sampling_rate=16000
                        ).to(device)
                    ).last_hidden_state,
                    dim=1
                )

            elif mode is AttackMode.NOISE_UNTARGETED:
                # Create random noise embedding of same dimension
                wav2vec_embedding_target = torch.randn_like(wav2vec_embedding_gt)
                wav2vec_embedding_target /= wav2vec_embedding_target.norm()

            else:
                # No target direction needed
                wav2vec_embedding_target = None

            embedding_data.wav2vec_embedding_target = wav2vec_embedding_target

    return embedding_data