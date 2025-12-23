import os
import torch
import numpy as np
import jiwer
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Wav2Vec2Model, Wav2Vec2Processor

# Local imports
from _styletts2 import StyleTTS2
from _asr_model import AutomaticSpeechRecognitionModel
from _helper import FitnessObjective, AttackMode, addNumbersPattern


def initialize_environment(args, device):
    """
    Processes arguments, loads models, and computes reference data.
    """
    # 1. Process Enums and Thresholds
    try:
        mode = AttackMode[args.mode]
    except KeyError:
        print(f"Invalid mode '{args.mode}'. Available modes: {[m.name for m in AttackMode]}")
        return None, None

    ACTIVE_OBJECTIVES = set()
    for obj_name in args.ACTIVE_OBJECTIVES:
        try:
            ACTIVE_OBJECTIVES.add(FitnessObjective[obj_name])
        except KeyError:
            print(f"Warning: '{obj_name}' is not a valid FitnessObjective. Skipping.")

    if not ACTIVE_OBJECTIVES:
        print("Error: No valid ACTIVE_OBJECTIVES selected.")
        return None, None

    THRESHOLDS = {}
    if args.thresholds:
        for t in args.thresholds:
            try:
                key_str, val_str = t.split("=")
                obj_enum = FitnessObjective[key_str]
                THRESHOLDS[obj_enum] = float(val_str)
            except Exception as e:
                print(f"Error parsing threshold '{t}': {e}")
                return None, None

    # 3. Print Configuration
    print("=== Configuration ===")
    print(f"Mode: {mode.name}")
    print(f"GT Text: {args.ground_truth_text}")
    print(f"Target Text: {args.target_text}")
    print(f"Generations: {args.num_generations}, Pop Size: {args.pop_size}")
    print(f"Objectives: {[o.name for o in ACTIVE_OBJECTIVES]}")
    print("=====================")

    # 2. Set Constants & Configuration
    OBJECTIVE_ORDER: list[FitnessObjective] = [
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

    diffusion_steps = 5
    embedding_scale = 1
    subspace_optimization = False

    noise = torch.randn(1, 1, 256).to(device)
    random_matrix = torch.from_numpy(np.random.rand(args.size_per_phoneme, 512)).to(device).float()

    # 3. Initialize StyleTTS2 and Base Audio
    print("Loading StyleTTS2...")
    tts = StyleTTS2()
    tts.load_models()
    tts.load_checkpoints()
    tts.sample_diffusion()

    # Handle Text Embeddings
    if mode is AttackMode.TARGETED:
        tokens_gt, tokens_target = addNumbersPattern(
            tts.preprocessText(args.ground_truth_text),
            tts.preprocessText(args.target_text),
            [16, 4]
        )
        h_text_gt, h_bert_raw_gt, h_bert_gt, input_lengths, text_mask = tts.extract_embeddings(tokens_gt)
        h_text_target, h_bert_raw_target, h_bert_target, _, _ = tts.extract_embeddings(tokens_target)
    else:
        tokens_gt = tts.preprocessText(args.ground_truth_text)
        h_text_gt, h_bert_raw_gt, h_bert_gt, input_lengths, text_mask = tts.extract_embeddings(tokens_gt)

        h_text_target = torch.randn_like(h_text_gt)
        h_text_target /= h_text_target.norm()
        h_bert_raw_target = torch.randn_like(h_bert_raw_gt)
        h_bert_raw_target /= h_bert_raw_target.norm()
        h_bert_target = torch.randn_like(h_bert_gt)
        h_bert_target /= h_bert_target.norm()

    # Generate Style Vector
    style_ac, style_pro = tts.computeStyleVector(noise, h_bert_raw_gt, embedding_scale, diffusion_steps)

    # Run rest of inference for ground-truth and target
    audio_gt = tts.inference_after_interpolation(input_lengths, text_mask, h_bert_gt, h_text_gt, style_ac, style_pro)
    audio_target = tts.inference_after_interpolation(input_lengths, text_mask, h_bert_target, h_text_target, style_ac, style_pro)

    print("Loading ASR Model...")
    models = {'tts_model': tts, 'asr_model': AutomaticSpeechRecognitionModel("tiny", device=device)}
    data = {
        'mode': mode, 'ACTIVE_OBJECTIVES': ACTIVE_OBJECTIVES, 'OBJECTIVE_ORDER': OBJECTIVE_ORDER,
        'THRESHOLDS': THRESHOLDS, 'h_text_gt': h_text_gt, 'h_text_target': h_text_target,
        'h_bert_gt': h_bert_gt, 'input_lengths': input_lengths, 'text_mask': text_mask,
        'style_vector_acoustic': style_ac, 'style_vector_prosodic': style_pro,
        'random_matrix': random_matrix, 'audio_gt': audio_gt, 'audio_target': audio_target,
        'noise': noise, 'device': device
    }

    # 4. Load Conditional Models (Shortened for brevity)
    _load_conditional_assets(models, data, ACTIVE_OBJECTIVES, mode, args, device)

    return models, data


def _load_conditional_assets(models, data, ACTIVE_OBJECTIVES, mode, args, device):
    """
    Loads models and computes reference embeddings based on active objectives.
    Includes explicit print statements for each model being loaded.
    """
    text_gt = args.ground_truth_text
    text_target = args.target_text
    audio_gt = data['audio_gt']
    audio_target = data['audio_target']

    # Sentence Transformers (MPNet)
    if FitnessObjective.TEXT_EMB_TARGET in ACTIVE_OBJECTIVES or FitnessObjective.TEXT_EMB_GT in ACTIVE_OBJECTIVES:
        print("Loading SentenceTransformer (all-mpnet-base-v2)...")
        embedding_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
        embedding_model.eval()
        models['embedding_model'] = embedding_model

        text_embedding_gt = embedding_model.encode(text_gt, convert_to_tensor=True, normalize_embeddings=True)
        data['text_embedding_gt'] = text_embedding_gt

        if mode is AttackMode.TARGETED:
            text_embedding_target = embedding_model.encode(text_target, convert_to_tensor=True,
                                                           normalize_embeddings=True)
        elif mode is AttackMode.NOISE_UNTARGETED:
            text_embedding_target = torch.randn_like(text_embedding_gt)
            text_embedding_target /= text_embedding_target.norm()
        elif mode is AttackMode.UNTARGETED:
            text_embedding_target = None
        data['text_embedding_target'] = text_embedding_target

    # UTMOS
    if FitnessObjective.UTMOS in ACTIVE_OBJECTIVES:
        print("Loading UTMOS Model...")
        utmos_model = torch.jit.load(
            hf_hub_download(repo_id="balacoon/utmos", filename="utmos.jit", repo_type="model", local_dir="./"),
            map_location=device
        )
        utmos_model.eval()
        models['utmos_model'] = utmos_model

    # SBERT (MiniLM)
    if FitnessObjective.SBERT_GT in ACTIVE_OBJECTIVES or FitnessObjective.SBERT_TARGET in ACTIVE_OBJECTIVES:
        print("Loading SBERT Model (all-MiniLM-L6-v2)...")
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        sbert_model.eval()
        models['sbert_model'] = sbert_model

        s_bert_embedding_gt = sbert_model.encode(text_gt, convert_to_tensor=True, normalize_embeddings=True)
        data['s_bert_embedding_gt'] = s_bert_embedding_gt

        if mode is AttackMode.TARGETED:
            s_bert_embedding_target = sbert_model.encode(text_target, convert_to_tensor=True, normalize_embeddings=True)
        elif mode is AttackMode.NOISE_UNTARGETED:
            s_bert_embedding_target = torch.randn_like(s_bert_embedding_gt)
            s_bert_embedding_target /= s_bert_embedding_target.norm()
        elif mode is AttackMode.UNTARGETED:
            s_bert_embedding_target = None
        data['s_bert_embedding_target'] = s_bert_embedding_target

    # JIWER
    if FitnessObjective.WER_TARGET in ACTIVE_OBJECTIVES or FitnessObjective.WER_GT in ACTIVE_OBJECTIVES:
        # Note: No specific print needed for jiwer as it's a lightweight library, not a model weight file
        data['wer_transformations'] = jiwer.Compose([
            jiwer.ExpandCommonEnglishContractions(),
            jiwer.RemoveEmptyStrings(),
            jiwer.ToLowerCase(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords(),
        ])

    # GPT-2 (Perplexity)
    if FitnessObjective.PPL in ACTIVE_OBJECTIVES:
        print("Loading GPT-2 (Perplexity Model)...")
        perplexity_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
        perplexity_model.eval()
        perplexity_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        models['perplexity_model'] = perplexity_model
        models['perplexity_tokenizer'] = perplexity_tokenizer

    # Wav2Vec2
    if any(x in ACTIVE_OBJECTIVES for x in
           [FitnessObjective.WAV2VEC_SIMILAR, FitnessObjective.WAV2VEC_DIFFERENT, FitnessObjective.WAV2VEC_ASR]):
        print("Loading Wav2Vec2 Model...")
        wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
        wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h-lv60-self").to(device)
        wav2vec_model.eval()
        models['wav2vec_processor'] = wav2vec_processor
        models['wav2vec_model'] = wav2vec_model

        with torch.no_grad():
            wav2vec_embedding_gt = torch.mean(wav2vec_model(
                **wav2vec_processor(audio_gt, return_tensors="pt", sampling_rate=16000).to(device)).last_hidden_state,
                                              dim=1
                                              )
            data['wav2vec_embedding_gt'] = wav2vec_embedding_gt

            if mode is AttackMode.TARGETED:
                wav2vec_embedding_target = torch.mean(wav2vec_model(
                    **wav2vec_processor(audio_target, return_tensors="pt", sampling_rate=16000).to(
                        device)).last_hidden_state, dim=1)
            elif mode is AttackMode.NOISE_UNTARGETED:
                wav2vec_embedding_target = torch.randn_like(wav2vec_embedding_gt)
                wav2vec_embedding_target /= wav2vec_embedding_target.norm()
            elif mode is AttackMode.UNTARGETED:
                wav2vec_embedding_target = None
            data['wav2vec_embedding_target'] = wav2vec_embedding_target