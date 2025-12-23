import torch
import numpy as np
import re
import librosa
import jiwer
import torch.nn.functional as F
from sentence_transformers import util
from pesq import pesq
from tqdm import tqdm
from _helper import adjustInterpolationVector, AttackMode, FitnessObjective


def run_optimization_generation(optimizer, iteration, models, data, args, device):
    """
    Runs the inner generation loop.
    Unpacks models and data dictionaries into original variable names
    to keep the logic and comments identical.
    """

    # ==== UNPACK MODELS ====
    tts_model = models['tts_model']
    asr_model = models['asr_model']
    utmos_model = models.get('utmos_model')
    sbert_model = models.get('sbert_model')
    embedding_model = models.get('embedding_model')
    perplexity_model = models.get('perplexity_model')
    perplexity_tokenizer = models.get('perplexity_tokenizer')
    wav2vec_model = models.get('wav2vec_model')
    wav2vec_processor = models.get('wav2vec_processor')

    # ==== UNPACK DATA ====
    h_text_gt = data['h_text_gt']
    h_text_target = data['h_text_target']
    h_bert_gt = data['h_bert_gt']
    input_lengths = data['input_lengths']
    text_mask = data['text_mask']
    style_vector_acoustic = data['style_vector_acoustic']
    style_vector_prosodic = data['style_vector_prosodic']
    random_matrix = data['random_matrix']
    audio_gt = data['audio_gt']
    noise = data['noise']

    text_gt = args.ground_truth_text
    text_target = args.target_text

    # Conditional Data
    wer_transformations = data.get('wer_transformations')
    s_bert_embedding_target = data.get('s_bert_embedding_target')
    text_embedding_target = data.get('text_embedding_target')
    s_bert_embedding_gt = data.get('s_bert_embedding_gt')
    text_embedding_gt = data.get('text_embedding_gt')
    wav2vec_embedding_gt = data.get('wav2vec_embedding_gt')

    # ==== UNPACK ARGS & CONSTANTS ====
    mode = data['mode']
    ACTIVE_OBJECTIVES = data['ACTIVE_OBJECTIVES']
    OBJECTIVE_ORDER = data['OBJECTIVE_ORDER']
    THRESHOLDS = data['THRESHOLDS']

    num_generations = args.num_generations
    iv_scalar = args.iv_scalar
    subspace_optimization = False

    # ==== Main Optimization Loop ====
    fitness_history = []
    mean_model = []
    stop_optimization = False

    progress_bar = tqdm(range(num_generations), desc=f"Current Generation {iteration + 1}", leave=False)
    gen = -1

    for gen in progress_bar:
        gen_scores = {obj: [] for obj in ACTIVE_OBJECTIVES}
        population_vectors = optimizer.get_x_current()

        for j, interpolation_vector_np in enumerate(population_vectors):
            IV = torch.from_numpy(interpolation_vector_np).to(device).float()
            interpolation_vector = adjustInterpolationVector(IV, random_matrix, subspace_optimization)

            # Initialize dictionary using Enum keys type hint
            current_ind_scores = {}

            # Interpolate Values depending on AttackMode
            if mode is AttackMode.NOISE_UNTARGETED or mode is AttackMode.TARGETED:
                h_text_mixed = (1.0 - interpolation_vector) * h_text_gt + interpolation_vector * h_text_target
            else:
                if h_text_gt.shape != interpolation_vector.shape:
                    raise ValueError(
                        "AttackMode.UNTARGETED requires h_text_gt and interpolation_vector to be of same shape.")
                h_text_mixed = h_text_gt + iv_scalar * interpolation_vector

            h_bert_mixed = h_bert_gt

            audio_mixed = tts_model.inference_after_interpolation(
                input_lengths, text_mask, h_bert_mixed, h_text_mixed, style_vector_acoustic, style_vector_prosodic
            )

            # ASR Analysis
            asr_result, asr_logprob = asr_model.analyzeAudio(audio_mixed)
            asr_text = asr_result["text"]
            # Force English characters (A-Z, a-z, spaces)
            clean_text = re.sub(r'[^a-zA-Z\s]', '', asr_text).strip()

            # Handle garbage text
            if len(clean_text) < 2:
                val = 10.0
                for obj in ACTIVE_OBJECTIVES:
                    gen_scores[obj].append(val)
                    current_ind_scores[obj] = val
                record = {"Generation": gen, "Individual_ID": j}
                record.update(current_ind_scores)
                fitness_history.append(record)
                continue

            asr_text = clean_text

            # ==== Increase Naturalness ====
            if FitnessObjective.PHONEME_COUNT in ACTIVE_OBJECTIVES:
                tokens_asr = tts_model.preprocessText(asr_text)
                n_asr = int(tokens_asr.shape[-1])
                n_gt = int(input_lengths.item())

                if n_asr == 0 or n_asr > n_gt * 2:
                    # Assign penalty to all active objectives if ASR failed completely
                    val = 1.0
                else:
                    error = abs(n_asr - n_gt) / max(1, n_gt)
                    val = float(min(1.0, error * error))

                gen_scores[FitnessObjective.PHONEME_COUNT].append(val)
                current_ind_scores[FitnessObjective.PHONEME_COUNT] = val

            if FitnessObjective.AVG_LOGPROB in ACTIVE_OBJECTIVES:
                # asr_logprob = mean(log(probability_token)) [average log of token_probability]
                # Values = usually (-3, 0), rarely < -3.0
                # -3 ~ log(0.05) = 5% Probability of token, 0 = 100% Probability of token

                val = - float((asr_logprob / 3.0))

                gen_scores[FitnessObjective.AVG_LOGPROB].append(val)
                current_ind_scores[FitnessObjective.AVG_LOGPROB] = val

            if FitnessObjective.UTMOS in ACTIVE_OBJECTIVES:
                # predicted_mos = utmos_model(audio).item()
                # Values: [1, 5]
                # 1 = bad audio, 5 = perfect audio

                audio_mos = torch.as_tensor(audio_mixed, dtype=torch.float32, device=device).unsqueeze(0)
                predicted_mos = utmos_model(audio_mos).item()

                val = (predicted_mos - 1.0) / 4.0
                val = - val + 1

                gen_scores[FitnessObjective.UTMOS].append(val)
                current_ind_scores[FitnessObjective.UTMOS] = val

            if FitnessObjective.PPL in ACTIVE_OBJECTIVES:
                # ppl_naturalness = GPT-2 perplexity: the more surprised the model is by the text,
                # Values: usually (0, 1)
                # 0.0 = very unnatural sentence (rare, strange, or ungrammatical), 1.0 = very natural, fluent sentence (likely to be common human language)

                min_loss = 1.0
                max_loss = 10.0

                ppl_tokens = perplexity_tokenizer(asr_text, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = perplexity_model(**ppl_tokens, labels=ppl_tokens["input_ids"])
                    loss = outputs.loss

                loss_val = float(loss.item())
                loss_clamped = max(min_loss, min(loss_val, max_loss))
                ppl_naturalness = 1.0 - (loss_clamped - min_loss) / (max_loss - min_loss)

                val = float(ppl_naturalness)
                val = - val + 1.0

                gen_scores[FitnessObjective.PPL].append(val)
                current_ind_scores[FitnessObjective.PPL] = val

            if FitnessObjective.PESQ in ACTIVE_OBJECTIVES:
                # score_pesq
                # Values: [-0.5, 4.5]
                # -0.5 absolute floor / unintelligible, 4.5 = Perfect audio

                audio_gt_16khz = librosa.resample(audio_gt, orig_sr=24000, target_sr=16000)
                audio_mixed_16khz = librosa.resample(audio_mixed, orig_sr=24000, target_sr=16000)

                score_pesq = pesq(16000, audio_gt_16khz, audio_mixed_16khz, 'wb')

                val = score_pesq + 0.5
                val /= 5.0
                val = - val + 1.0

                gen_scores[FitnessObjective.PESQ].append(val)
                current_ind_scores[FitnessObjective.PESQ] = val

            # ==== Interpolation Vector Restrictions ====
            if FitnessObjective.L1 in ACTIVE_OBJECTIVES:
                # L1 = mean(|IV|) [Average value of interpolation vector]
                # Values: (0,1)
                # 0 = only GT, 1 = only Target

                val = float(interpolation_vector.abs().mean().item())

                gen_scores[FitnessObjective.L1].append(val)
                current_ind_scores[FitnessObjective.L1] = val

            if FitnessObjective.L2 in ACTIVE_OBJECTIVES:
                # L2 = sqrt(mean(IV²)) [Average, but punishes larger numbers more]
                # Values: (0,1)
                # 0 = only GT, 1 = only Target

                val = float((interpolation_vector ** 2).mean().sqrt().item())

                gen_scores[FitnessObjective.L2].append(val)
                current_ind_scores[FitnessObjective.L2] = val

            # ==== Optimize Text Towards Target ====
            if FitnessObjective.WER_TARGET in ACTIVE_OBJECTIVES:
                # wer = (Substitutions + Deletions + Insertions) / Number_of_reference_words
                # Values: usually (0, 1), rarely > 1
                # 0 = perfect, 1 = 100% of words wrong

                wer = jiwer.wer(
                    text_target,
                    asr_text,
                    reference_transform=wer_transformations,
                    hypothesis_transform=wer_transformations,
                )

                val = float(wer)

                gen_scores[FitnessObjective.WER_TARGET].append(val)
                current_ind_scores[FitnessObjective.WER_TARGET] = val

            if FitnessObjective.SBERT_TARGET in ACTIVE_OBJECTIVES:

                # sbert_target = cos_sim(emb_target, emb_asr)
                # Values: [-1, 1]
                # -1 = ASR very different to Target, 1 = ASR same as Target

                if mode is AttackMode.UNTARGETED:
                    raise ValueError("AttackMode.UNTARGETED incompatable with FitnessObjective.SBERT_TARGET")

                sbert_target = util.cos_sim(
                    s_bert_embedding_target,
                    sbert_model.encode(asr_text, convert_to_tensor=True, normalize_embeddings=True)
                ).item()

                val = (sbert_target + 1) / 2.0
                val = - val + 1
                val = float(val)

                gen_scores[FitnessObjective.SBERT_TARGET].append(val)
                current_ind_scores[FitnessObjective.SBERT_TARGET] = val

            if FitnessObjective.TEXT_EMB_TARGET in ACTIVE_OBJECTIVES:

                # text_dist_target = cos_sim(emb_target, emb_asr)
                # Values: [-1, 1]
                # -1 = ASR very different to Target, 1 = ASR same as Target

                if mode is AttackMode.UNTARGETED:
                    raise ValueError("AttackMode.UNTARGETED incompatable with FitnessObjective.TEXT_EMB_TARGET")

                text_dist_target = F.cosine_similarity(
                    text_embedding_target,
                    embedding_model.encode(asr_text, convert_to_tensor=True, normalize_embeddings=True),
                    dim=0
                ).item()
                val = (text_dist_target + 1) / 2.0
                val = - val + 1
                val = float(val)

                gen_scores[FitnessObjective.TEXT_EMB_TARGET].append(val)
                current_ind_scores[FitnessObjective.TEXT_EMB_TARGET] = val

            # ==== Optimize Text Away From Ground-Truth ====
            if FitnessObjective.WER_GT in ACTIVE_OBJECTIVES:
                # wer = (Substitutions + Deletions + Insertions) / Number_of_reference_words
                # Values: usually (0, 1), rarely > 1
                # 0 = perfect, 1 = 100% of words wrong

                wer = jiwer.wer(
                    text_gt,
                    asr_text,
                    reference_transform=wer_transformations,
                    hypothesis_transform=wer_transformations,
                )
                val = float(wer)
                val = -val + 1.0

                gen_scores[FitnessObjective.WER_GT].append(val)
                current_ind_scores[FitnessObjective.WER_GT] = val

            if FitnessObjective.SBERT_GT in ACTIVE_OBJECTIVES:
                # sbert_gt = cos_sim(emb_gt, emb_asr)
                # Values: [-1, 1]
                # -1 = ASR very different to GT, 1 = ASR same as GT

                sbert_gt = util.cos_sim(
                    s_bert_embedding_gt,
                    sbert_model.encode(asr_text, convert_to_tensor=True, normalize_embeddings=True)
                ).item()
                val = (sbert_gt + 1) / 2.0
                val = float(val)

                gen_scores[FitnessObjective.SBERT_GT].append(val)
                current_ind_scores[FitnessObjective.SBERT_GT] = val

            if FitnessObjective.TEXT_EMB_GT in ACTIVE_OBJECTIVES:
                # text_dist_gt = cos_sim(emb_gt, emb_asr)
                # Values: [-1, 1]
                # -1 = ASR very different to GT, 1 = ASR same as GT

                text_dist_gt = F.cosine_similarity(
                    text_embedding_gt,
                    embedding_model.encode(asr_text, convert_to_tensor=True, normalize_embeddings=True),
                    dim=0
                ).item()
                val = (text_dist_gt + 1) / 2.0
                val = float(val)

                gen_scores[FitnessObjective.TEXT_EMB_GT].append(val)
                current_ind_scores[FitnessObjective.TEXT_EMB_GT] = val

            # ==== Optimize Audio Similarity ====
            if FitnessObjective.WAV2VEC_SIMILAR in ACTIVE_OBJECTIVES:
                # wav2vec_gt = cos_sim(emb_gt, emb_asr)
                # Values = [-1, 1]
                # -1 = ASR very different to GT, 1 = ASR same as GT

                with torch.no_grad():
                    wav2vec_embedding_mixed = torch.mean(
                        wav2vec_model(
                            **wav2vec_processor(
                                audio_mixed, return_tensors="pt", sampling_rate=16000
                            ).to(device)
                        ).last_hidden_state,
                        dim=1
                    )

                wav2vec_gt = F.cosine_similarity(wav2vec_embedding_gt, wav2vec_embedding_mixed).item()
                val = (wav2vec_gt + 1) / 2.0
                val = - val + 1
                val = float(val)

                gen_scores[FitnessObjective.WAV2VEC_SIMILAR].append(val)
                current_ind_scores[FitnessObjective.WAV2VEC_SIMILAR] = val

            if FitnessObjective.WAV2VEC_DIFFERENT in ACTIVE_OBJECTIVES:

                # wav2vec_target = cos_sim(emb_target, emb_asr)
                # Values = [-1, 1]
                # -1 = ASR very different to Target, 1 = ASR same as Target

                if mode is AttackMode.UNTARGETED:
                    raise ValueError("AttackMode.UNTARGETED incompatable with FitnessObjective.WAV2VEC_DIFFERENT")

                with torch.no_grad():
                    wav2vec_embedding_mixed = torch.mean(
                        wav2vec_model(
                            **wav2vec_processor(
                                audio_mixed, return_tensors="pt", sampling_rate=16000
                            ).to(device)
                        ).last_hidden_state,
                        dim=1
                    )

                wav2vec_sim = F.cosine_similarity(wav2vec_embedding_gt, wav2vec_embedding_mixed).item()
                val = (wav2vec_sim + 1) / 2.0
                val = float(val)

                gen_scores[FitnessObjective.WAV2VEC_DIFFERENT].append(val)
                current_ind_scores[FitnessObjective.WAV2VEC_DIFFERENT] = val

            if FitnessObjective.WAV2VEC_ASR in ACTIVE_OBJECTIVES:
                if mode is AttackMode.UNTARGETED:
                    raise ValueError("AttackMode.UNTARGETED incompatable with FitnessObjective.WAV2VEC_ASR")

                audio_asr = tts_model.inference(asr_text, noise)

                with torch.no_grad():
                    wav2vec_embedding_asr = torch.mean(
                        wav2vec_model(
                            **wav2vec_processor(
                                audio_asr, return_tensors="pt", sampling_rate=16000
                            ).to(device)
                        ).last_hidden_state,
                        dim=1
                    )

                    wav2vec_embedding_mixed = torch.mean(
                        wav2vec_model(
                            **wav2vec_processor(
                                audio_mixed, return_tensors="pt", sampling_rate=16000
                            ).to(device)
                        ).last_hidden_state,
                        dim=1
                    )

                wav2vec_asr = F.cosine_similarity(wav2vec_embedding_asr, wav2vec_embedding_mixed).item()
                val = (wav2vec_asr + 1) / 2.0
                val = - val + 1
                val = float(val)

                gen_scores[FitnessObjective.WAV2VEC_ASR].append(val)
                current_ind_scores[FitnessObjective.WAV2VEC_ASR] = val

            # ==== EARLY STOPPING CHECK ====
            # Only run this logic if the user actually provided thresholds via terminal
            if THRESHOLDS:
                meets_all_criteria = True

                for obj in ACTIVE_OBJECTIVES:
                    # We only care about objectives that HAVE a threshold set
                    if obj in THRESHOLDS:
                        current_fitness = current_ind_scores[obj]
                        target_fitness = THRESHOLDS[obj]

                        # Optimization Goal: MINIMIZE Fitness.
                        # We fail if: current_fitness > target_fitness
                        if current_fitness > target_fitness:
                            meets_all_criteria = False
                            break

                            # If we survived the loop above, this individual passed all checks
                if meets_all_criteria:
                    stop_optimization = True

            # Store record
            record = {"Generation": gen, "Individual_ID": j}
            record.update(current_ind_scores)
            fitness_history.append(record)

        # 3. Calculate per-generation means
        gen_mean: dict[str, float] = {"Generation": gen}
        fitness_arrays_for_optimizer: list[np.ndarray] = []

        for obj in OBJECTIVE_ORDER:
            if obj not in ACTIVE_OBJECTIVES:
                continue

            arr = np.array(gen_scores[obj], dtype=float)

            gen_mean[f"{obj.name}_Mean"] = float(np.mean(arr))
            fitness_arrays_for_optimizer.append(arr)

        mean_model.append(gen_mean)

        # 4. Update Optimizer
        optimizer.assign_fitness(fitness_arrays_for_optimizer)
        optimizer.update()

        if stop_optimization:
            print(f"\n[!] Early Stopping Triggered at Generation {gen + 1} (Thresholds met).")
            break

    # ==== RETURN VARIABLES FOR MAIN ====
    return fitness_history, mean_model, progress_bar, stop_optimization, gen