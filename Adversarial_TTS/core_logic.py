import torch
import numpy as np
import re
import librosa
import jiwer
import torch.nn.functional as torch_functional
import torchaudio.functional as torchaudio_functional
from sentence_transformers import util
from pesq import pesq
from tqdm import tqdm
import whisper

from Datastructures.dataclass import FitnessData
from Datastructures.enum import AttackMode, FitnessObjective
from helper import adjustInterpolationVector

def run_optimization_generation(config_data, model_data, audio_data, embedding_data, iteration, device):
    """
    Runs the inner generation loop.
    Unpacks models and data dictionaries into original variable names
    to keep the logic and comments identical.
    """

    # ==== Extract Data ====
    tts_model = model_data.tts_model
    asr_model = model_data.asr_model
    mode = config_data.mode
    active_objectives = config_data.active_objectives
    batch_size = config_data.batch_size

    # ==== Main Optimization Loop ====
    pareto_fitness_history = []
    mean_fitness_history = []
    total_fitness_history = []
    stop_optimization = False

    progress_bar = tqdm(range(config_data.num_generations), desc=f"Current Generation {iteration + 1}", leave=False)
    gen = -1

    for gen in progress_bar:

        gen_scores: dict[FitnessObjective, list[float]] = {obj: [] for obj in active_objectives}
        interpolation_vectors_full = torch.from_numpy(model_data.optimizer.get_x_current()).to(device).float()

        for batch in range(0, config_data.pop_size, batch_size):

            # 1. Get Population
            # Shape: [Pop_Size, Vector_Dim]
            interpolation_vectors_batch = interpolation_vectors_full[batch : batch + batch_size]

            current_batch_size = interpolation_vectors_batch.size(0)

            # 2. Adjust Interpolation Vectors
            interpolation_vectors = adjustInterpolationVector(interpolation_vectors_batch, config_data.random_matrix, config_data.subspace_optimization)

            # 3. Prepare Batch Inputs (Expand/Tile Shared Data)
            input_lengths_batch = audio_data.input_lengths.expand(current_batch_size)
            text_mask_batch = audio_data.text_mask.expand(current_batch_size, -1)
            h_bert_batch = audio_data.h_bert_gt.expand(current_batch_size, -1, -1)
            style_ac_batch = audio_data.style_vector_acoustic.expand(current_batch_size, -1)
            style_pro_batch = audio_data.style_vector_prosodic.expand(current_batch_size, -1)

            # 4. Interpolate Embeddings
            #iv_expanded = interpolation_vectors.unsqueeze(-1)  # [Batch, 1, Dim] for broadcasting against [1, 512]

            if mode is AttackMode.NOISE_UNTARGETED or mode is AttackMode.TARGETED:
                # h_text_gt/target are typically [1, 512]. Broadcasting handles the batch dimension.
                h_text_mixed_batch = (1.0 - interpolation_vectors) * audio_data.h_text_gt + interpolation_vectors * audio_data.h_text_target
            else:
                # Untargeted logic
                h_text_mixed_batch = audio_data.h_text_gt + config_data.iv_scalar * interpolation_vectors

            # 5. Inference
            # Returns [Batch, Audio_Length]
            audio_mixed_batch = tts_model.inference_after_interpolation_batches(
                input_lengths_batch,
                text_mask_batch,
                h_bert_batch,
                h_text_mixed_batch,
                style_ac_batch,
                style_pro_batch,
            )


            # 2. Prepare Input Tensor
            audio_tensor = torch.from_numpy(audio_mixed_batch).squeeze(1).float().to(device)
            audio_tensor = torchaudio_functional.resample(audio_tensor, 24000, 16000)
            audio_tensor = whisper.pad_or_trim(audio_tensor)

            mel_batch = whisper.log_mel_spectrogram(audio_tensor, n_mels=asr_model.dims.n_mels).to(device)

            # decode the audio
            options = whisper.DecodingOptions()
            results = whisper.decode(asr_model, mel_batch, options)


            for i in range(current_batch_size):

                # Extract individual results
                audio_mixed = audio_mixed_batch[i]  # [Audio_Len]
                asr_text = results[i].text  # Dict
                interpolation_vector = interpolation_vectors[i]  # [Dim]

                # Initialize dictionary using Enum keys type hint
                current_ind_scores: dict[FitnessObjective, float] = {}

                # Force English characters (A-Z, a-z, spaces)
                clean_text = re.sub(r'[^a-zA-Z\s]', '', asr_text).strip()

                # Handle garbage text
                if len(clean_text) < 2:
                    val = 1.0
                    for obj in active_objectives:
                        gen_scores[obj].append(val)
                        current_ind_scores[obj] = val
                    continue

                asr_text = clean_text

                # ==== Increase Naturalness ====
                if FitnessObjective.PHONEME_COUNT in active_objectives:
                    tokens_asr = tts_model.preprocessText(asr_text)
                    n_asr = int(tokens_asr.shape[-1])
                    n_gt = int(audio_data.input_lengths.item())

                    if n_asr == 0 or n_asr > n_gt * 2:
                        # Assign penalty to all active objectives if ASR failed completely
                        val = 1.0
                    else:
                        error = abs(n_asr - n_gt) / max(1, n_gt)
                        val = float(min(1.0, error * error))

                    gen_scores[FitnessObjective.PHONEME_COUNT].append(val)
                    current_ind_scores[FitnessObjective.PHONEME_COUNT] = val

                # if FitnessObjective.AVG_LOGPROB in active_objectives:
                #
                #     # asr_logprob = mean(log(probability_token)) [average log of token_probability]
                #     # Values = usually (-3, 0), rarely < -3.0
                #     # -3 ~ log(0.05) = 5% Probability of token, 0 = 100% Probability of token
                #
                #     val = - float((asr_logprob / 3.0))
                #
                #     gen_scores[FitnessObjective.AVG_LOGPROB].append(val)
                #     current_ind_scores[FitnessObjective.AVG_LOGPROB] = val

                if FitnessObjective.UTMOS in active_objectives:

                    # predicted_mos = utmos_model(audio).item()
                    # Values: [1, 5]
                    # 1 = bad audio, 5 = perfect audio

                    audio_mos = torch.as_tensor(audio_mixed, dtype=torch.float32, device=device).unsqueeze(0)
                    predicted_mos = model_data.utmos_model(audio_mos).item()

                    val = (predicted_mos - 1.0) / 4.0
                    val = - val + 1

                    gen_scores[FitnessObjective.UTMOS].append(val)
                    current_ind_scores[FitnessObjective.UTMOS] = val

                if FitnessObjective.PPL in active_objectives:

                    # ppl_naturalness = GPT-2 perplexity: the more surprised the model is by the text,
                    # Values: usually (0, 1)
                    # 0.0 = very unnatural sentence (rare, strange, or ungrammatical), 1.0 = very natural, fluent sentence (likely to be common human language)

                    min_loss = 1.0
                    max_loss = 10.0

                    ppl_tokens = model_data.perplexity_tokenizer(asr_text, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model_data.perplexity_model(**ppl_tokens, labels=ppl_tokens["input_ids"])
                        loss = outputs.loss

                    loss_val = float(loss.item())
                    loss_clamped = max(min_loss, min(loss_val, max_loss))
                    ppl_naturalness = 1.0 - (loss_clamped - min_loss) / (max_loss - min_loss)

                    val = float(ppl_naturalness)
                    val = - val + 1.0

                    gen_scores[FitnessObjective.PPL].append(val)
                    current_ind_scores[FitnessObjective.PPL] = val

                if FitnessObjective.PESQ in active_objectives:

                    # score_pesq
                    # Values: [-0.5, 4.5]
                    # -0.5 absolute floor / unintelligible, 4.5 = Perfect audio

                    audio_gt_16khz = librosa.resample(audio_data.audio_gt, orig_sr=24000, target_sr=16000)
                    audio_mixed_16khz = librosa.resample(audio_mixed, orig_sr=24000, target_sr=16000)

                    score_pesq = pesq(16000, audio_gt_16khz, audio_mixed_16khz, 'wb')

                    val = score_pesq + 0.5
                    val /= 5.0
                    val = - val + 1.0

                    gen_scores[FitnessObjective.PESQ].append(val)
                    current_ind_scores[FitnessObjective.PESQ] = val

                # ==== Interpolation Vector Restrictions ====
                if FitnessObjective.L1 in active_objectives:

                    # L1 = mean(|IV|) [Average value of interpolation vector]
                    # Values: (0,1)
                    # 0 = only GT, 1 = only Target

                    val = float(interpolation_vector.abs().mean().item())

                    gen_scores[FitnessObjective.L1].append(val)
                    current_ind_scores[FitnessObjective.L1] = val

                if FitnessObjective.L2 in active_objectives:

                    # L2 = sqrt(mean(IV²)) [Average, but punishes larger numbers more]
                    # Values: (0,1)
                    # 0 = only GT, 1 = only Target

                    val = float((interpolation_vector ** 2).mean().sqrt().item())

                    gen_scores[FitnessObjective.L2].append(val)
                    current_ind_scores[FitnessObjective.L2] = val

                # ==== Optimize Text Towards Target ====
                if FitnessObjective.WER_TARGET in active_objectives:

                    # wer = (Substitutions + Deletions + Insertions) / Number_of_reference_words
                    # Values: usually (0, 1), rarely > 1
                    # 0 = perfect, 1 = 100% of words wrong

                    wer = jiwer.wer(
                        config_data.text_target,
                        asr_text,
                        reference_transform=model_data.wer_transformations,
                        hypothesis_transform=model_data.wer_transformations,
                    )

                    val = float(wer)
                    val = min(val, 1.0) # Clamp to (0, 1)

                    gen_scores[FitnessObjective.WER_TARGET].append(val)
                    current_ind_scores[FitnessObjective.WER_TARGET] = val

                if FitnessObjective.SBERT_TARGET in active_objectives:

                    # sbert_target = cos_sim(emb_target, emb_asr)
                    # Values: [-1, 1]
                    # -1 = ASR very different to Target, 1 = ASR same as Target

                    if mode is AttackMode.UNTARGETED:
                        raise ValueError("AttackMode.UNTARGETED incompatable with FitnessObjective.SBERT_TARGET")

                    sbert_target = util.cos_sim(
                        embedding_data.s_bert_embedding_target,
                        model_data.sbert_model.encode(asr_text, convert_to_tensor=True, normalize_embeddings=True)
                    ).item()

                    val = (sbert_target + 1) / 2.0
                    val = - val + 1
                    val = float(val)

                    gen_scores[FitnessObjective.SBERT_TARGET].append(val)
                    current_ind_scores[FitnessObjective.SBERT_TARGET] = val

                if FitnessObjective.TEXT_EMB_TARGET in active_objectives:

                    # text_dist_target = cos_sim(emb_target, emb_asr)
                    # Values: [-1, 1]
                    # -1 = ASR very different to Target, 1 = ASR same as Target

                    if mode is AttackMode.UNTARGETED:
                        raise ValueError("AttackMode.UNTARGETED incompatable with FitnessObjective.TEXT_EMB_TARGET")

                    text_dist_target = torch_functional.cosine_similarity(
                        embedding_data.text_embedding_target,
                        model_data.embedding_model.encode(asr_text, convert_to_tensor=True, normalize_embeddings=True),
                        dim=0
                    ).item()
                    val = (text_dist_target + 1) / 2.0
                    val = - val + 1
                    val = float(val)

                    gen_scores[FitnessObjective.TEXT_EMB_TARGET].append(val)
                    current_ind_scores[FitnessObjective.TEXT_EMB_TARGET] = val

                # ==== Optimize Text Away From Ground-Truth ====
                if FitnessObjective.WER_GT in active_objectives:

                    # wer = (Substitutions + Deletions + Insertions) / Number_of_reference_words
                    # Values: usually (0, 1), rarely > 1
                    # 0 = perfect, 1 = 100% of words wrong

                    if isinstance(config_data.text_gt, list) and isinstance(asr_text, list):
                        if len(config_data.text_gt) > 0 and len(asr_text) > 0:
                            tqdm.write("--- LIST STRUCTURE DEBUG ---")
                            tqdm.write(f"GT  [Len {len(config_data.text_gt)}]: {config_data.text_gt[0]}")
                            tqdm.write(f"ASR [Len {len(asr_text)}]: {asr_text[0]}")
                            # If GT Len is 1 but ASR Len is >1, you found the bug.

                    wer = jiwer.wer(
                        config_data.text_gt,
                        asr_text,
                        reference_transform=model_data.wer_transformations,
                        hypothesis_transform=model_data.wer_transformations,
                    )
                    val = float(wer)
                    val = min(val, 2.0) # Clamp to (0, 1)
                    val = -val + 2.0

                    gen_scores[FitnessObjective.WER_GT].append(val)
                    current_ind_scores[FitnessObjective.WER_GT] = val

                if FitnessObjective.SBERT_GT in active_objectives:

                    # sbert_gt = cos_sim(emb_gt, emb_asr)
                    # Values: [-1, 1]
                    # -1 = ASR very different to GT, 1 = ASR same as GT

                    sbert_gt = util.cos_sim(
                        embedding_data.s_bert_embedding_gt,
                        model_data.sbert_model.encode(asr_text, convert_to_tensor=True, normalize_embeddings=True)
                    ).item()
                    val = (sbert_gt + 1) / 2.0
                    val = float(val)

                    gen_scores[FitnessObjective.SBERT_GT].append(val)
                    current_ind_scores[FitnessObjective.SBERT_GT] = val

                if FitnessObjective.TEXT_EMB_GT in active_objectives:

                    # text_dist_gt = cos_sim(emb_gt, emb_asr)
                    # Values: [-1, 1]
                    # -1 = ASR very different to GT, 1 = ASR same as GT

                    text_dist_gt = torch_functional.cosine_similarity(
                        embedding_data.text_embedding_gt,
                        model_data.embedding_model.encode(asr_text, convert_to_tensor=True, normalize_embeddings=True),
                        dim=0
                    ).item()
                    val = (text_dist_gt + 1) / 2.0
                    val = float(val)

                    gen_scores[FitnessObjective.TEXT_EMB_GT].append(val)
                    current_ind_scores[FitnessObjective.TEXT_EMB_GT] = val

                # ==== Optimize Audio Similarity ====
                if FitnessObjective.WAV2VEC_SIMILAR in active_objectives:

                    # wav2vec_gt = cos_sim(emb_gt, emb_asr)
                    # Values = [-1, 1]
                    # -1 = ASR very different to GT, 1 = ASR same as GT

                    with torch.no_grad():
                        wav2vec_embedding_mixed = torch.mean(
                            model_data.wav2vec_model(
                                **model_data.wav2vec_processor(
                                    audio_mixed, return_tensors="pt", sampling_rate=16000
                                ).to(device)
                            ).last_hidden_state,
                            dim=1
                        )

                    wav2vec_gt = torch_functional.cosine_similarity(embedding_data.wav2vec_embedding_gt, wav2vec_embedding_mixed).item()
                    val = (wav2vec_gt + 1) / 2.0
                    val = - val + 1
                    val = float(val)

                    gen_scores[FitnessObjective.WAV2VEC_SIMILAR].append(val)
                    current_ind_scores[FitnessObjective.WAV2VEC_SIMILAR] = val

                if FitnessObjective.WAV2VEC_DIFFERENT in active_objectives:

                    # wav2vec_target = cos_sim(emb_target, emb_asr)
                    # Values = [-1, 1]
                    # -1 = ASR very different to Target, 1 = ASR same as Target

                    if mode is AttackMode.UNTARGETED:
                        raise ValueError("AttackMode.UNTARGETED incompatable with FitnessObjective.WAV2VEC_DIFFERENT")

                    with torch.no_grad():
                        wav2vec_embedding_mixed = torch.mean(
                            model_data.wav2vec_model(
                                **model_data.wav2vec_processor(
                                    audio_mixed, return_tensors="pt", sampling_rate=16000
                                ).to(device)
                            ).last_hidden_state,
                            dim=1
                        )

                    wav2vec_sim = torch_functional.cosine_similarity(embedding_data.wav2vec_embedding_gt, wav2vec_embedding_mixed).item()
                    val = (wav2vec_sim + 1) / 2.0
                    val = float(val)

                    gen_scores[FitnessObjective.WAV2VEC_DIFFERENT].append(val)
                    current_ind_scores[FitnessObjective.WAV2VEC_DIFFERENT] = val

                if FitnessObjective.WAV2VEC_ASR in active_objectives:
                    if mode is AttackMode.UNTARGETED:
                        raise ValueError("AttackMode.UNTARGETED incompatable with FitnessObjective.WAV2VEC_ASR")

                    audio_asr = tts_model.inference(asr_text, audio_data.noise)

                    with torch.no_grad():
                        wav2vec_embedding_asr = torch.mean(
                            model_data.wav2vec_model(
                                **model_data.wav2vec_processor(
                                    audio_asr, return_tensors="pt", sampling_rate=16000
                                ).to(device)
                            ).last_hidden_state,
                            dim=1
                        )

                        wav2vec_embedding_mixed = torch.mean(
                            model_data.wav2vec_model(
                                **model_data.wav2vec_processor(
                                    audio_mixed, return_tensors="pt", sampling_rate=16000
                                ).to(device)
                            ).last_hidden_state,
                            dim=1
                        )

                    wav2vec_asr = torch_functional.cosine_similarity(wav2vec_embedding_asr, wav2vec_embedding_mixed).item()
                    val = (wav2vec_asr + 1) / 2.0
                    val = - val + 1
                    val = float(val)

                    gen_scores[FitnessObjective.WAV2VEC_ASR].append(val)
                    current_ind_scores[FitnessObjective.WAV2VEC_ASR] = val

                # ==== EARLY STOPPING CHECK ====
                # Only run this logic if the user actually provided thresholds via terminal
                if config_data.thresholds:
                    meets_all_criteria = True

                    for obj in active_objectives:
                        # We only care about objectives that HAVE a threshold set
                        if obj in config_data.thresholds:
                            current_fitness = current_ind_scores[obj]
                            target_fitness = config_data.thresholds[obj]

                            # Optimization Goal: MINIMIZE Fitness.
                            # We fail if: current_fitness > target_fitness
                            if current_fitness > target_fitness:
                                meets_all_criteria = False
                                break

                    # If we survived the loop above, this individual passed all checks
                    if meets_all_criteria:
                        stop_optimization = True


        # 3. Calculate per-generation means
        gen_mean: dict[str, float] = {"Generation": gen}
        gen_total: list[np.ndarray] = []

        for obj in config_data.objective_order:
            if obj not in active_objectives:
                continue

            arr = np.array(gen_scores[obj], dtype=float)

            gen_mean[f"{obj.name}_Mean"] = float(np.mean(arr))
            gen_total.append(arr)
        total_fitness = np.column_stack(gen_total)

        # 4. Update Optimizer
        model_data.optimizer.assign_fitness(gen_total)
        model_data.optimizer.update()

        # 5. Capture Pareto Front
        current_front = np.array([c.fitness for c in model_data.optimizer.best_candidates])

        # 6. Add fitness to history
        mean_fitness_history.append(gen_mean)
        total_fitness_history.append(total_fitness)
        pareto_fitness_history.append(current_front)

        if stop_optimization:
            print(f"\n[!] Early Stopping Triggered at Generation {gen + 1} (Thresholds met).")
            break

    # ==== RETURN VARIABLES FOR MAIN ====
    return FitnessData(mean_fitness_history, pareto_fitness_history, total_fitness_history), progress_bar, stop_optimization, gen