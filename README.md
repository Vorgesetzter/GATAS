# Creating Adversarial Examples on Black-Box ASR Models via Latent Manipulation on StyleTTS2

### Yanis Wilbrand


## Pre-requisites
1. Python >= 3.7
2. Clone this repository:
```bash
git clone https://github.com/Vorgesetzter/StyleTTS2
cd StyleTTS2
```
3. Install python requirements:
```bash
pip install -r requirements.txt
sudo apt-get install espeak-ng
```
On Windows add:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -U
```
4. Download the pretrained StyleTTS 2 on LJSpeech corpus in 24 kHz from [https://huggingface.co/yl4579/StyleTTS2-LJSpeech/tree/main](https://huggingface.co/yl4579/StyleTTS2-LJSpeech/tree/main).


## Inference

### Python Script (Recommended)
```bash
python Scripts/adversarial_tts.py [OPTIONS]
```

### Jupyter Notebooks
- Local notebook: [Scripts/adversarial_tts_classes.ipynb](Scripts/Adversarial/adversarial_tts.ipynb)
- Google Colab: [Scripts/adversarial_tts_classes_colab.ipynb](Scripts/adversarial_tts_colab_old.ipynb)


## CLI Arguments

### Text Input
| Argument | Default | Description |
|----------|---------|-------------|
| `--ground_truth_text` | "I think the NFL is lame and boring" | The ground truth text input |
| `--target_text` | "The Seattle Seahawks are the best Team in the world" | The target text for targeted attacks |

### Optimization Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--loop_count` | 1 | Number of optimization loops |
| `--num_generations` | 4 | Generations per loop |
| `--pop_size` | 4 | Population size for genetic algorithm |
| `--iv_scalar` | 0.5 | Interpolation vector scalar |
| `--size_per_phoneme` | 1 | Dimensions per phoneme in optimization |
| `--batch_size` | -1 | Batch size (-1 for full batch) |

### Flags
| Argument | Description |
|----------|-------------|
| `--notify` | Send WhatsApp notification on completion |
| `--subspace_optimization` | Enable subspace optimization for embedding vector |
| `--multi_gpu` | Enable multi-GPU support (requires multiple CUDA devices) |

### Attack Configuration
| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | TARGETED | Attack mode: `TARGETED`, `UNTARGETED`, or `NOISE_UNTARGETED` |
| `--ACTIVE_OBJECTIVES` | PESQ WER_GT | Space-separated list of objectives |
| `--thresholds` | PESQ=0.3 WER_GT=0.5 | Early stopping thresholds (format: `OBJ=value`) |


## Attack Modes

| Mode | Description |
|------|-------------|
| `TARGETED` | Optimize audio to transcribe as target text while maintaining naturalness |
| `UNTARGETED` | Optimize audio to transcribe differently from ground truth |
| `NOISE_UNTARGETED` | Add noise to make transcription differ from ground truth |


## Available Objectives

### Naturalness Objectives
| Objective | Description |
|-----------|-------------|
| `PHONEME_COUNT` | Penalize difference in phoneme count between ASR output and ground truth |
| `UTMOS` | Mean Opinion Score prediction for speech naturalness |
| `PPL` | Perplexity of ASR transcription (language model fluency) |
| `PESQ` | Perceptual Evaluation of Speech Quality |

### Interpolation Vector Constraints
| Objective | Description |
|-----------|-------------|
| `L1` | L1 norm of interpolation vector (sparsity) |
| `L2` | L2 norm of interpolation vector (magnitude) |

### Target-Oriented Objectives
| Objective | Description |
|-----------|-------------|
| `WER_TARGET` | Word Error Rate between ASR output and target text (lower = closer to target) |
| `SBERT_TARGET` | Sentence-BERT similarity to target text |
| `TEXT_EMB_TARGET` | Text embedding similarity to target |
| `WHISPER_PROB` | Probability of Whisper transcribing as target |

### Ground-Truth Divergence Objectives
| Objective | Description |
|-----------|-------------|
| `WER_GT` | Word Error Rate from ground truth (higher = more different) |
| `SBERT_GT` | Sentence-BERT dissimilarity from ground truth |
| `TEXT_EMB_GT` | Text embedding dissimilarity from ground truth |

### Audio Similarity Objectives
| Objective | Description |
|-----------|-------------|
| `WAV2VEC_SIMILAR` | Wav2Vec2 embedding similarity to original audio |
| `WAV2VEC_DIFFERENT` | Wav2Vec2 embedding dissimilarity from original |
| `WAV2VEC_ASR` | Wav2Vec2-based ASR objective |


## Example Usage

### Basic Targeted Attack
```bash
python Scripts/adversarial_tts.py \
    --ground_truth_text "Hello world" \
    --target_text "Goodbye world" \
    --mode TARGETED \
    --ACTIVE_OBJECTIVES PESQ WER_TARGET \
    --num_generations 100 \
    --pop_size 50
```

### Untargeted Attack with Early Stopping
```bash
python Scripts/adversarial_tts.py \
    --ground_truth_text "The quick brown fox" \
    --mode UNTARGETED \
    --ACTIVE_OBJECTIVES PESQ WER_GT UTMOS \
    --thresholds PESQ=0.2 WER_GT=0.6 \
    --num_generations 200
```

### Multi-Objective with Subspace Optimization
```bash
python Scripts/adversarial_tts.py \
    --ground_truth_text "Sample text" \
    --target_text "Different text" \
    --ACTIVE_OBJECTIVES PESQ WHISPER_PROB L2 \
    --subspace_optimization \
    --size_per_phoneme 8
```


## Architecture

The codebase uses a modular architecture:

- **Objectives/** - Pluggable fitness objectives with auto-registration
- **Optimizer/** - Pymoo-based multi-objective genetic algorithm (NSGA-II)
- **Trainer/** - Training loop, logging, and visualization
- **Datastructures/** - Configuration and data classes
- **Models/** - StyleTTS2 and ASR model wrappers


## Output

Results are saved to `outputs/<objectives>/<timestamp>/` containing:
- `best_candidate.wav` - Best adversarial audio
- `ground_truth.wav` - Original audio
- `target.wav` - Target audio (if applicable)
- `run_summary.txt` - Detailed run report
- `reconstruction_pack.pt` - Torch state for reproducibility
- Visualization plots (Pareto front, convergence, etc.)


---

***Before using these pre-trained models, you agree to inform the listeners that the speech samples are synthesized by the pre-trained models, unless you have the permission to use the voice you synthesize. That is, you agree to only use voices whose speakers grant the permission to have their voice cloned, either directly or by license before making synthesized voices public, or you have to publicly announce that these voices are synthesized if you do not have the permission to use these voices.***
