import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

import time
import argparse
import librosa
import soundfile as sf
from genetic import GeneticAlgorithm
from gradient import GradientEstimation


parser = argparse.ArgumentParser(description='Run the SMACK attack.')

# Add the arguments
parser.add_argument('--audio',
                    type=str,
                    required=True,
                    help='The original speech audio path')

parser.add_argument('--model',
                    type=str,
                    required=True,
                    help='The target model can be "googleASR" or "iflytekASR" or "whisperASR" or "gmmSV" or "ivectorSV"')

parser.add_argument('--content',
                    type=str,
                    required=True,
                    help='The reference speech content in the audio.')

# Parse the arguments
args = parser.parse_args()


reference_audio = args.audio
reference_text = args.content
# target_model can be 'googleASR', 'iflytekASR', or 'whisperASR'
target_model = args.model

# Resample audio to 16 kHz if needed
audio, sr = librosa.load(reference_audio, sr=None)
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    reference_audio_16k = reference_audio.replace('.wav', '_16k.wav')
    sf.write(reference_audio_16k, audio, 16000)
    reference_audio = reference_audio_16k

population_size = 1
genetic_iterations = 1
gradient_iterations = 1

# Record the start time
start_time = time.time()

ga = GeneticAlgorithm(reference_audio, reference_text, target_model, population_size)

# Run the Genetic Algorithm
fittest_individual = ga.run(genetic_iterations)

print("The adapted genetic algorithm finished. Now launching the gradient estimation. \n")

# Initialize the GradientEstimation
gradient_estimator = GradientEstimation(reference_audio, reference_text, target_model, sigma=0.1, learning_rate=0.01, K=20)

# Run the Gradient Estimation
p_refined = gradient_estimator.refine_prosody_vector(fittest_individual, gradient_iterations)

# Record the end time
end_time = time.time()

# Calculate and display the elapsed time
elapsed_time = end_time - start_time
elapsed_hours = int(elapsed_time // 3600)
elapsed_minutes = int((elapsed_time % 3600) // 60)
elapsed_seconds = elapsed_time % 60

print(f"The adapted gradient estimation finished. Time elapsed: {elapsed_hours} hours, {elapsed_minutes} minutes, and {elapsed_seconds:.2f} seconds")