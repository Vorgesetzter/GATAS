import os
import numpy as np
import soundfile as sf

from utils import levenshteinDistance, unique_wav_path
from CMUPhoneme.string_similarity import CMU_similarity
from ALINEPhoneme.string_dissimilarity import ALINE_dissimilarity
from NISQA.predict import NISQA_score
from synthesis import audio_synthesis
from google_ASR import google_ASR
from iflytek_ASR import iflytek_ASR
from whisper_ASR import whisper_ASR


class GradientEstimation:
    def __init__(self, reference_audio, reference_text, target_model, sigma, learning_rate, K):
        """
        :param sigma: Scaling factor for noise.
        :param learning_rate: Learning rate for updating the prosody vector.
        :param K: Number of noise vectors used for gradient approximation.
        """
        self.reference_audio = reference_audio
        self.reference_text = reference_text
        self.target_model = target_model
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.K = K
    def _calculate_loss(self, p_i):
        """ Calculates the loss of a given noise vector """

        l_emo_numpy = p_i.reshape(-1, 32)
        transcription = ""

        audio_numpy = audio_synthesis(l_emo_numpy, self.reference_audio, self.reference_text)
        tmp_audio_file = './SampleDir/synthesis.wav'
        
        audio_quality = NISQA_score(tmp_audio_file)

        if self.target_model == 'googleASR':
            transcription = google_ASR(tmp_audio_file)

        if self.target_model == 'iflytekASR':
            transcription = iflytek_ASR(tmp_audio_file)

        if self.target_model == 'whisperASR':
            transcription = whisper_ASR(tmp_audio_file)

        transcriped_file_name = self.target_model + '_' + transcription + '.wav'
        transcriped_file_path = unique_wav_path(os.path.join('./SampleDir', transcriped_file_name))
        sf.write(transcriped_file_path, audio_numpy, 22050)

        if transcription == 'NA':
            loss_levenshtein = 100
            loss_CMU = 0
            loss_ALINE = 10000
        else:
            # Maximize distance from reference_text (untargeted)
            loss_levenshtein = levenshteinDistance(transcription, self.reference_text) / ((len(transcription) + len(self.reference_text)) / 2)
            loss_CMU = CMU_similarity(transcription, self.reference_text)
            loss_ALINE = ALINE_dissimilarity(transcription, self.reference_text)

        # Untargeted loss: flip all directions to maximize distance from reference_text
        # loss_levenshtein: [0, 1]; loss_CMU: [0, 1]; loss_ALINE: [0, 1000]; audio_quality: [0, 5]
        loss = -10*loss_levenshtein + 0.1*loss_CMU - 0.0001*loss_ALINE - 0.05*audio_quality

        print(f'loss:{loss}, loss_levenshtein: {-10*loss_levenshtein}, loss_CMU: {0.1*loss_CMU}, loss_ALINE: {-0.0001*loss_ALINE}, audio_quality: {-0.05*audio_quality} \n')

        return loss
    
    def _estimate_gradient(self, p_i):
        """      
        :param p_i: The prosody vector at iteration i.
        :return: Estimated gradient.
        """
        gradient = 0
        for k in range(self.K):
            # Create Gaussian noise, and update the prosody vector
            u_k = np.random.normal(0, 1, size=p_i.shape)
            loss = self._calculate_loss(p_i + self.sigma * u_k)
            gradient += loss * u_k
        
        gradient = gradient / (self.sigma * self.K)
        
        return gradient
    
    def refine_prosody_vector(self, p_i, num_iterations):
        """
        Refines an initially optimized prosody vector p_i through gradient estimation.
        
        :param num_iterations: Number of iterations to run the gradient estimation.
        :return: Refined prosody vector.
        """

        for _ in range(num_iterations):
            gradient = self._estimate_gradient(p_i)
            p_i = p_i + self.learning_rate * np.sign(gradient)
            
        return p_i

# For testing purposes
if __name__ == '__main__':

    reference_audio = './Original_MyVoiceIsThePassword.wav'
    reference_text = "My voice is the password"
    # target_model can be 'googleASR' or 'iflytekASR' or 'whisperASR'
    target_model = 'whisperASR'
    # Run a small number of iterations
    gradient_iterations = 20

    # Initialize the GradientEstimation
    gradient_estimator = GradientEstimation(reference_audio, reference_text, target_model, sigma=0.1, learning_rate=0.01, K=20)

    # Initialize a prosody vector for testing
    exp_p0_tmp = np.exp(np.random.randn(8, 32) * 1)
    softmax_p0_tmp = exp_p0_tmp / np.sum(exp_p0_tmp, axis=-1, keepdims=True)
    p_0 = softmax_p0_tmp * 0.25

    p_refined = gradient_estimator.refine_prosody_vector(p_0, gradient_iterations)