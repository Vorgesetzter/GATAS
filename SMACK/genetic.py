import os
import random
import uuid
import copy
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


class GeneticAlgorithm():

    def __init__(self, reference_audio, reference_text, target_model, population_size):
        self.reference_audio = reference_audio
        self.reference_text = reference_text
        self.target_model = target_model
        self.population_size = population_size

    def _np_softmax(self, input):
        exp_input = np.exp(input)
        softmax = exp_input / np.sum(exp_input, axis=-1, keepdims=True)
        return softmax

    def _initialize(self):
        """ Initialize population with random strings """
        self.population = []
        for _ in range(self.population_size):
            length = random.randint(20, 640)
            # Empirical Parameter 1 vs. 100
            individual = self._np_softmax(np.random.randn(length, 32) * 1) * 0.25
            individual = individual.flatten()
            individual_id = str(uuid.uuid4())
            self.population.append((individual, individual_id))

    def _calculate_fitness(self):
        """ Calculates the fitness of each individual in the population """
        population_fitness = []
        transcription = ""
        for individual, individual_id in self.population:

            l_emo_numpy = individual.reshape(-1, 32)

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
                fitness_levenshtein = 100
                fitness_CMU = 0
                fitness_ALINE = 10000
            else:
                # Maximize distance from reference_text (untargeted)
                fitness_levenshtein = levenshteinDistance(transcription, self.reference_text) / ((len(transcription) + len(self.reference_text)) / 2)
                fitness_CMU = CMU_similarity(transcription, self.reference_text)
                fitness_ALINE = ALINE_dissimilarity(transcription, self.reference_text)

            # Untargeted fitness: flip all directions to maximize distance from reference_text
            # fitness_levenshtein: [0, 1]; fitness_CMU: [0, 1]; fitness_ALINE: [0, 1000]; audio_quality: [0, 5]
            fitness = -10*fitness_levenshtein + 0.1*fitness_CMU - 0.0001*fitness_ALINE - 0.05*audio_quality

            print(f"[Individual {individual_id} Fitness: {fitness:.2f}]")
            print(f"[Individual {individual_id} Levenshtein: {-10*fitness_levenshtein:.2f}]")
            print(f"[Individual {individual_id} CMU: {0.1*fitness_CMU:.2f}]")
            print(f"[Individual {individual_id} ALINE: {-0.0001*fitness_ALINE:.2f}]")
            print(f"[Individual {individual_id} NISQA: {0.05*audio_quality:.2f}]")
            print('\n')
            
            population_fitness.append(fitness)
            
        return population_fitness

    def _mutate(self, individual, individual_id, mutation_rate=0.5, mutation_factor=0.8):
        """ Randomly change the individual's values with probability
        mutation_rate """
        individual = copy.deepcopy(individual)
        for j in range(individual.shape[0]):
            if np.random.random() < mutation_rate:
                individual[j] = individual[j] * (1 + np.random.randn() * mutation_factor)

        return individual, individual_id

    def _crossover(self, parent1, parent2):
        """ Create children from parents by crossover """
        # Select random crossover point
        min_length = min(len(parent1[0]), len(parent2[0]))
        cross_i = np.random.randint(1, min_length)
        child1 = np.concatenate((parent1[0][:cross_i], parent2[0][cross_i:]))
        child2 = np.concatenate((parent2[0][:cross_i], parent1[0][cross_i:])) 
        
        child1_id = str(uuid.uuid4())
        child2_id = str(uuid.uuid4())

        return (child1, child1_id), (child2, child2_id)

    def _insdel(self, individual, individual_id, current_fitness_dict, former_fitness_dict, epoch):
        
        # Parameters that can be tuned
        alpha = -0.9
        beta = 1e-4
        pr = 0.05  # Percentage of original length
        c = 20  # Decay constant
        
        # Get the current and former fitness of the individual using its ID
        current_fitness = current_fitness_dict.get(individual_id, 0)
        former_fitness = former_fitness_dict.get(individual_id, 0)

        # Calculate insdel_rate
        total_fitness = sum(current_fitness_dict.values())
        insdel_rate = alpha * (current_fitness / total_fitness) + beta / (abs(current_fitness - former_fitness) + 1e-3)
        insdel_rate = 1 / (1 + np.exp(-insdel_rate))
        
        # print(f"[Individual {individual_id} Insdel Rate: {insdel_rate:.2f}] \n")

        # Perform insertion or deletion
        if np.random.random() < insdel_rate:
            
            # Calculating the number of genes to be inserted or deleted
            original_length = len(individual)
            edit_length = int(np.ceil(pr * np.exp(-epoch / c) * original_length))
            edit_length = int(np.ceil(edit_length / 32) * 32)
            
            # print(f"[Individual {individual_id} Edit Length: {edit_length}] \n")

            # Insert
            if np.random.random() < 0.5:
                # Insert genes at random positions
                for _ in range(edit_length):
                    pos = np.random.randint(0, len(individual) + 1)
                    # Estimate mean and variance of the original distribution
                    mu = np.mean(individual)
                    sigma = np.std(individual)
                    # Draw new gene value from the original distribution
                    insert_value = np.random.normal(mu, sigma)
                    individual = np.insert(individual, pos, insert_value)
            # Delete
            else:
                # Delete genes at random positions
                for _ in range(edit_length):
                    if len(individual) > 1:
                        pos = np.random.randint(0, len(individual))
                        individual = np.delete(individual, pos)

        # Return the modified individual
        return individual, individual_id
    
    
    def run(self, iterations):
        """ Run the genetic algorithm for a number of iterations """

        self._initialize()
        former_fitness_dict = {}
        epoch=0
        fittest_individual = None
        
        for epoch in range(iterations):
            # Step 1: Calculate fitnesses
            population_fitness = self._calculate_fitness()
            
            current_fitness_dict = {}
            for i, (_, individual_id) in enumerate(self.population):
                current_fitness_dict[individual_id] = population_fitness[i]
            
            # Step 2: Create a new generation by selecting parents and producing offspring through crossover and mutation
            parent_probabilities = [fitness / sum(population_fitness) for fitness in population_fitness]
            new_population = []
            
            # Step 3: Elitism selection and conduct Mutation on the best individuals
            population_elitism_rate = 0.1 # 10% of the fittest individuals will be selected
            num_elites = int(population_elitism_rate * self.population_size)
            sorted_indices = np.argsort(population_fitness)[::-1]  # Indices of individuals sorted by fitness
            for index in sorted_indices[:num_elites]:
                elite_individual, elite_individual_id = self.population[index]
                # InsDel Operation
                if epoch != 0:
                    elite_individual, elite_individual_id = self._insdel(elite_individual, elite_individual_id, current_fitness_dict, former_fitness_dict, epoch)
                # Mutation Operation
                mutated_elite, mutated_elite_id = self._mutate(elite_individual, elite_individual_id)
                new_population.append((mutated_elite, mutated_elite_id))
            
            for i in np.arange(0, self.population_size - num_elites, 2):
                # Select two parents randomly according to probabilities
                parents = random.choices(self.population, k=2, weights=parent_probabilities, cum_weights=None)
                # Perform crossover to produce offspring
                child1, child2 = self._crossover(parents[0], parents[1])
                # Save mutated offspring for next generation
                new_population.append(self._mutate(child1[0], child1[1]))
                new_population.append(self._mutate(child2[0], child2[1]))
            
            # Assign the new population to self.population
            self.population = new_population
            
            # Step 4: Print log
            fittest_individual = self.population[np.argmax(population_fitness)][0]
            highest_fitness = max(population_fitness)
            print(f"[Epoch {epoch} Closest Candidate: '{fittest_individual}', Fitness: {highest_fitness:.2f}] \n")
            
            # Step 5: Update former_fitness_dict
            former_fitness_dict = current_fitness_dict
            
        # Print the final answer
        print ("[Final Epoch %d Answer: '%s'] \n" % (epoch, fittest_individual))

        return fittest_individual
        
        
# For testing purposes
if __name__ == '__main__':

    reference_audio = './Original_MyVoiceIsThePassword.wav'
    reference_text = "My voice is the password"
    # target_model can be 'googleASR', 'iflytekASR', or 'whisperASR'
    target_model = 'whisperASR'
    # Run a small number of iterations with a small population size
    population_size = 5
    genetic_iterations = 10

    ga = GeneticAlgorithm(reference_audio, reference_text, target_model, population_size)

    # Run the Genetic Algorithm
    fittest_individual = ga.run(genetic_iterations)