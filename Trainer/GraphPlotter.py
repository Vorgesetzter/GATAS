import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from helper import get_local_pareto_front, calculate_2d_hypervolume

class GraphPlotter:
    def __init__(self, folder_path, active_objectives, total_generations):
        self.folder_path = folder_path
        self.objectives = active_objectives
        self.total_gens = max(1, total_generations)

        # 1. Define the Global Gradient
        self.cmap = plt.get_cmap('viridis')

        # 2. Pre-calculate colors for EVERY generation
        # This ensures Gen X is always the same color in every function
        self.colors = self.cmap(np.linspace(0, 1, total_generations))

    def generate_all_visualizations(self, fitness):

        if not fitness.total_fitness or len(fitness.total_fitness) == 0:
            print("[Log] No fitness data available to plot.")
            return

        if len(self.objectives) == 2:
            self._generate_hypervolume_graph(fitness.total_fitness)
            if len(fitness.total_fitness) >= 4:
                self._generate_pareto_population_graph(fitness.total_fitness)

        self._generate_mean_population_graph(fitness.mean_fitness)

        self._generate_minimal_population_graph(fitness.total_fitness)

        plt.close('all')

    def _generate_pareto_population_graph(self, total_fitness):
        active_objectives = self.objectives

        # Determine which 4 generations to plot
        total_gens = len(total_fitness)
        indices = np.linspace(0, total_gens - 1, 4, dtype=int)

        # Setup Plot
        obj_names = [obj.name for obj in active_objectives]
        fig, ax = plt.subplots(figsize=(12, 10))

        # Generate 4 distinct colors using a colormap (e.g., 'viridis', 'plasma', 'coolwarm')
        fig.suptitle(f"Pareto Front Evolution: {obj_names[0]} vs {obj_names[1]}", fontsize=18)

        for i, idx in enumerate(indices):
            fit_matrix = get_local_pareto_front(total_fitness[idx])
            if fit_matrix.size == 0 or fit_matrix.shape[1] < 2: continue

            color = self.colors[idx]
            # Sort by first objective so the connecting line is clean, not a web
            fit_matrix = fit_matrix[fit_matrix[:, 0].argsort()]

            # Create Label (e.g., "Gen 1 (0%)" or "Gen 50 (33%)")
            label_text = f"Gen {idx + 1} ({(idx + 1) / total_gens:.0%})"

            # Plot Scatter (Dots)
            ax.scatter(fit_matrix[:, 0], fit_matrix[:, 1], color=color, s=80, alpha=0.9, edgecolors='white', label=label_text, zorder=i + 10)

            # Plot Line (Connection)
            ax.plot(fit_matrix[:, 0], fit_matrix[:, 1], color=color, linestyle='-', alpha=0.4, linewidth=2, zorder=i + 5)

        # Final Styling
        ax.set_xlabel(f"{obj_names[0]} (Lower is better)")
        ax.set_ylabel(f"{obj_names[1]} (Lower is better)")
        ax.grid(True, linestyle='--', alpha=0.3)

        # Add a legend to explain the colors
        ax.legend(title="Evolution Progress", loc='upper right', frameon=True)
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        save_path = os.path.join(self.folder_path, "pareto_evolution.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"[Log] Pareto evolution graph saved to {save_path}")

    def _generate_mean_population_graph(self, mean_history):
        active_objectives = self.objectives

        # 1. Convert list of dicts directly to a DataFrame
        df = pd.DataFrame(mean_history)

        # Define x-axis based on the actual length of the data received
        actual_gens = len(df)
        generations = np.arange(actual_gens)

        # 2. Setup Plot
        num_objectives = len(active_objectives)
        fig, axs = plt.subplots(num_objectives, 1, figsize=(12, 5 * num_objectives), squeeze=False)
        fig.suptitle("Mean Fitness Evolution per Objective", fontsize=18)

        for i, obj in enumerate(active_objectives):
            ax = axs[i, 0]
            y_values = df[obj.name].values

            # 4. Gradient Line Logic
            for j in range(actual_gens - 1):
                # Use pre-calculated colors, fallback to last color if out of range
                color_idx = j if j < len(self.colors) else -1
                segment_color = self.colors[color_idx]

                ax.plot(generations[j:j + 2], y_values[j:j + 2],
                        color=segment_color, linewidth=2.5)

            # Styling
            ax.plot([], [], color=self.colors[-1], label="Population Mean")
            ax.set_title(f"Objective: {obj.name}", fontsize=14)
            ax.set_ylabel("Fitness Score")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.xlabel("Generation")
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))

        save_path = os.path.join(self.folder_path, "mean_fitness_stack.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Log] Mean fitness graph saved to {save_path}")

    def _generate_minimal_population_graph(self, total_fitness):
        active_objectives = self.objectives

        # 1. Extract the Minimums
        # total_fitness is [Gens, Pop_Size, Objs]
        mins_per_gen = [np.min(gen_data, axis=0) for gen_data in total_fitness]

        # 2. Convert to DataFrame for robust indexing
        # We pass the objectives as column names to make lookups easy
        df = pd.DataFrame(mins_per_gen, columns=active_objectives)

        actual_gens = len(df)
        generations = np.arange(actual_gens)

        # 3. Setup Plot
        fig, axs = plt.subplots(len(active_objectives), 1, figsize=(12, 5 * len(active_objectives)), squeeze=False)
        fig.suptitle("Best (Minimal) Fitness Evolution per Objective", fontsize=18)

        # 4. Plot each objective
        for i, obj in enumerate(active_objectives):
            ax = axs[i, 0]

            # Pull values safely (using the column name directly)
            y_values = df[obj].values

            for j in range(actual_gens - 1):
                color_idx = j if j < len(self.colors) else -1
                ax.plot(generations[j:j + 2], y_values[j:j + 2],
                        color=self.colors[color_idx], linewidth=2.5)

            ax.plot([], [], color=self.colors[-1], label="Best (Min) Fitness")
            ax.set_title(f"Objective: {obj.name}", fontsize=14)
            ax.set_ylabel("Min Fitness Score")
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper right')

        plt.xlabel("Generation")
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))

        save_path = os.path.join(self.folder_path, "minimal_fitness_stack.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[Log] Minimal fitness graph saved to {save_path}")

    def _generate_hypervolume_graph(self, total_fitness):
        """
        Plots the Hypervolume convergence over generations.
        """
        if not total_fitness: return

        # Define a reference point (Worst case)
        # For WER, 1.1 is a safe 'worse than max' limit.
        # For PESQ (if inverted to 5 - PESQ), 5.1 is a safe limit.
        # Ideally, pick values slightly larger than your thresholds.
        ref_point = [1.1, 1.1]

        hv_history = []
        for gen_data in total_fitness:
            # 1. Get only the non-dominated points for this generation
            front = get_local_pareto_front(gen_data)

            # 2. If you have more than 2 objectives, you'd need a library.
            # Here we assume the first two active objectives.
            if front.shape[1] >= 2:
                hv = calculate_2d_hypervolume(front[:, :2], ref_point)
                hv_history.append(hv)

        if not hv_history: return

        # 3. Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(hv_history)), hv_history, color='teal', linewidth=2.5)

        plt.title("Hypervolume Convergence (Overall Front Quality)", fontsize=16)
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Hypervolume (Area)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add a shaded area for visual "volume"
        plt.fill_between(range(len(hv_history)), hv_history, color='teal', alpha=0.1)

        plt.tight_layout()
        save_path = os.path.join(self.folder_path, "hypervolume_convergence.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
