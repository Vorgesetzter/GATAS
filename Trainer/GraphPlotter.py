import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.interpolate import interp1d
import numpy as np
import os
import pandas as pd

from helper import get_local_pareto_front, calculate_2d_hypervolume


class GraphPlotter:
    def __init__(self, objectives, generations, folder_path, fitness_history):
        self.objectives = objectives
        self.folder_path = folder_path
        if not fitness_history or len(fitness_history) == 0:
            print("[Log] No fitness data available to plot.")
            return
        else:
            self.fitness_history = fitness_history

        # 1. Define the Global Gradient
        self.cmap = plt.get_cmap('viridis')
        self.colors = self.cmap(np.linspace(0, 1, generations))

    def _create_gradient_line(self, ax, x_data, y_data, num_interp_points=500):
        """
        Create a line with continuous color gradient using interpolation.
        """
        if len(x_data) < 2:
            return

        # Interpolate to create smooth gradient
        x_smooth = np.linspace(x_data.min(), x_data.max(), num_interp_points)

        try:
            interpolator = interp1d(x_data, y_data, kind='linear')
            y_smooth = interpolator(x_smooth)
        except Exception:
            # Fallback if interpolation fails (e.g. flat line)
            y_smooth = np.interp(x_smooth, x_data, y_data)

        # Create segments for LineCollection
        points = np.array([x_smooth, y_smooth]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Normalize colors across the full range
        norm = plt.Normalize(x_data.min(), x_data.max())
        lc = LineCollection(segments, cmap=self.cmap, norm=norm)
        lc.set_array(x_smooth[:-1])
        lc.set_linewidth(2.5)
        ax.add_collection(lc)

        # Set axis limits
        y_min, y_max = y_data.min(), y_data.max()
        y_margin = 0.05 * (y_max - y_min + 1e-6)
        ax.set_xlim(x_data.min(), x_data.max())
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

    def generate_all_visualizations(self):
        """
        Orchestrates all plotting using self.fitness_history.
        """

        self.generate_hypervolume_graph()
        self.generate_pareto_population_graph()

        # 2. Mean Fitness Evolution
        self.generate_mean_population_graph()

        # 3. Minimal (Best) Fitness Evolution
        self.generate_minimal_population_graph()

        plt.close('all')

    def generate_pareto_population_graph(self):
        if len(self.objectives) != 2 or len(self.fitness_history) < 2: return

        # Use self.fitness_history directly
        total_gens = len(self.fitness_history)
        active_objectives = self.objectives

        # Determine which 4 generations to plot
        indices = np.linspace(0, total_gens - 1, 4, dtype=int)
        indices = np.unique(indices)

        # Setup Plot
        obj_names = [obj.name for obj in active_objectives]
        fig, ax = plt.subplots(figsize=(12, 10))

        fig.suptitle(f"Pareto Front Evolution: {obj_names[0]} vs {obj_names[1]}", fontsize=18)

        for i, idx in enumerate(indices):
            # Access raw matrix directly
            gen_data = self.fitness_history[idx]

            fit_matrix = get_local_pareto_front(gen_data)
            if fit_matrix.size == 0 or fit_matrix.shape[1] < 2: continue

            # Safe color access
            color_idx = min(int(idx), len(self.colors) - 1)
            color = self.colors[color_idx]

            # Sort by first objective
            fit_matrix = fit_matrix[fit_matrix[:, 0].argsort()]

            label_text = f"Gen {idx + 1}"

            # Plot Scatter & Line
            ax.scatter(fit_matrix[:, 0], fit_matrix[:, 1], color=color, s=80, alpha=0.9, edgecolors='white',
                       label=label_text, zorder=i + 10)
            ax.plot(fit_matrix[:, 0], fit_matrix[:, 1], color=color, linestyle='-', alpha=0.4, linewidth=2,
                    zorder=i + 5)

        ax.set_xlabel(f"{obj_names[0]} (Lower is better)")
        ax.set_ylabel(f"{obj_names[1]} (Lower is better)")
        ax.grid(True, linestyle='--', alpha=0.3)

        ax.legend(title="Evolution", loc='upper right', frameon=True)
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))
        save_path = os.path.join(self.folder_path, "pareto_evolution.png")
        plt.savefig(save_path, dpi=300)
        print("[Log] Pareto evolution graph saved as pareto_evolution.png")

    def generate_mean_population_graph(self):
        active_objectives = self.objectives

        # 1. Calculate Means manually from history
        means_list = []
        for gen_matrix in self.fitness_history:
            # Axis 0 = Mean across population (rows)
            means_list.append(np.mean(gen_matrix, axis=0))

        # 2. Convert to DataFrame
        obj_names = [obj.name for obj in active_objectives]
        df = pd.DataFrame(means_list, columns=obj_names)

        generations = np.arange(len(df), dtype=float) + 1

        # 3. Setup Plot
        num_objectives = len(active_objectives)
        fig, axs = plt.subplots(num_objectives, 1, figsize=(12, 5 * num_objectives), squeeze=False)
        fig.suptitle("Mean Fitness Evolution per Objective", fontsize=18)

        for i, obj in enumerate(active_objectives):
            ax = axs[i, 0]
            y_values = df[obj.name].values.astype(float)

            self._create_gradient_line(ax, generations, y_values)

            ax.plot([], [], color=self.colors[-1], label="Population Mean")
            ax.set_title(f"Objective: {obj.name}", fontsize=14)
            ax.set_ylabel("Fitness Score")
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.xlabel("Generation")
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))

        save_path = os.path.join(self.folder_path, "mean_fitness_stack.png")
        plt.savefig(save_path, dpi=300)
        print("[Log] Mean fitness graph saved as mean_fitness_stack.png")

    def generate_minimal_population_graph(self):
        active_objectives = self.objectives

        # 1. Calculate Mins manually from history
        mins_list = []
        for gen_matrix in self.fitness_history:
            mins_list.append(np.min(gen_matrix, axis=0))

        # 2. Convert to DataFrame
        obj_names = [obj.name for obj in active_objectives]
        df = pd.DataFrame(mins_list, columns=obj_names)

        generations = np.arange(len(df), dtype=float) + 1

        # 3. Setup Plot
        fig, axs = plt.subplots(len(active_objectives), 1, figsize=(12, 5 * len(active_objectives)), squeeze=False)
        fig.suptitle("Best (Minimal) Fitness Evolution per Objective", fontsize=18)

        for i, obj in enumerate(active_objectives):
            ax = axs[i, 0]
            # Use obj.name to access DataFrame column
            y_values = df[obj.name].values.astype(float)

            self._create_gradient_line(ax, generations, y_values)

            ax.plot([], [], color=self.colors[-1], label="Best (Min) Fitness")
            ax.set_title(f"Objective: {obj.name}", fontsize=14)
            ax.set_ylabel("Min Fitness Score")
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='upper right')

        plt.xlabel("Generation")
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))

        save_path = os.path.join(self.folder_path, "minimal_fitness_stack.png")
        plt.savefig(save_path, dpi=300)
        print("[Log] Minimal fitness graph saved as minimal_fitness_stack.png")

    def generate_hypervolume_graph(self):
        """
        Plots the Hypervolume convergence over generations.
        """
        if len(self.objectives) != 2: return

        # Define a reference point (Worst case)
        ref_point = [1.1, 1.1]

        hv_history = []
        for gen_data in self.fitness_history:
            front = get_local_pareto_front(gen_data)

            if front.shape[1] >= 2:
                hv = calculate_2d_hypervolume(front[:, :2], ref_point)
                hv_history.append(hv)

        if not hv_history: return

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(hv_history) + 1), hv_history, color='teal', linewidth=2.5)

        plt.title("Hypervolume Convergence", fontsize=16)
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Hypervolume (Area)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.fill_between(range(1, len(hv_history) + 1), hv_history, color='teal', alpha=0.1)

        plt.tight_layout()
        save_path = os.path.join(self.folder_path, "hypervolume_convergence.png")
        plt.savefig(save_path, dpi=300)
        plt.close()