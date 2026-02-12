import random
import numpy as np
import matplotlib.pyplot as plt

# --- Toy Universe: NKS × Game-of-Life Style Cellular Phase Automaton ---
# Cells live on a 2D lattice with Z_3 phase states {0,1,2}
# Evolution is local, rule-based, and synchronous

MOD = 3
GRID_SIZE = 40


class PhaseUniverse:
    def __init__(self, size=GRID_SIZE):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)

        # Seed: small random perturbation (early-universe fluctuation)
        cx = size // 2
        cy = size // 2
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                self.grid[cx + dx, cy + dy] = random.randint(0, MOD - 1)

    def neighbors(self, x, y):
        # Moore neighborhood with wraparound
        vals = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % self.size
                ny = (y + dy) % self.size
                vals.append(self.grid[nx, ny])
        return vals

    def step(self):
        new_grid = np.copy(self.grid)

        for x in range(self.size):
            for y in range(self.size):
                cell = self.grid[x, y]
                neigh = self.neighbors(x, y)

                # --- NKS/GOL hybrid rule ---
                # Local phase pressure drives evolution
                total = sum(neigh) % MOD
                dominant = max(set(neigh), key=neigh.count)

                if neigh.count(cell) in (2, 3):
                    # stability window (GOL-like)
                    new_grid[x, y] = cell
                else:
                    # phase drift toward local dominance
                    new_grid[x, y] = (cell + dominant + total) % MOD

        self.grid = new_grid


def visualize(universe, step_num, pause=0.5):
    plt.clf()
    plt.imshow(universe.grid, interpolation="nearest")
    plt.title(f"Phase Cellular Universe — Step {step_num}")
    plt.colorbar(label="Z3 Phase")
    plt.pause(pause)


def simulate(steps=1_000_000, pause=0.5):
    universe = PhaseUniverse()

    plt.figure(figsize=(6, 6))
    visualize(universe, 0, pause)

    for i in range(steps):
        universe.step()
        visualize(universe, i + 1, pause)


if __name__ == "__main__":
    simulate()
