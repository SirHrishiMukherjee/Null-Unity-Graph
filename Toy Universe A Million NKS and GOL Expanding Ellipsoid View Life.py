import random
import numpy as np
import matplotlib.pyplot as plt

MOD = 3
GRID_SIZE = 40
EDGE_THRESHOLD = 3
ZOOM_PADDING = 20
HORIZONTAL_STRETCH = 2.5

LIFE_SPAWN_STEP = 37


class PhaseUniverse:

    def __init__(self, size=GRID_SIZE):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.life = np.zeros((size, size), dtype=int)
        self.step_count = 0

        cx = size // 2
        cy = size // 2
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                self.grid[cx + dx, cy + dy] = random.randint(0, MOD - 1)

    def neighbors(self, field, x, y):
        vals = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx = x + dx
                ny = y + dy

                if 0 <= nx < self.size and 0 <= ny < self.size:
                    vals.append(field[nx, ny])
                else:
                    vals.append(0)

        return vals

    def zoom_out_if_needed(self):
        active = np.argwhere((self.grid != 0) | (self.life != 0))

        if active.size == 0:
            return

        min_x, min_y = active.min(axis=0)
        max_x, max_y = active.max(axis=0)

        near_edge = (
            min_x < EDGE_THRESHOLD
            or min_y < EDGE_THRESHOLD
            or max_x > self.size - EDGE_THRESHOLD - 1
            or max_y > self.size - EDGE_THRESHOLD - 1
        )

        if near_edge:
            new_size = self.size + 2 * ZOOM_PADDING

            new_grid = np.zeros((new_size, new_size), dtype=int)
            new_life = np.zeros((new_size, new_size), dtype=int)

            offset = ZOOM_PADDING
            new_grid[offset:offset+self.size, offset:offset+self.size] = self.grid
            new_life[offset:offset+self.size, offset:offset+self.size] = self.life

            self.grid = new_grid
            self.life = new_life
            self.size = new_size

    def spawn_life(self):
        gradients = np.zeros_like(self.grid, dtype=float)

        for x in range(1, self.size-1):
            for y in range(1, self.size-1):
                neigh = self.neighbors(self.grid, x, y)
                gradients[x, y] = np.var(neigh)

        cx, cy = np.unravel_index(np.argmax(gradients), gradients.shape)

        # Spawn a visible cluster
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                x = cx + dx
                y = cy + dy
                if 0 <= x < self.size and 0 <= y < self.size:
                    self.life[x, y] = 1

    def step(self):

        if self.step_count == LIFE_SPAWN_STEP:
            self.spawn_life()

        new_grid = np.copy(self.grid)
        new_life = np.copy(self.life)

        active = np.argwhere((self.grid != 0) | (self.life != 0))

        if active.size == 0:
            return

        min_x, min_y = active.min(axis=0)
        max_x, max_y = active.max(axis=0)

        pad = 2
        min_x = max(min_x-pad, 0)
        min_y = max(min_y-pad, 0)
        max_x = min(max_x+pad, self.size-1)
        max_y = min(max_y+pad, self.size-1)

        for x in range(min_x, max_x+1):
            for y in range(min_y, max_y+1):

                cell = self.grid[x, y]
                neigh = self.neighbors(self.grid, x, y)

                total = sum(neigh) % MOD
                dominant = max(set(neigh), key=neigh.count)

                if neigh.count(cell) in (2, 3):
                    new_grid[x, y] = cell
                else:
                    new_grid[x, y] = (cell + dominant + total) % MOD

                # --- Life dynamics (stronger) ---
                life_neighbors = sum(self.neighbors(self.life, x, y))
                phase_variance = np.var(neigh)

                if self.life[x, y]:
                    # Survival: generous window
                    new_life[x, y] = 1 if 1 <= life_neighbors <= 5 else 0
                else:
                    # Birth: tension-driven
                    if life_neighbors >= 1 and phase_variance > 0.2:
                        new_life[x, y] = 1

        self.grid = new_grid
        self.life = new_life

        self.zoom_out_if_needed()
        self.step_count += 1


def elliptical_mask(grid):
    h, w = grid.shape
    y, x = np.ogrid[:h, :w]

    cx = w / 2
    cy = h / 2

    ellipse = ((x-cx)**2)/(cx**2) + ((y-cy)**2)/(cy**2)
    return ellipse <= 1


def visualize(universe, step_num, pause=0.3):

    plt.clf()

    mask = elliptical_mask(universe.grid)

    rgb = np.zeros((*universe.grid.shape, 3))

    # Phase background (blue)
    rgb[..., 2] = universe.grid / (MOD - 1)

    # Life overlay (pure red, dominant)
    life_mask = universe.life.astype(bool)
    rgb[life_mask] = [1.0, 0.0, 0.0]


    plt.imshow(rgb, origin="lower", aspect=1/HORIZONTAL_STRETCH)
    plt.title(f"Life Emergence — Step {step_num}")
    plt.axis("off")
    plt.pause(pause)


def simulate(steps=1000, pause=0.3):

    universe = PhaseUniverse()

    plt.figure(figsize=(10, 4))
    visualize(universe, 0, pause)

    for i in range(steps):
        universe.step()
        visualize(universe, i+1, pause)


if __name__ == "__main__":
    simulate()
