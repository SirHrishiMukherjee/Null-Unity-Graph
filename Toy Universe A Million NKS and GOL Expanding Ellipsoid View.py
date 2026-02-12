import random
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
MOD = 3
GRID_SIZE = 40
EDGE_THRESHOLD = 3
ZOOM_PADDING = 20

HORIZONTAL_STRETCH = 2.5  # observable-universe ellipsoid ratio


class PhaseUniverse:
    def __init__(self, size=GRID_SIZE):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)

        # Seed: small central fluctuation
        cx = size // 2
        cy = size // 2
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                self.grid[cx + dx, cy + dy] = random.randint(0, MOD - 1)

    # --- Open-boundary neighbors ---
    def neighbors(self, x, y):
        vals = []

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue

                nx = x + dx
                ny = y + dy

                if 0 <= nx < self.size and 0 <= ny < self.size:
                    vals.append(self.grid[nx, ny])
                else:
                    vals.append(0)

        return vals

    # --- Zoom-out expansion ---
    def zoom_out_if_needed(self):
        active = np.argwhere(self.grid != 0)

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

            offset = ZOOM_PADDING
            new_grid[
                offset : offset + self.size,
                offset : offset + self.size,
            ] = self.grid

            self.grid = new_grid
            self.size = new_size

    # --- NKS × GOL rule ---
    def step(self):
        new_grid = np.copy(self.grid)

        for x in range(self.size):
            for y in range(self.size):
                cell = self.grid[x, y]
                neigh = self.neighbors(x, y)

                total = sum(neigh) % MOD
                dominant = max(set(neigh), key=neigh.count)

                if neigh.count(cell) in (2, 3):
                    new_grid[x, y] = cell
                else:
                    new_grid[x, y] = (cell + dominant + total) % MOD

        self.grid = new_grid
        self.zoom_out_if_needed()


# --- Elliptical viewport rendering ---
def elliptical_mask(grid):
    h, w = grid.shape
    y, x = np.ogrid[:h, :w]

    cx = w / 2
    cy = h / 2

    # stretched ellipse equation
    ellipse = ((x - cx) ** 2) / (cx**2) + ((y - cy) ** 2) / (cy**2)

    masked = np.ma.array(grid, mask=ellipse > 1)
    return masked


def visualize(universe, step_num, pause=0.5):
    plt.clf()

    masked = elliptical_mask(universe.grid)

    plt.imshow(
        masked,
        interpolation="nearest",
        origin="lower",
        aspect=1 / HORIZONTAL_STRETCH,
    )

    plt.title(f"Ellipsoidal Phase Universe — Step {step_num}")
    plt.axis("off")
    plt.pause(pause)


def simulate(steps=1_000_000, pause=0.5):
    universe = PhaseUniverse()

    plt.figure(figsize=(10, 4))  # wide cosmic window
    visualize(universe, 0, pause)

    for i in range(steps):
        universe.step()
        visualize(universe, i + 1, pause)


if __name__ == "__main__":
    simulate()
