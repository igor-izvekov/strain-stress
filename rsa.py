import numpy as np
import csv
import os

SHAPE = (100, 100, 100)
FIBER_RADIUS = 3.5
FIBER_LENGTH = SHAPE[2]

RNG = np.random.default_rng()


class Fiber:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        self.direction = np.array([0.0, 0.0, 1.0])
        self.length = FIBER_LENGTH

    def check_collision(self, other):
        dx = self.center[0] - other.center[0]
        dy = self.center[1] - other.center[1]
        dist_xy = np.hypot(dx, dy)
        return dist_xy < 2.0 * self.radius


def rasterize_fibers(shape, fibers):
    rve = np.zeros(shape, dtype=np.uint8)
    nx, ny, nz = shape
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([X, Y, Z], axis=-1)

    for fiber in fibers:
        dz = points[..., 2] - fiber.center[2]
        in_z = np.abs(dz) <= fiber.length / 2.0
        dx = points[..., 0] - fiber.center[0]
        dy = points[..., 1] - fiber.center[1]
        in_xy = (dx*dx + dy*dy) <= fiber.radius * fiber.radius
        rve[in_z & in_xy] = 1
    return rve


class RSMGenerator:
    def __init__(self, shape, fiber_radius, max_iterations=20000):
        self.shape = shape
        self.fiber_radius = fiber_radius
        self.max_iterations = max_iterations
        self.fibers = []

    def generate(self, target_percent):
        if target_percent <= 0:
            return np.zeros(self.shape, dtype=np.uint8)
        if target_percent >= 100:
            return np.ones(self.shape, dtype=np.uint8)

        self.fibers = []
        margin_xy = self.fiber_radius
        center_z = self.shape[2] / 2.0

        target_vf = target_percent / 100.0
        attempts = 0

        while attempts < self.max_iterations:
            center = np.array([
                RNG.uniform(margin_xy, self.shape[0] - margin_xy),
                RNG.uniform(margin_xy, self.shape[1] - margin_xy),
                center_z
            ])
            new_fiber = Fiber(center, self.fiber_radius)

            collision = any(new_fiber.check_collision(ex) for ex in self.fibers)
            if not collision:
                self.fibers.append(new_fiber)

            if self._compute_current_volume_fraction() >= target_vf:
                break
            attempts += 1

        return rasterize_fibers(self.shape, self.fibers)

    def _compute_current_volume_fraction(self):
        total_volume = 0.0
        for _ in self.fibers:
            total_volume += np.pi * self.fiber_radius**2 * FIBER_LENGTH
        rve_volume = self.shape[0] * self.shape[1] * self.shape[2]
        return total_volume / rve_volume


def latin_hypercube(n, low=44, high=73):
    bins = np.linspace(low, high, n+1)
    samples = [RNG.uniform(bins[i], bins[i+1]) for i in range(n)]
    RNG.shuffle(samples)
    return [samples]


def generate_sample(sampling_parameters):
    rves = []
    for i in range(len(sampling_parameters[0])):
        percent = sampling_parameters[0][i]
        generator = RSMGenerator(SHAPE, FIBER_RADIUS)
        rve = generator.generate(percent)
        rves.append(rve)
    return rves


if __name__ == "__main__":
    sampling_parameters = latin_hypercube(45)
    sample = generate_sample(sampling_parameters)

    os.makedirs('rves', exist_ok=True)
    for i, rve in enumerate(sample, start=1):
        os.makedirs(f'rves/rve_{i}', exist_ok=True)
        for layer_idx in range(rve.shape[0]):
            with open(f'rves/rve_{i}/rve_{i}_layer_{layer_idx+1}.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for row in rve[layer_idx]:
                    writer.writerow(row)
