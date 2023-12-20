import numpy as np
from chi2_cost import chi2_cost
from color_histogram import color_histogram

def observe(particles, frame, bbox_height, bbox_width, hist_bin, hist_target, sigma_observe):
    num_particles = particles.shape[0]
    particles_w = np.zeros(num_particles)

    x_min = particles[:, 0] - bbox_width / 2
    x_max = particles[:, 0] + bbox_width / 2
    y_min = particles[:, 1] - bbox_height / 2
    y_max = particles[:, 1] + bbox_height / 2

    for i in range(num_particles):
        x_min[i] = max(1, min(frame.shape[1], x_min[i]))
        x_max[i] = max(1, min(frame.shape[1], x_max[i]))
        y_min[i] = max(1, min(frame.shape[0], y_min[i]))
        y_max[i] = max(1, min(frame.shape[0], y_max[i]))

        hist_i = color_histogram(x_min[i], y_min[i], x_max[i], y_max[i], frame, hist_bin)

        chi_dist = chi2_cost(hist_target, hist_i)

        particles_w[i] = 1 / (np.sqrt(2 * np.pi) * sigma_observe) * np.exp(-0.5 * chi_dist**2 / (sigma_observe**2))

    particles_w /= np.sum(particles_w)

    return particles_w