import numpy as np

def resample(particles, particles_w):
    num_particles = particles.shape[0]
    state_length = particles.shape[1]

    particles_resampled = np.zeros((num_particles, state_length))
    particles_w_resampled = np.zeros(num_particles)

    r = np.random.uniform(0, 1/num_particles)
    w = particles_w[0]
    i = 0

    for m in range(num_particles):
        U = r + m * (1 / num_particles)

        while U > w:
            i = (i + 1) % num_particles
            w = w + particles_w[i]

        particles_resampled[m, :] = particles[i, :]
        particles_w_resampled[m] = particles_w[i]

    particles_w_resampled /= np.sum(particles_w_resampled)

    return particles_resampled, particles_w_resampled
