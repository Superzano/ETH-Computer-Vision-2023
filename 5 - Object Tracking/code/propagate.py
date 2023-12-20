import numpy as np

def propagate(particles, frame_height, frame_width, params):
    num_particles = params['num_particles']
    state_length = particles.shape[1]
    sigma_p = params['sigma_position']
    sigma_v = params['sigma_velocity']
    dt = 1

    if params['model'] == 0:
        A = np.array([[1, 0],
                      [0, 1]])
        noise = np.array([sigma_p, sigma_p])

    elif params['model'] == 1:
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
        noise = np.array([sigma_p, sigma_p, sigma_v, sigma_v])

    for i in range(num_particles):
        w = noise * np.random.randn(state_length)
        particles[i, :] = A @ particles[i, :] + w

        particles[i, 0] = np.clip(particles[i, 0], 1, frame_width)
        particles[i, 1] = np.clip(particles[i, 1], 1, frame_height)

        if state_length == 4:
            particles[i, 2:] = A[2:, 2:] @ particles[i, 2:]

    return particles
