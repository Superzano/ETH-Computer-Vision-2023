import numpy as np

def estimate(particles, particles_w):
    particles_w = particles_w.reshape(-1, 1)
    mean_state = np.sum(particles * particles_w, axis=0) / np.sum(particles_w)
    
    return mean_state
