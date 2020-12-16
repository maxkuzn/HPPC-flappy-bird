import sys
import random
from itertools import cycle

import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE

import numpy as np


FPS = 30
SCREENWIDTH  = 288.0
SCREENHEIGHT = 512.0
PIPEGAPSIZE  = 100
BASEY        = SCREENHEIGHT * 0.79


def np_sigmoid(x):
    return 1/(1+np.exp(-x))

class np_mlp(object):
    def __init__(self, n_objects = 50, in_dim = 3, mid_dim = 7, out_dim = 1):
        self.W1 = np.random.uniform(-1,1, size = (n_objects, mid_dim, in_dim))*np.sqrt(6/(in_dim+mid_dim))
#         self.b1 = np.random.uniform(-1,1, size=(n_objects, mid_dim))*np.sqrt(in_dim+mid_dim)
        self.b1 = np.zeros((n_objects, mid_dim))
        if out_dim == 1:
            self.W2 = np.random.uniform(-1,1, size=(n_objects, mid_dim))*np.sqrt(6/(mid_dim+out_dim))
#             self.b2 = np.random.uniform(-1,1, size=n_objects)*np.sqrt(mid_dim+out_dim)
            self.b2 = np.zeros(n_objects)
        else:
            raise NotImplemented

    def predict(self,x):
        y = np_sigmoid((self.W1@x[:,:,None]).T[0].T + self.b1)
        return np_sigmoid(np.sum(self.W2*y, axis = 1) + self.b2)


def init_pool(load_saved_pool):
    TOTAL_MODELS = 50

    pool = {'model': np_mlp(n_objects = TOTAL_MODELS),
            'fitness': -100*np.ones(TOTAL_MODELS),
             'len': TOTAL_MODELS}
    # Initialize all models
    if load_saved_pool:

        pool['model'].W1 = np.load('saved_model_pool_numpy/W1.npy')
        pool['model'].W2 = np.load('saved_model_pool_numpy/W2.npy')
        pool['model'].b1 = np.load('saved_model_pool_numpy/b1.npy')
        pool['model'].b2 = np.load('saved_model_pool_numpy/b2.npy')


    return pool


def save_pool(pool):
    np.save('saved_model_pool_numpy/W1.npy',pool['model'].W1)
    np.save('saved_model_pool_numpy/W2.npy',pool['model'].W2)
    np.save('saved_model_pool_numpy/b1.npy',pool['model'].b1)
    np.save('saved_model_pool_numpy/b2.npy',pool['model'].b2)
    print("Saved current pool!")


def model_crossover(pool, model_id1, model_id2):

    weights1 = pool['model'].W1[model_id1]
    weights2 = pool['model'].W1[model_id2]


    return np.asarray([weights2, weights1])


def model_mutate(weights):
    change = np.random.uniform(-0.5,0.5, weights.shape)
    cond = np.random.uniform(0,1, weights.shape)

    return np.where(cond>0.85, weights + change, weights)


def change_weights(pool, new_weights):
    for idx in range(len(new_weights)):
        pool['model'].W1[idx] = new_weights[idx]
        pool['model'].W2[idx] = model_mutate(pool['model'].W2[idx])
        pool['model'].b1[idx] = model_mutate(pool['model'].b1[idx])
        pool['model'].b2[idx] = model_mutate(pool['model'].b2[idx])


def predict_action(pool, height, dist, pipe_height):
    height = np.array(height)
    # The height, dist and pipe_height must be between 0 to 1 (Scaled by SCREENHEIGHT)
    height = np.minimum(SCREENHEIGHT, height) / SCREENHEIGHT - 0.5
    dist = dist / 450 - 0.5 # Max pipe distance from player will be 450
    pipe_height = min(SCREENHEIGHT, pipe_height) / SCREENHEIGHT - 0.5
    neural_input = np.tile([0, dist, pipe_height], (len(height),1))
    neural_input[:,0] = height
    output_prob = pool['model'].predict(neural_input)
    # Perform the jump action
    return output_prob <= 0.5

