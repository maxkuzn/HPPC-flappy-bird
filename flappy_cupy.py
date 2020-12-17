import sys
import random
from itertools import cycle

import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE

import cupy as cp


FPS = 30
SCREENWIDTH  = 288.0
SCREENHEIGHT = 512.0
PIPEGAPSIZE  = 100
BASEY        = SCREENHEIGHT * 0.79


def cp_sigmoid(x):
    return 1/(1+cp.exp(-x))

class cp_mlp(object):
    def __init__(self, n_objects = 50, in_dim = 3, mid_dim = 7, out_dim = 1):
        self.W1 = cp.random.uniform(-1,1, size = (n_objects, mid_dim, in_dim))*cp.sqrt(6/(in_dim+mid_dim))
#         self.b1 = cp.random.uniform(-1,1, size=(n_objects, mid_dim))*cp.sqrt(in_dim+mid_dim)
        self.b1 = cp.zeros((n_objects, mid_dim))
        if out_dim == 1:
            self.W2 = cp.random.uniform(-1,1, size=(n_objects, mid_dim))*cp.sqrt(6/(mid_dim+out_dim))
#             self.b2 = cp.random.uniform(-1,1, size=n_objects)*cp.sqrt(mid_dim+out_dim)
            self.b2 = cp.zeros(n_objects)
        else:
            raise NotImplemented

    def predict(self,x):
        y = cp_sigmoid((self.W1@x[:,:,None]).T[0].T + self.b1)
        return cp_sigmoid(cp.sum(self.W2*y, axis = 1) + self.b2)


def init_pool(load_saved_pool):
    TOTAL_MODELS = 50

    pool = {'model': np_mlp(n_objects = TOTAL_MODELS),
            'fitness': -100*cp.ones(TOTAL_MODELS),
             'len': TOTAL_MODELS}
    # Initialize all models
    if load_saved_pool:

        pool['model'].W1 = cp.load('saved_model_pool_numpy/W1.npy')
        pool['model'].W2 = cp.load('saved_model_pool_numpy/W2.npy')
        pool['model'].b1 = cp.load('saved_model_pool_numpy/b1.npy')
        pool['model'].b2 = cp.load('saved_model_pool_numpy/b2.npy')


    return pool


def save_pool(pool):
    cp.save('saved_model_pool_numpy/W1.npy',pool['model'].W1)
    cp.save('saved_model_pool_numpy/W2.npy',pool['model'].W2)
    cp.save('saved_model_pool_numpy/b1.npy',pool['model'].b1)
    cp.save('saved_model_pool_numpy/b2.npy',pool['model'].b2)
    print("Saved current pool!")


def model_crossover(pool, model_id1, model_id2):

    W1_1 = pool['model'].W1[model_id1]
    b1_1 = pool['model'].b1[model_id1]
    W2_1 = pool['model'].W2[model_id1]
    b2_1 = pool['model'].b2[model_id1]

    W1_2 = pool['model'].W1[model_id2]
    b1_2 = pool['model'].b1[model_id2]
    W2_2 = pool['model'].W2[model_id2]
    b2_2 = pool['model'].b2[model_id2]

    # return ([W1_1, b1_1, W2_2, b2_2], [W1_2, b1_2, W2_1, b2_1])
    return ([W1_1, b1_1, W2_1, b2_1], [W1_2, b1_2, W2_2, b2_2])


def model_mutate(weights):
    for i in range(4):
        change = cp.random.uniform(-0.5,0.5, weights[i].shape)
        cond = cp.random.uniform(0,1, weights[i].shape)

        weights[i] = cp.where(cond>0.85, weights[i] + change, weights[i])
    return weights


def change_weights(pool, new_weights):
    for idx in range(len(new_weights)):
        pool['model'].W1[idx] = new_weights[idx][0]
        pool['model'].b1[idx] = new_weights[idx][1]
        pool['model'].W2[idx] = new_weights[idx][2]
        pool['model'].b2[idx] = new_weights[idx][3]
        # pool['model'].W2[idx] = model_mutate(pool['model'].W2[idx])
        # pool['model'].b1[idx] = model_mutate(pool['model'].b1[idx])
        # pool['model'].b2[idx] = model_mutate(pool['model'].b2[idx])


def predict_action(pool, height, dist, pipe_height):
    height = cp.array(height)
    # The height, dist and pipe_height must be between 0 to 1 (Scaled by SCREENHEIGHT)
    height = cp.minimum(SCREENHEIGHT, height) / SCREENHEIGHT - 0.5
    dist = dist / 450 - 0.5 # Max pipe distance from player will be 450
    pipe_height = min(SCREENHEIGHT, pipe_height) / SCREENHEIGHT - 0.5
    neural_input = cp.tile([0, dist, pipe_height], (len(height),1))
    neural_input[:,0] = height
    output_prob = pool['model'].predict(neural_input)
    # Perform the jump action
    return output_prob <= 0.5

