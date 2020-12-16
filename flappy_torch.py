import sys
import random
from itertools import cycle

import pygame
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE

import numpy as np

import torch
import torch.nn as nn


FPS = 30
SCREENWIDTH  = 288.0
SCREENHEIGHT = 512.0
PIPEGAPSIZE  = 100
BASEY        = SCREENHEIGHT * 0.79


class Net(nn.Module):

    def __init__(self, n_objects=50, in_dim = 3, mid_dim = 7, out_dim = 1):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.n_objects=n_objects
        self.dict_models=dict()

        for ind_model in range(n_objects):
          self.dict_models[ind_model]=dict()
          self.dict_models[ind_model]['fc1']=nn.Linear(in_dim, mid_dim)
          self.dict_models[ind_model]['fc2']=nn.Linear(mid_dim, mid_dim)
          self.dict_models[ind_model]['fc3']=nn.Linear(mid_dim, out_dim)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # Max pooling over a (2, 2) window
        res=torch.zeros(self.n_objects)
        for ind_model in range(self.n_objects):
          y=self.dict_models[ind_model]['fc1'](x[ind_model,:])
          y=self.sigmoid(self.dict_models[ind_model]['fc2'](y))
          res[ind_model]=self.sigmoid(self.dict_models[ind_model]['fc3'](y))

        return res


def init_pool(load_saved_pool):
    TOTAL_MODELS = 50
    net_dict=dict()

    pool = {'model': Net(n_objects = TOTAL_MODELS),
            'fitness': -100*np.ones(TOTAL_MODELS),
             'len': TOTAL_MODELS}
    # Initialize all models
    if load_saved_pool:
        for ind_model in range(TOTAL_MODELS):
            pool['model'].dict_models[ind_model]['fc1'].weight.data=torch.tensor(np.load('saved_model_pool_torch/{}model_W1.npy'.format(ind_model)))
            pool['model'].dict_models[ind_model]['fc2'].weight.data=torch.tensor(np.load('saved_model_pool_torch/{}model_W2.npy'.format(ind_model)))
            pool['model'].dict_models[ind_model]['fc3'].weight.data=torch.tensor(np.load('saved_model_pool_torch/{}model_W3.npy'.format(ind_model)))

            pool['model'].dict_models[ind_model]['fc1'].bias.data=torch.tensor(np.load('saved_model_pool_torch/{}model_b1.npy'.format(ind_model)))
            pool['model'].dict_models[ind_model]['fc2'].bias.data=torch.tensor(np.load('saved_model_pool_torch/{}model_b2.npy'.format(ind_model)))
            pool['model'].dict_models[ind_model]['fc3'].bias.data=torch.tensor(np.load('saved_model_pool_torch/{}model_b3.npy'.format(ind_model)))


    return pool


def save_pool(pool):
    for ind_model in range(pool['len']):
        np.save('saved_model_pool_torch/{}model_W1.npy'.format(ind_model),pool['model'].dict_models[ind_model]['fc1'].weight.detach().numpy())
        np.save('saved_model_pool_torch/{}model_W2.npy'.format(ind_model),pool['model'].dict_models[ind_model]['fc2'].weight.detach().numpy())
        np.save('saved_model_pool_torch/{}model_W3.npy'.format(ind_model),pool['model'].dict_models[ind_model]['fc3'].weight.detach().numpy())
        np.save('saved_model_pool_torch/{}model_b1.npy'.format(ind_model),pool['model'].dict_models[ind_model]['fc1'].bias.detach().numpy())
        np.save('saved_model_pool_torch/{}model_b2.npy'.format(ind_model),pool['model'].dict_models[ind_model]['fc2'].bias.detach().numpy())
        np.save('saved_model_pool_torch/{}model_b3.npy'.format(ind_model),pool['model'].dict_models[ind_model]['fc3'].bias.detach().numpy())
    print("Saved current pool!")


def model_crossover(pool, model_id1, model_id2):
    #weights of a
    #pool['model'].dict_models[model_id1]['fc1'].weight
    weights1 = pool['model'].dict_models[model_id1]['fc1'].weight
    weights1 = weights1.detach().numpy()
    weights2 = pool['model'].dict_models[model_id2]['fc1'].weight
    weights2=weights2.detach().numpy()


    return np.asarray([weights2, weights1])


def model_mutate(weights):
    change = np.random.uniform(-0.5,0.5, weights.shape)
    cond = np.random.uniform(0,1, weights.shape)

    return np.where(cond>0.85, weights + change, weights)

def change_weights(pool, new_weights):
    for idx in range(len(new_weights)):
        pool['model'].dict_models[idx]['fc1'].weight.data=torch.tensor(new_weights[idx])

def predict_action(pool, height, dist, pipe_height):
    height = np.array(height)
    # The height, dist and pipe_height must be between 0 to 1 (Scaled by SCREENHEIGHT)
    height = np.minimum(SCREENHEIGHT, height) / SCREENHEIGHT - 0.5
    dist = dist / 450 - 0.5 # Max pipe distance from player will be 450
    pipe_height = min(SCREENHEIGHT, pipe_height) / SCREENHEIGHT - 0.5
    neural_input = np.tile(torch.tensor([0, dist, pipe_height]), (len(height),1))
    neural_input[:,0] = height
    #output_prob = .predict(neural_input)
    output_prob=pool['model'].forward(torch.tensor(neural_input))

    # Perform the jump action
    return output_prob <= 0.5

