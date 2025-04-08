#!/usr/bin/env python3
import neat
import numpy as np
from data import x, y

def squared_error_fit(xi, yi):
    xi = float(xi)
    yi = float(yi)
    return (xi - yi) ** 2

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    y_true_one_hot = np.zeros_like(y_pred)
    y_true_one_hot[int(y_true)] = 1

    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    return -np.sum(y_true_one_hot * np.log(y_pred))

def eval_genomes(genomes, config):
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        g.fitness = 4.0
        for xi, yi in zip(x, y):
            output = net.activate(xi)
            Y = softmax(output)
            #g.fitness -= squared_error_fit(np.argmax(Y), yi)
            g.fitness -= cross_entropy_loss(Y, yi)
