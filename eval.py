#!/usr/bin/env python3
import neat
import numpy as np
from data import x, y

def squared_error_fit(xi, yi):
    xi = float(xi)
    yi = float(yi)
    return (xi - yi) ** 2

def eval_genomes(genomes, config):
    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        g.fitness = 4.0
        for xi, yi in zip(x, y):
            output = net.activate(xi)
            g.fitness -= squared_error_fit(np.argmax(output), yi)
