#!/usr/bin/env python3
from eval import eval_genomes
from data import x, y
import neat
import matplotlib.pyplot as plt


def plot_train(gens: list, avg_fitness: list):
    plt.plot(gens, avg_fitness, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('NEAT Algorithm Fitness Progress')
    plt.legend()
    plt.grid(True)
    plt.show()

def run(config_file, plot=True):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    avg_fitness = []

    num_gens = 16384
    step = 1024
    for generation in range(0, num_gens, step):
        p.run(eval_genomes, step)
        current_avg_fitness = stats.get_fitness_mean()[-1]
        avg_fitness.append(current_avg_fitness)
        print(f"Generation {generation}: Avg Fitness = {current_avg_fitness}")

    gens = list(range(0, num_gens, step))
    plot_train(gens, avg_fitness)

    winner = stats.best_genome()
    print('\nBest genome:\n{!s}'.format(winner))

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(x, y):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))


if __name__ == '__main__':
    run('config-feedforward')
