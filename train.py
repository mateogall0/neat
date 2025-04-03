#!/usr/bin/env python3
from eval import eval_genomes
from data import val_x, val_y
import neat
import matplotlib.pyplot as plt


def plot_train(num_gens: int, avg_fitness: list, genome_size: list):
    plt.subplot(2, 1, 1)
    plt.plot(list(range(num_gens)), avg_fitness, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('NEAT Algorithm Fitness Progress')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(list(range(num_gens)), genome_size, label='Best Genome Size')
    plt.xlabel('Generation')
    plt.ylabel('Size')
    plt.title('Best Genome Size Progress')
    plt.legend()
    plt.tight_layout()
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
    genome_size = []

    num_gens = 8
    step = 1
    for generation in range(0, num_gens, step):
        p.run(eval_genomes, step)
        avg_fitness.append(stats.get_fitness_mean()[-1:])
        current_best = stats.best_genome()
        genome_size.append(len(current_best.connections) + len(current_best.nodes))

    plot_train(len(avg_fitness), avg_fitness, genome_size)

    winner = stats.best_genome()
    print('\nBest genome:\n{!s}'.format(winner))

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, xo in zip(val_x, val_y):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))


if __name__ == '__main__':
    run('config-feedforward')
