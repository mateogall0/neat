#!/usr/bin/env python3
import numpy as np
import graphviz
from eval import eval_genomes
from data import val_x, val_y
import neat
import matplotlib.pyplot as plt


def plot_train(num_gens: int, avg_fitness: list, best_fitness: list, genome_size: list,
        val_fitness: list, validation_amount: list):

    subplot_layout_v = 4
    plt.subplot(subplot_layout_v, 1, 1)
    plt.plot(list(range(num_gens)), avg_fitness, label='Average Fitness')
    plt.plot(list(range(num_gens)), best_fitness, label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('NEAT Algorithm Fitness Progress')
    plt.legend()
    plt.grid(True)

    plt.subplot(subplot_layout_v, 1, 2)
    plt.plot(list(range(num_gens)), genome_size, label='Best Genome Size')
    plt.xlabel('Generation')
    plt.ylabel('Size')
    plt.title('Best Genome Size Progress')
    plt.legend()
    plt.grid(True)

    plt.subplot(subplot_layout_v, 1, 3)
    plt.plot(list(range(num_gens)), genome_size, label='Genome Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Best Genome Validation Fitness')
    plt.legend()
    plt.grid(True)

    plt.subplot(subplot_layout_v, 1, 4)
    plt.bar(list(range(num_gens)), validation_amount, label='Validation Assertions Passed')
    plt.xlabel('Generation')
    plt.ylabel('Count')
    plt.title('Validation Assertions per Generation')
    plt.grid(True, axis='y')

    plt.tight_layout()
    plt.show()


def plot_genome(config, genome, view=True, filename="nerual_network", node_names=None,
                show_disabled=True, prune_unused=False, fmt='svg'):
    if prune_unused:
        genome = genome.get_pruned_copy(config.genome_config)

    if node_names is None:
        node_names = {}

    assert type(node_names) is dict

    node_colors = {}

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set()
    for k in config.genome_config.input_keys:
        inputs.add(k)
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    outputs = set()
    for k in config.genome_config.output_keys:
        outputs.add(k)
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}

        dot.node(name, _attributes=node_attrs)

    used_nodes = set(genome.nodes.keys())
    for n in used_nodes:
        if n in inputs or n in outputs:
            continue

        attrs = {'style': 'filled',
                 'fillcolor': node_colors.get(n, 'white')}
        dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            # if cg.input not in used_nodes or cg.output not in used_nodes:
            #    continue
            input, output = cg.key
            a = node_names.get(input, str(input))
            b = node_names.get(output, str(output))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(a, b, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(f'{filename}.{fmt}', view=view)

    return dot

def validate_data(genome, verbose=True) -> list:
    outputs = []
    for xi, xo in zip(val_x, val_y):
        output = np.argmax(genome.activate(xi))
        outputs.append(output)
        if verbose:
            print("input {!r}, expected output {!r}, got {!r}".format(xi, xo, output))
    return outputs


def run(config_file, plot=True):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    if plot:
        avg_fitness = []
        best_fitness = []
        genome_size = []
        best_val_fitness = []
        validation_amount = []

    num_gens = 8
    step = 1
    for generation in range(0, num_gens, step):
        p.run(eval_genomes, step)
        if plot:
            avg_fitness.append(stats.get_fitness_mean()[-1:])
            current_best = stats.best_genome()
            best_fitness.append(current_best.fitness)
            genome_size.append(len(current_best.connections) + len(current_best.nodes))
            current_best_net = neat.nn.FeedForwardNetwork.create(current_best, config)
            val_data_res = validate_data(current_best_net, False)
            best_val_fitness.append(np.mean(val_data_res))
            vals = [1 if i == 0.0 else 0 for i in val_data_res]
            validation_amount.append(sum(vals))

    winner = stats.best_genome()
    if plot:
        plot_train(len(avg_fitness), avg_fitness, best_fitness, genome_size,
                   best_val_fitness, validation_amount)
        plot_genome(config, winner)


    print('\nBest genome:\n{!s}'.format(winner))


    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    validate_data(winner_net, verbose=True)


if __name__ == '__main__':
    run('config-feedforward', plot=True)
