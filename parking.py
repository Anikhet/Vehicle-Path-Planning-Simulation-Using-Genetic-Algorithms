"""
Anikhet Mulky
am9559@g.rit.edu
file name : parking.py
"""
import math
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline


def plot_constraints(x_p, y_p, alpha_plot, v_plot, gam, bet):
    """
    Plot the graphs.
    """
    x1 = np.linspace(-16, -4, 1000)
    x3 = np.linspace(4, 15, 1000)
    y1 = 3 * np.ones_like(x1)
    y2 = np.linspace(3, -1, 1000)
    x2 = -4 * np.ones_like(y2)
    y3 = 3 * np.ones_like(x3)
    x4 = 4 * np.ones_like(y2)
    x5 = np.linspace(-4, 4, 1000)
    y5 = -1 * np.ones_like(x5)

    fig, ax = plt.subplots()
    ax.plot(x1, y1)
    ax.plot(x2, y2)
    ax.plot(x3, y3)
    ax.plot(x4, y2)
    ax.plot(x5, y5)
    ax.plot(x_p, y_p)
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.grid()
    plt.xlabel("x (ft)")
    plt.ylabel("y (ft)")
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 10, 1000), x_p)
    plt.xlabel("Time (s)")
    plt.ylabel("x (ft)")
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 10, 1000), y_p)
    plt.xlabel("Time (s)")
    plt.ylabel("y (ft)")
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 10, 1000), alpha_plot)
    plt.xlabel("Time (s)")
    plt.ylabel("alpha (rad)")
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 10, 1000), v_plot)
    plt.xlabel("Time (s)")
    plt.ylabel("v (ft/s)")
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 10, 1000), gam)
    plt.xlabel("Time (s)")
    plt.ylabel("gamma (rad/s)")
    fig, ax = plt.subplots()
    ax.plot(np.linspace(0, 10, 1000), bet)
    plt.xlabel("Time (s)")
    plt.ylabel("beta (ft/s^2 )")
    ax.legend()
    plt.show()


def gray_decimal(n):
    """
    Grey code to decimal.
    https://stackoverflow.com/questions/72027920/python-graycode-saved-as-string-directly-to-decimal
    """
    n = int(n, 2)
    m = n >> 1
    while m:
        n ^= m
        m >>= 1
    return n


def binary_to_gray(n):
    """
    Cited from :
    https://www.sanfoundry.com/python-program-convert-binary-gray-code/
    """
    n = str(n)
    n = int(n, 2)
    n ^= (n >> 1)

    return bin(n)[2:]


def convert_array(pop, length, param):
    """
    Converting binary code to grey code and separating them to pieces of string for gamma,beta.
    """
    gray = []

    for index in range(len(pop)):
        gray.append([])
        for inside_index in range(0, len(pop[index]), length):
            if inside_index + length == length * param:
                grey = binary_to_gray(''.join(str(v) for v in pop[index][inside_index: inside_index + length]))
                if len(grey) < length:
                    while len(grey) != length:
                        grey = '0' + grey
                gray[index].append(grey)
                break
            concat = ''
            for string in range(inside_index, inside_index + length):
                concat += str(pop[index][string])
            grey = binary_to_gray(concat)
            if len(grey) < length:
                while len(grey) != length:
                    grey = '0' + grey
            gray[index].append(grey)

    return gray


def gamma_beta(grey, length):
    """
    Nomralizing and making two arrays for each gamma and beta.
    """
    gamma = []
    beta = []
    for index in range(len(grey)):
        gamma.append([])
        beta.append([])
        for inside_index in range(0, len(grey[index]), 2):
            d1 = gray_decimal(grey[index][inside_index])
            d2 = gray_decimal(grey[index][inside_index + 1])
            normalization_gamma = ((d1 / ((2 ** length) - 1)) * 1.048) + (-0.524)
            normalization_beta = ((d2 / ((2 ** length) - 1)) * 10) + (-5)

            gamma[index].append(normalization_gamma)
            beta[index].append(normalization_beta)
    return gamma, beta


def interp(gamma, beta, size, param):
    """
    Interpolation.
    """
    time_array = []
    for index in range(param // 2):
        time_array.append(index)

    interpolate_gamma = []
    interpolate_beta = []
    for index in range(size):
        interpolate_gamma.append([])
        interpolate_beta.append([])
        f = CubicSpline(time_array, gamma[index])
        x_new = np.linspace(0, 10, 1000)
        y_new = f(x_new)
        interpolate_gamma[index].append(y_new)
        f = CubicSpline(time_array, beta[index])
        x_new = np.linspace(0, 10, 1000)
        y_new = f(x_new)
        interpolate_beta[index].append(y_new)

    return interpolate_gamma, interpolate_beta


def cost(x, y, a, v):
    """
    Calculating Cost.
    """
    return math.sqrt((x ** 2 + y ** 2)) + abs(a) + abs(v)


def fit(cost_function):
    """
    Calculating fitness.
    """
    fitness = []
    for index in range(len(cost_function)):
        x = (1 / (cost_function[index] + 1))
        fitness.append(x)

    return fitness


def euler(gamma, beta, index):
    """
    Performing euler to get ODE's.
    """
    x_0 = 0
    y_0 = 8
    alpha = gamma[index][0][0]
    v = beta[index][0][0]
    x_plot = []
    y_plot = []
    alpha_plot = []
    v_plot = []
    alpha_plot.append(alpha)
    v_plot.append(v)
    x_plot.append(x_0)
    y_plot.append(y_0)

    flag = True
    for index_main in range(1, len(gamma[index][0])):
        x = x_0 + (v * math.cos(alpha)) * 0.01
        y = y_0 + (v * math.sin(alpha)) * 0.01
        alpha_new = alpha + (gamma[index][0][index_main]) * 0.01
        v_new = v + (beta[index][0][index_main]) * 0.01
        x_0 = x
        y_0 = y
        alpha = alpha_new
        v = v_new

        x_plot.append(x_0)
        y_plot.append(y_0)
        alpha_plot.append(alpha)
        v_plot.append(v)
        if (x_0 <= -4 and y_0 > 3) or ((-4 < x_0 < 4) and y_0 > -1) or (x_0 >= 4 and y_0 > 3):
            pass

        else:
            flag = False
    if flag:
        return cost(x_0, y_0, alpha, v), x_plot, y_plot, alpha_plot, v_plot, [x_0, y_0, alpha, v]
    else:
        return 200, x_plot, y_plot, alpha_plot, v_plot, [x_0, y_0, alpha, v]


def bitflip(r1, r2):
    """
    A bit flip if less than mutation rate.
    """
    s1 = ''
    s2 = ''
    for index in range(len(r1)):
        if r1[index] == '1':
            s1 += '0'
        else:
            s1 += '1'
        if r2[index] == '1':
            s2 += '0'
        else:
            s2 += '1'
    return s1, s2


def mutate(grey, fitness, index_help, mutation_rate, param, length):
    """
    Mutation to make a new population.
    """
    children = []
    child = 0

    for index in range(0, int((len(grey) - 2) / 2)):
        elitism = random.choices(index_help, fitness, k=2)
        children.append([])
        children.append([])
        s = ''
        s1 = ''
        for index_main in range(0, param):
            s += grey[elitism[0]][index_main]
            s1 += grey[elitism[1]][index_main]
        crossover = random.randint(1, len(s))
        c1 = s[:crossover] + s1[crossover:len(s)]
        c2 = s1[:crossover] + s[crossover:len(s)]

        for i in range(len(c1)):
            if random.random() < mutation_rate:
                c1 = c1[:i] + ('0' if c1[i] == '1' else '1') + c1[i + 1:]
        for i in range(len(c2)):
            if random.random() < mutation_rate:
                c2 = c2[:i] + ('0' if c2[i] == '1' else '1') + c2[i + 1:]
        for index_main in range(0, length * param, length):
            children[child].append(c1[index_main:index_main + length])
            children[child + 1].append(c2[index_main:index_main + length])

        child += 2

    return children


def program_done(final, x, y, final_cost, alpha_plot, v_plot, gam, bet):
    """
    This function executes to finish the program.
    """
    print("Final state values:")
    print("x_f = ", final[0])
    print("y_f = ", final[1])
    print("alpha_f = ", final[2])
    print("v_f = ", final[3])

    plot_constraints(x, y, alpha_plot, v_plot, gam, bet)


def generation_one(POP_SIZE, NUM_PARAMS_PER_CONTROL_VAR, BIN_CODE_LENGTH, MUTATION_RATE):
    """
    The 0th generation.
    """
    pop = np.random.randint(2, size=(POP_SIZE, NUM_PARAMS_PER_CONTROL_VAR * BIN_CODE_LENGTH))
    grey = convert_array(pop, BIN_CODE_LENGTH, NUM_PARAMS_PER_CONTROL_VAR)
    gamma, beta = gamma_beta(grey, BIN_CODE_LENGTH)
    interpolate_gamma, interpolate_beta = interp(gamma, beta, POP_SIZE, NUM_PARAMS_PER_CONTROL_VAR)
    final_cost = []
    x_plot = []
    y_plot = []
    alpha_plot = []
    v_plot = []
    for index in range(0, POP_SIZE):
        e, x, y, alpha, v_var, final = euler(interpolate_gamma, interpolate_beta, index)

        x_plot.append(x)
        y_plot.append(y)
        alpha_plot.append(alpha)
        v_plot.append(v_var)
        final_cost.append(e)
        if e < 0.1:
            program_done(final, x_plot, y_plot, final_cost, alpha, v_plot[-1],
                         interpolate_gamma[index][0], interpolate_beta[index][0])

    fitness = fit(final_cost)
    index_helper = []
    for index in range(0, POP_SIZE):
        index_helper.append(index)

    top_two_indices = np.argsort(fitness)[-2:][::-1]
    new = mutate(grey, fitness, index_helper, MUTATION_RATE, NUM_PARAMS_PER_CONTROL_VAR, BIN_CODE_LENGTH)

    new.append(grey[top_two_indices[0]])
    new.append(grey[top_two_indices[1]])
    grey = new
    print("Generation 0 : J = ", final_cost[top_two_indices[0]])
    return grey, index_helper


def main():
    start = time.time()
    POP_SIZE = 200
    NUM_PARAMS_PER_CONTROL_VAR = 20
    NUM_PARAMS_PER_CONTROL_VAR *= 2
    BIN_CODE_LENGTH = 7
    MUTATION_RATE = 0.005
    MAX_GEN = 1200
    MAX_TIME = 7 * 60

    grey, index_helper = generation_one(POP_SIZE, NUM_PARAMS_PER_CONTROL_VAR, BIN_CODE_LENGTH, MUTATION_RATE)

    for index_main in range(1, MAX_GEN):
        # Iterates from generation 1 until the max generation or 7 minutes or convergence occurs

        gamma, beta = gamma_beta(grey, BIN_CODE_LENGTH)
        interpolate_gamma, interpolate_beta = interp(gamma, beta, POP_SIZE, NUM_PARAMS_PER_CONTROL_VAR)
        final_cost = []
        x_plot = []
        y_plot = []
        alpha_plot = []
        v_plot = []
        end_var = []

        for index in range(0, POP_SIZE):
            e, temp1, temp2, alpha, v_var, final = euler(interpolate_gamma, interpolate_beta, index)
            x_plot.append(temp1)
            y_plot.append(temp2)
            alpha_plot.append(alpha)
            v_plot.append(v_var)
            end_var.append(final)
            final_cost.append(e)
            if e < 0.1:
                controls_final = []
                for grey_index in range(len(gamma[index])):
                    controls_final.append((gamma[index][grey_index]))
                    controls_final.append((beta[index][grey_index]))
                arr = np.array(controls_final)
                np.savetxt('controls.dat', arr, delimiter='\t')
                program_done(final, temp1, temp2, final_cost, alpha, v_var,
                             interpolate_gamma[index][0], interpolate_beta[index][0])

                quit()

        fitness = fit(final_cost)
        top_two_indices = np.argsort(fitness)[-2:][::-1]
        new = mutate(grey, fitness, index_helper, MUTATION_RATE, NUM_PARAMS_PER_CONTROL_VAR, BIN_CODE_LENGTH)
        new.append(grey[top_two_indices[0]])
        new.append(grey[top_two_indices[1]])
        grey = new
        print(f"Generation {index_main} : J = ", final_cost[top_two_indices[0]])
        if index_main == MAX_GEN - 1 or time.time() - start >= MAX_TIME:
            indice = np.argsort(final_cost)[:1]
            indice = indice[0]
            controls_final = []
            for grey_index in range(len(gamma[indice])):
                controls_final.append((gamma[indice][grey_index]))
                controls_final.append((beta[indice][grey_index]))
            arr = np.array(controls_final)
            np.savetxt('controls.dat', arr, delimiter=' ')

            program_done(end_var[indice], x_plot[indice], y_plot[indice], final_cost, alpha_plot[indice],
                         v_plot[indice],
                         interpolate_gamma[index_main][0], interpolate_beta[index_main][0])


            quit()


if __name__ == '__main__':
    main()
