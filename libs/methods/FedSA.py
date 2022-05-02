import time
import random
import math
import numpy as np
#import matplotlib.pyplot as plt
import warnings


warnings.simplefilter("ignore")


# ------------------------------------------------------------------------------
# Customization section:


class SimulatedAnnealing:

    def __init__(self, initial_temperature=0.5, cooling=0.08, lr=(0.0001, 0.1), local_update=(1, 20),
                 participants=(0, 99, 30), computing_time=1, threshold=0.1):
        self.initial_temperature = initial_temperature
        self.cooling = cooling  # cooling coefficient
        # self.number_variables = number_variables
        self.local_update = local_update
        self.participants = participants
        self.lr = lr
        self.computing_time = computing_time  # second(s)
        self.record_best_fitness = []
        self.record_best_fitness_acc = []
        self.plus = True
        self.threshold = threshold
        self.comunication = 1
        self.model = None

    def _select_range_int(self, list_space, choice, best=None, neigh_size=2, plus=False):
        print(plus)
        lista_escolhidos = list()
        list_space = list(np.arange(list_space[0], list_space[1] + 1))
        if best is None:
            for _ in range(choice):
                escolhido = False
                escolha = random.choice(list_space)
                while not escolhido:
                    if escolha in list_space and escolha not in lista_escolhidos:
                        idx_escolha = list_space.index(escolha)
                        list_space.pop(idx_escolha)
                        lista_escolhidos.append(escolha)
                        escolhido = True
                    else:
                        escolha = random.choice(list_space)
        else:
            for i in best:
                escolhido = False
                giveup = False
                neigh = 1
                if not plus:
                    escolha = i - neigh
                else:
                    escolha = i + neigh
                while not escolhido:
                    # print("I: ", i)
                    # print("Escolha: ", escolha)
                    if escolha in list_space:
                        idx_escolha = list_space.index(escolha)
                        list_space.pop(idx_escolha)
                        lista_escolhidos.append(escolha)
                        escolhido = True
                    else:
                        neigh += 1
                        if neigh <= neigh_size:
                            if not plus:
                                # print("subtrai")
                                escolha = i - neigh
                            else:
                                # print("soma")
                                escolha = i + neigh
                        elif not giveup:
                            plus = not plus
                            giveup = True
                            neigh = 1
                            if plus:
                                escolha = i + neigh
                            else:
                                escolha = i - neigh
                    if escolha == (i + neigh_size):
                        if i in list_space:
                            lista_escolhidos.append(i)
                            escolhido = True
                        else:
                            escolha = random.choice(list_space)
                            lista_escolhidos.append(escolha)
                            escolhido = True
        return lista_escolhidos

    def _select_range_float(self, list_space, best=None, plus=False):
        if best is None:
            choice = random.uniform(list_space[0], list_space[1])
        else:
            if plus:
                choice = best + 0.1 * (random.uniform(list_space[0], list_space[1]))
            else:
                choice = best - 0.1 * (random.uniform(list_space[0], list_space[1]))
        return abs(choice)

    def save_model(self, nome, model):
        np.savetxt(nome + '_weights.txt', model['weights'], delimiter=',')
        np.savetxt(nome + '_bias.txt', model['bias'], delimiter=',')

    def objective_function(self, X, ob, model, data, kwargs):
        model, loss, acc = ob(X, model, data, self.comunication, kwargs)
        self.comunication += 1
        return model, loss, acc

    # ------------------------------------------------------------------------------
    # Simulated Annealing Algorithm:
    def run(self, epoch, obj, model, data, ig_enable=False, **kwargs):
        initial_solution = list()

        initial_solution.append(self._select_range_float(self.lr))
        initial_solution.append(self._select_range_int(self.local_update, 1)[0])
        if not ig_enable:
            [initial_solution.append(i) for i in self._select_range_int(self.participants[:2], self.participants[2])]
        current_solution = initial_solution
        # print(initial_solution)
        best_solution = current_solution
        n = 1  # no of solutions accepted
        model, best_fitness, acc = self.objective_function(best_solution, obj, model, data, kwargs)  # Melhor perda
        current_temperature = self.initial_temperature  # current temperature
        start = time.time()
        # number of attempts in each level of temperature

        for i in range(1, epoch):
            print("Escolha de novo parâmetros")
            current_solution = list()
            current_solution.append(self._select_range_float(self.lr, best=best_solution[0]))
            current_solution.append(self._select_range_int(self.local_update, 1, best=[best_solution[1]],
                                                           plus=self.plus)[0])
            if not ig_enable:
                [current_solution.append(i) for i in self._select_range_int(self.participants[:2], self.participants[2],
                                                                            best=best_solution[2:], plus=self.plus)]
            print("Teste dos novos parâmetros")
            model, current_fitness, acc = self.objective_function(best_solution, obj, model, data, kwargs)
            energy = abs(current_fitness - best_fitness)
            # print(current_solution)
            if i == 1:
                EA = energy
            print("Fitness atual: ", current_fitness)
            print("Best Fitness: ", best_fitness)
            print("Energy", energy)
            print("Threshold", self.threshold)
            if current_fitness > best_fitness or energy < self.threshold:
                print("Solução atual é pior")
                self.plus = not self.plus
                # p = math.exp(-energy.numpy() / (EA * current_temperature))
                p = math.exp(-energy / (current_temperature))
                aleatorio = random.random()
                # print("Energia: ", energy)
                # print("Temperatura: ", current_temperature)
                # print("P: ", p)
                # print("Aleatorio: ", aleatorio)
                # make a decision to accept the worse solution or not
                if aleatorio < p:
                    print("A solução pior foi aceita")
                    accept = True  # this worse solution is accepted
                else:
                    print("A solução pior não foi aceita")
                    accept = False  # this worse solution is not accepted
                    print("Avaliando o best")
                    model, test_best_fitness, acc = self.objective_function(best_solution, obj, model, data, kwargs)
                    if best_fitness > test_best_fitness:
                        print("A best solution não é mais a melhor")
                        print("Escolhendo novos parâmetros")
                        current_temperature = min(self.initial_temperature,
                                                  current_temperature + (self.initial_temperature * 0.4))
                        best_solution = list()
                        best_solution.append(self._select_range_float(self.lr))
                        best_solution.append(self._select_range_int(self.local_update, 1)[0])
                        if not ig_enable:
                            [best_solution.append(i) for i in self._select_range_int(self.participants[:2],
                                                                                    self.participants[2])]
                    else:
                        print("O best continua sendo o melhor")
            else:
                print("Solução atual é melhor")
                accept = True  # accept better solution
            if accept:
                best_solution = current_solution  # update the best solution
                model, best_fitness, acc = self.objective_function(best_solution, obj, model, data, kwargs)
                n = n + 1  # count the solutions accepted
                EA = (EA * (n - 1) + energy) / n  # update EA
                # Cooling the temperature
                current_temperature = current_temperature * self.cooling
            print(best_solution)

            # print('interation: {}, best_solution: {}, best_fitness: {}'.format(i, best_solution, best_fitness))
            self.record_best_fitness.append(best_fitness)
            self.record_best_fitness_acc.append(acc)
            # Stop by computing time
            end = time.time()
            self.model = model
            #if end - start >= self.computing_time:
                #pass
                # break
        # self.save_model('proposta', model)

    '''def plot(self):
        plt.plot(self.record_best_fitness)
        plt.show()'''

    def save(self, name="name"):

        f = open(name + '_loss.txt', 'a')
        f.write(str(self.record_best_fitness)[1:-1])
        f.close()
        f = open(name + '_accuracy.txt', 'a')
        f.write(str(self.record_best_fitness_acc)[1:-1])
        f.close()