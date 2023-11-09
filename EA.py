from random import random, choice, randint
from functools import reduce
from collections import namedtuple
from queue import PriorityQueue, SimpleQueue, LifoQueue, deque
from copy import  copy, deepcopy
from math import ceil, floor
import numpy as np
from tqdm.auto import tqdm
import time 
PROBLEM_SIZE  = 250 #elements to cover
NUM_SETS = 1000
SETS =  tuple(np.array([random() < 0.1 for _ in range(PROBLEM_SIZE)])  for _ in range(NUM_SETS) )

class Individual() :
    def __init__(self, current_solution : list, fitness : callable):
        self.current_solution = current_solution
        self.covering = [sum(SETS[val]) for val in current_solution]
        self.fitness = fitness
        self.fitness_value = fitness(current_solution)

    def __str__(self):
        return f'{[value for value in self.current_solution]} with fitness : {self.fitness_value} and covering sets :{self.covering}'    

class Evolutionary_algorithm():
    def uniform_crossover_increase_population(self, population : list[Individual]) -> list[Individual]:
        offspring = []
        for i in range(0, len(population), 2):
            first_par = randint(0, floor(len(population)/2))  #first half
            second_par = randint(floor(len(population)/2)+1 , len(population)-1) #second half
            new_individual = []
            for choosen in range(max(len(population[first_par].current_solution), len(population[second_par].current_solution))): 
                
                if random() < 0.5:
                     new_individual.append(population[first_par].current_solution[choosen]) if choosen < len(population[first_par].current_solution) else None
                else :
                    new_individual.append(population[second_par].current_solution[choosen]) if choosen < len(population[second_par].current_solution) else None
            if goal_check(new_individual):
                new_indiv = Individual(new_individual, fitness)
                offspring.append(new_indiv)
        value_to_return = population + offspring
        return value_to_return

    def mutation(self, population : list[Individual], mutation_rate : float = 0.1) -> list[Individual]:
        copy_pop =population
        for i in range(len(copy_pop)):
            if random() < mutation_rate:
                ##first action, we try to mutate the individual by removing the set that covers the minimum elements
                index_to_remove = np.argmin(copy_pop[i].covering)
                val = copy_pop[i].current_solution.pop(index_to_remove)
                valid = goal_check(copy_pop[i].current_solution)
                if valid: 
                    copy_pop[i].covering.pop(index_to_remove)
                    copy_pop[i].fitness_value = copy_pop[i].fitness(copy_pop[i].current_solution)
                else : 
                    copy_pop[i].current_solution.insert(index_to_remove,val)
        return copy_pop
    
    def k_tournament_selection(self, population : list[Individual], k :int = 2) -> list[Individual]:
        selected_parents = []
        copy_pop = population
        ##we select k individuals randomly and we select the best one
        for _ in range(floor(0.65* len(population))): #keep 65% of the population
            selected = []
            for index in range(k):  
                selected.append(choice(copy_pop))
            selected.sort(key = lambda x : x.fitness_value)
            selected_parents.append(selected[0])
        return selected_parents

def fitness(current_solution : list) -> int:
    return len(current_solution)

def covered(state : list) -> int:
    return reduce(
        np.logical_or,
        [SETS[i] for i in state],
        np.array([False for _ in range(PROBLEM_SIZE)]),
    )

def goal_check(state):
    return np.all(covered(state)) 

def create_random_individual(num_individuals : int) -> list[Individual]:  #initialize the population 
    count = 0
    population = []
    while count < num_individuals:
        solution = list(set([randint(0, NUM_SETS - 1) for _ in range(randint(0, NUM_SETS - 1))]))
        if goal_check(solution):

            indiv = Individual(solution, fitness)
            population.append(indiv)
            count += 1
    return population

def search_best(population : list[Individual], current_best : Individual) -> Individual:
    copy_pop = population
    copy_pop.sort(key = lambda x : x.fitness_value)
    best_solution = copy_pop[0]
    if best_solution.fitness_value < current_best.fitness_value:
        return best_solution
    else :
        return copy(current_best)


def evolutionary_algorithm (generations : int = 1000) :
    time_start = time.time()
    population = create_random_individual(PROBLEM_SIZE *5)
    gen = 0
    copy_pop = deepcopy(population)
    copy_pop.sort(key = lambda x : x.fitness_value)
    best_solution = copy_pop[0]
    print(f'Original best solution : {best_solution} and goal check : {goal_check(best_solution.current_solution)}') 
    current_best = deepcopy(best_solution)
    evolve = Evolutionary_algorithm()
    while gen < generations:
        selected = evolve.k_tournament_selection(population,5)
        
        offspring = evolve.uniform_crossover_increase_population(selected)

        offspring = evolve.mutation(offspring, mutation_rate = 0.1)

        population = offspring
        current_best = search_best(population, current_best)

        gen += 1
    print(f'Current best solution : {current_best}  and goal check : {goal_check(current_best.current_solution)}')   
    time_end = time.time()
    print(f'Generation process in {time_end - time_start} seconds')



if __name__ == '__main__':

    
        #the value True means that the set contains the element
        #we randomly create NUM_SETS sets of PROBLEM_SIZE elements (True/False)\
    evolutionary_algorithm(5000)