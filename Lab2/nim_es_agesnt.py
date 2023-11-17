import inspect
import logging
from pprint import pprint, pformat
from collections import namedtuple
import random
from copy import deepcopy
from math import floor, ceil, inf
from random import randint, random, choice
import pprint 
from tqdm.auto import tqdm
import sys
import numpy as np
import time
import matplotlib.pyplot as plt
NUM_ROWS = 5
POPULATION_SIZE = 20
NUM_RULES = 15
GENERATIONS = 80
TOURNAMENT_SIZE = 10
NUM_MATCHES = 100
MUTATION_RATE = 0.1
TEST_MATCHES = 100

Nimply = namedtuple("Nimply", "row, num_objects")
class Nim:
    def __init__(self, num_rows: int, k: int = None) -> None:
        self._rows = [i * 2 + 1 for i in range(num_rows)]
        self._k = k

    def __bool__(self):
        return sum(self._rows) > 0

    def __str__(self):
        return "<" + " ".join(str(_) for _ in self._rows) + ">"

    @property
    def rows(self) -> tuple:
        return tuple(self._rows)
    
    def full_rows(self) -> int:
        return sum(1 for r in self._rows if r != 0)

    def nimming(self, ply: Nimply) -> None:
        row, num_objects = ply
        assert self._rows[row] >= num_objects
        assert self._k is None or num_objects <= self._k
        self._rows[row] -= num_objects

###basic strategies
def pure_random(state: Nim) -> Nimply:
    """A completely random move"""
    row = choice([r for r, c in enumerate(state.rows) if c > 0])
    num_objects = randint(1, state.rows[row])
    return Nimply(row, num_objects)

def gabriele(state: Nim) -> Nimply:
    """Pick always the maximum possible number of the lowest row"""
    possible_moves = [(r, o) for r, c in enumerate(state.rows) for o in range(1, c + 1)]
    return Nimply(*max(possible_moves, key=lambda m: (-m[0], m[1])))

def nim_sum(state: Nim) -> int:
    tmp = np.array([tuple(int(x) for x in f"{c:032b}") for c in state.rows])
    xor = tmp.sum(axis=0) % 2
    return int("".join(str(_) for _ in xor), base=2)


def analize(raw: Nim) -> dict:
    cooked = dict()
    cooked["possible_moves"] = dict()
    for ply in (Nimply(r, o) for r, c in enumerate(raw.rows) for o in range(1, c + 1)):
        tmp = deepcopy(raw)
        tmp.nimming(ply)
        cooked["possible_moves"][ply] = nim_sum(tmp)
    return cooked
def optimal(state: Nim) -> Nimply:
    analysis = analize(state)
    logging.debug(f"analysis:\n{pformat(analysis)}")
    spicy_moves = [ply for ply, ns in analysis["possible_moves"].items() if ns != 0]
    if not spicy_moves:
        spicy_moves = list(analysis["possible_moves"].keys())
    ply = choice(spicy_moves)
    return ply

class Rule : 
    def __init__(self , condition : callable , action : Nimply, weight : int =0.5) : 
        self.condition = condition
        self.action = action
        self.weight = weight

    def update_rule_weight(self, improve :bool = True) : 
        if improve :     
            self.weight *= 1.00001
        else :
            self.weight /= 1.00001

    def __str__(self) :
        return f"{inspect.getsource(self.condition)} -> {self.action} : {self.weight}"

class EsNimAgent : 
    def __init__(self, rules : list[Rule], played : int = 0, won : int = 0):
        self.rules = rules
        self.match_played = played
        self.match_won = won

    def win_match(self, win :bool = True) : 
        self.match_played += 1
        if win : 
            self.match_won += 1
        self.update_rules_weight(win)

    def update_rules_weight(self, win :bool = True) :
        '''
        A win means that the rules can be assumed as good rules, so we increase their weight
        A loss means that the rules are not good, so we decrease their weight 
        '''
        for rule in self.rules : 
            rule.update_rule_weight(win)
        

    def agent_fitness(self) : 
        if self.match_played == 0 : 
            return 0
        return self.match_won/self.match_played
    
    def move_selection(self, state : Nim) -> Nimply:
        possible_moves = list()
        for rule in self.rules :
            if rule.condition(state) and rule.weight > 0 \
            and rule.action.num_objects <= state.rows[rule.action.row] : ## to take the correct rule with the highest weight
                possible_moves.append([rule.action, rule.weight])
                #return rule.action
        if len(possible_moves) == 0 :
            if random() < 0.15 :
                self.update_rules_weight(False) ##the agent is penalized if it cannot find a rule to apply 
                #print("random penality") 
            return pure_random(state)
        else : 
            # a random choice between the possible moves
            return choice(possible_moves)[0]


####for the creation of a population 
def all_possible_moves(state : Nim) -> list : 
    return [Nimply(r, o) for r, c in enumerate(state.rows) for o in range(1, c + 1)]
   

def condition_for_rule(number_of_row : int) -> list[callable] :
    possible_rules = list()
    
    ##random element
    possible_rules.append(lambda state:  state.rows[randint(0, number_of_row-1)] > 1 and state.rows[randint(0, number_of_row-1)] > 1)

    ###if there are many element or not
    possible_rules.append(lambda state:   sum(state.rows) >  len(state.rows) * 2) ##many elements
    possible_rules.append(lambda state:   sum(state.rows) <= len(state.rows) * 2) ##few elements

    ##one possible final move
    possible_rules.append(lambda state:  sum(state.rows) <= 2) 
    possible_rules.append(lambda state:  sum(state.rows) ==3 and state.full_rows() == 2) 

    ###if there are many full rows or not
    possible_rules.append(lambda state:  state.full_rows() >= ceil(len(state.rows)/2)) ##many full rows
    possible_rules.append(lambda state:  state.full_rows() < ceil(len(state.rows)/2))



    ##add more rules 
    #possible_rules.append(lambda state: nim_sum(state) != 0)
    return possible_rules

def global_set_rules(conditions_for_rules : list, moves : list, fixed_weight : int = 0.5) -> list :
    global_set_of_rules = list()
    for condition in conditions_for_rules : 
        for move in moves : 
            global_set_of_rules.append(Rule(condition, move, fixed_weight))
    return global_set_of_rules  

def create_population (global_set_of_rules : list, population_size : int = 10, numbers_of_rules : int = 5) -> list : 
    population = list()
    for _ in range(population_size) : 
        one_element = list()
        for __ in range(numbers_of_rules) : 
            choosen_rule = choice(global_set_of_rules)
            #print(choosen_rule)
            one_element.append(choosen_rule)
        population.append(EsNimAgent(one_element))  
    return population

###########
####Evolutionary Strategy
def play_a_game (agent_one , agent_two, nim : Nim) -> int :
    '''
    Play a game between two agents, return 0 if the first agent wins, 1 otherwise. 
    Randomly choose who starts.
    '''
    turn = randint(0,1)
    strategies = [agent_one, agent_two]
    while nim : 
        ply = strategies[turn](nim)
        nim.nimming(ply)
        turn = 1 - turn
    return turn

def mutation(agent_orig : EsNimAgent) -> EsNimAgent : 
    # agent = deepcopy(agent_orig)
    # for rule in agent.rules : 
    #     if random() > 0.5 :
    #         rule.weight *= 1.00001
    #     else : 
    #         rule.weight /= 1.00001
    # return agent
    #change the rules 
    rules = list()
    for rule in agent.rules :
        if random() < 0.5:
            r = choice(set_rules)
            rules.append(r)
        else : 
            r = rule
            rules.append(r)
    return EsNimAgent(rules, agent.match_played, agent.match_won)
    
            
            

def one_cut_crossover(agent_one : EsNimAgent, agent_two : EsNimAgent) -> EsNimAgent :
    '''
    This function keep the first half of rules from the first parent and the second half of rules from the second parent
    '''
    new_rules = list()
    for i in range(len(agent_one.rules)) :
        if i < ceil(len(agent_one.rules)/2) :
            new_rules.append(agent_one.rules[i])
        else : 
            new_rules.append(agent_two.rules[i])
    return EsNimAgent(new_rules, ceil((agent_one.match_played + agent_two.match_played)/2), ceil((agent_one.match_won + agent_two.match_won)/2) )
    ##the first value is always 3 * num matches 
    

def survival_selection(population : list[EsNimAgent], population_size : int = 10) -> list[EsNimAgent] : 
    ##sort the population by fitness
    sorted_population = sorted(population, key=lambda x : x.agent_fitness(), reverse=True)
    ##it is necessary to reset the number of matches played and won
    stats = [sorted_population[0].match_played, sorted_population[0].match_won]
    for agent in sorted_population[:population_size]: 
        agent.match_played = 0
        agent.match_won = 0
    return sorted_population[:population_size], stats

def tournament_selection(population : list[EsNimAgent], k :int = 2) -> list :
    selected = list()
    for _ in range(k):
        selected.append(choice(population))
    ### the one with the highest fitness wins
    return max(selected, key=lambda x : x.agent_fitness())

def generate_new_generation(population : list[EsNimAgent], tournament_size : int = 5,offspring_size : int = 0 , mutation_rate : float = 0.1) :
    new_offspring = list()
    for _ in range(offspring_size) : 
        parent_one = tournament_selection(population, k=tournament_size)
        
        if random() < mutation_rate : 
            new_offspring.append(mutation(parent_one))

        else : #cross over
            parent_two = tournament_selection(population, k=tournament_size)
            new_offspring.append(one_cut_crossover(parent_one, parent_two))
    return new_offspring
###########







if __name__ == "__main__":
    t1 = time.time()
    nim = Nim(NUM_ROWS)
    all_moves = all_possible_moves(nim)
    conditions = condition_for_rule(NUM_ROWS)
    global set_rules
    set_rules = global_set_rules(conditions, all_moves)
    population = create_population(set_rules, population_size=POPULATION_SIZE, numbers_of_rules=NUM_RULES)
    best_agent = list()
    bb = []
    ###GENERATION STRATEGIES
    for i in range(GENERATIONS):
        for agent in population : 
            #each agent plays against the std agent in the population
            current_win = 0
            for d in range(NUM_MATCHES) : 
                
                nim = Nim(NUM_ROWS)
                w = play_a_game(agent.move_selection, optimal, nim) #against the optimal strategy
                if w == 0 : 
                    current_win += 1
                nim = Nim(NUM_ROWS)
                w = play_a_game(agent.move_selection, pure_random, nim) #against the random strategy
                if w == 0 : 
                    current_win += 1
                nim = Nim(NUM_ROWS)
                w = play_a_game(agent.move_selection, gabriele, nim) #against the gabriele strategy
                if w == 0 : 
                    current_win += 1
                ###update the weight of the agent
            for _ in range(current_win) : 
                agent.win_match(True)
            for _ in range(NUM_MATCHES*3-current_win) :
                agent.win_match(False)
            
        offspring = generate_new_generation(population,TOURNAMENT_SIZE,offspring_size=len(population),mutation_rate= MUTATION_RATE)
        population.extend(offspring)
        population , best_stats = survival_selection(population, POPULATION_SIZE)
        print(f"generation {i} completed")
        print(f"best agent play : {best_stats[0]} and won {best_stats[1]}")
        ###la regola con il peso più alto
        bb.append(best_stats[1])
        print(f"best top rule : { [rule for rule in population[0].rules if rule.weight == max([rule.weight for rule in population[0].rules]) ].pop()} ")
        if(i==GENERATIONS-1 or i==ceil(GENERATIONS/2) or i==1):
            best_agent.append(population[0])
    for rule in best_agent[-1].rules : 
        print(rule)
    t2 = time.time()
    print(f"Time elapsed: {t2-t1}")
    ###TESTING THE BEST AGENT
    plt.plot([i for i in range(GENERATIONS)],bb, marker='o')  # 'marker' specifica il tipo di marker da usare sui punti
    plt.xlabel('Generazioni')  # Etichetta dell'asse x
    plt.ylabel('Vittorie')  # Etichetta dell'asse y
    plt.title('Andamento vittorie nelle generazioni')  # Titolo del grafico
    plt.show()  
    for best in best_agent :
        wins = [0, 0]
        sequence = []
        strategy = [best.move_selection, optimal]

        for i in range(TEST_MATCHES):
            turn = randint(0,1)
            nim = Nim(NUM_ROWS) 
            while nim:
                ply = strategy[turn](nim)
                nim.nimming(ply)
                turn = 1 - turn
            wins[turn] += 1
            sequence.append(turn)      
            
        print(f"wins : {wins} ")
        print(f"Percentage : player 0 {wins[0]/TEST_MATCHES}, optimal {wins[1]/TEST_MATCHES}")
    #print(f"sequence : {sequence}")
    for best in best_agent :
        wins = [0, 0]
        sequence = []
        strategy = [best.move_selection, pure_random]

        for i in range(TEST_MATCHES):
            turn = randint(0,1)
            nim = Nim(NUM_ROWS) 
            while nim:
                ply = strategy[turn](nim)
                nim.nimming(ply)
                turn = 1 - turn
            wins[turn] += 1
            sequence.append(turn)      
            
        print(f"wins : {wins} ")
        print(f"Percentage : player 0 {wins[0]/TEST_MATCHES}, pure_random {wins[1]/TEST_MATCHES}")
    #print(f"sequence : {sequence}")
    for best in best_agent :
        wins = [0, 0]
        sequence = []
        strategy = [best.move_selection, gabriele]

        for i in range(TEST_MATCHES):
            turn = randint(0,1)
            nim = Nim(NUM_ROWS) 
            while nim:
                ply = strategy[turn](nim)
                nim.nimming(ply)
                turn = 1 - turn
            wins[turn] += 1
            sequence.append(turn)      
            
        print(f"wins : {wins} ")
        print(f"Percentage : player 0 {wins[0]/TEST_MATCHES}, gabriele {wins[1]/TEST_MATCHES}")
        #print(f"sequence : {sequence}")
            