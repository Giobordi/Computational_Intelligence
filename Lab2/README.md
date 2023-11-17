## ES FIRST ALGORITHM

### Description

First approach (probably does not follow the requirements of the exercise)

My evolution algorithm start with a population equal to a percentage of the possible solutions. Each generation the number of new individuals created is equal to the population size / 2 the offspings are added to the population. The population is sorted by fitness that is the nim.sum() (there is a sort of 'pensality term' used where the nim.sum() is equal to 0) in a increasing order. /
The #population_size best individuals are selected and the others are discarded and the process start again. \
The mutation function  -> mutation : a individual is mutated by changing the number of elements to substract from the heap.
The crossover function -> crossover : two individuals are crossed by taking the row from the first individual and the elements to substract from the second individual. If it is not valid, we try the opposite and if even this is not valid we take the first individual.


ES approach 

- **Class Rule** : it is an implementation of a possible rule (like the nim.sum). The condition is a function that return a boolean based on some state of the Nim game. The action is a move done if the condition is true. \
 The weight value represent the importance of this rule, the default value is 0.5 and it is updated during the evolution process. 

- **Class EsNimAgent** : it is an implementation of the NimAgent for the ES, it has a list of rules, the number of games played and the number of games won. The fitness is the ratio between these to values. 
The function move_selection() is used to select the move to do based on the rules. The state is evaluated with the condition of the rule, if it is a possible move the action is put inside a list of possible moves. The move is selected based on the weight, if there are no possible moves the move is selected randomly.


- *Functions for initialization* : 
    - all_possible_moves() : it returns all the possible moves for a Nim game
    - condition_for_rule() : it returns alla the possible condition for the rule class
    - global_set_rules() : it associate all the conditions to all the possible moves
    - create_population() : it creates a population of NimAgent with random rules

- **Start the ES** : 
    - mutation() : change the rules inside an agent.
    - one_cut_cross_over() : it crosses two agents by taking the first part of the first agent and the second part of the second agent
    - survival_selection() : it selects the best agents based on the fitness
    - tournament_selection() : the best agent between a 'tournament_size' number of agents, the competition is based on the fitness
    - generate_new_generation() : it is the function that create a new generation (offspring) applying the mutation and the crossover and the tournament selection.
    - ES : it is the function that start the ES, each agent do a number of games equal to the 'number_of_games' parameter with the 3 different opponents (optimal, pure_random and gabriele) and update the weights of the rule based on win/lose. The main approach is a 'plus' approach, the new generation is added to the old one and the best agents are selected, the number of new agents created is equal to the number of agents in the old generation. At the end of the entire process the agent with the best fitness is returned