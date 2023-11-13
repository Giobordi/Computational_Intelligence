## ES FIRST ALGORITHM

### Description

My evolution strategy start with a population equal to a percentage of the possible solutions. Each generation the number of new individuals created is equal to the population size / 2 the offspings are added to the population. The population is sorted by fitness that is the nim.sum() (there is a sort of 'pensality term' used where the nim.sum() is equal to 0) in a increasing order. /
The #population_size best individuals are selected and the others are discarded and the process start again. \
The mutation function  -> mutation : a individual is mutated by changing the number of elements to substract from the heap.
The crossover function -> crossover : two individuals are crossed by taking the row from the first individual and the elements to substract from the second individual. If it is not valid, we try the opposite and if even this is not valid we take the first individual.




