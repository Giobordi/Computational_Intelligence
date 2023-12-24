# Tic Tac Toe with Reinforcement Learning
### Collaboration with Edoardo Franco s310228 [@ed0sh](https://github.com/ed0sh)

The algorithm is based on the Q-learning algorithm, which is a model-free reinforcement learning algorithm. \
The agent learns from playing a number of games equal to the variable `TRAINING_EPOCHS`, agains a random player. The reward is +1 if the agent wins, -3 if the agent loses and -1 if the game ends in a draw. These values has been chosen after some testing, tuning them to get the best results. 

The agent expoit the symmetry of the game, so it learns only specific states, avoiding to learn the same state multiple times, this technique helps the agent to learn faster and to get better results (obviously the learning space is reduced). 

The game-state is represented in a particular form, the Tic-Tac-Toe board is not treated as a linear vector of indexes.
Instead, it follows a spiral in order to easily manage the rotations and the symmetries.

### Code details

- The `class TicTacToe` is the environment in which the agent plays, it contains the methods to play a game and those 
to check if the game is over or not.

- There are the symmetry function `rotate_90_right` and `rotate_state_90_right` which are used to exploit the symmetry 
of the game.

- The `class ReinforcedPlayer` is the agent, it contains the Q table and the methods to update it and to choose the next action. The parameter *epsilon* help to weight the exploration-exploitation tradeoff, it is set to 0.01.

-There are 2 function `save_model` and `load_model` to save and load the ReinforcedPlayer object, to avoid to train 
the agent every time..

