# moska
A repository for simulating moska games, and trying different play algorithms.

This focuses mostly on 4 player games of Moska to get a constant input shape for the neural network. However, hard-coded playing algorithms can be used for varying number of players, BUT some of their coefficients could benefit from optimizing.

This is designed for simulations of moska games with different algorithms, and playing as a human can be quite tedious since playing happens through the command line.

Currently I have a neural net trained for 250 epochs on 250 000 simulated games (11 M states and labels, undersampled) that can evaluate the current game position. This prediction capability will be used to evaluate the possible future positions by randomly sampling them (as there is hidden information), and the move with the best evaluation by the neural net will be chosen. The model has a roughly 70% accuracy of predicting whether a player will lose or not-lose the game, which is good, because at the beginning of the game it is nearly impossible to say whether the player will win or not.

The goal of the network is to avoid losing. Not to win. This is traditional in the Moska card game.

## TODO LIST
- Create TESTS!!
- Clean moska.py, possibly add reading the configurations from file.
- Add MoskaGameResult class
- Create a bot, that calculates the expected value for each playable combination, and chooses the best
- Create a Play -folder containing files related to playing
- Add coefficient class that can be used as a component of a MoskaPlayer
- Improve play to self, play to target, deck lift fall method, choose move (Calculate a score for each play and choose the play with the highest score)
- Create ways to get all the possible moves from a state. The possible futures from each move (if the future state is random) are then randomly sampled, and the average evaluation of the sample of futures is considered the evaluation for the given move.
- Add methods to CardMonitor, to get all cards in deck
