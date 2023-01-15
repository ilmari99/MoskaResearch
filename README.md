# moska
A repository for simulating Moska card games, and trying different play algorithms.

This focuses mostly on 4 player games of Moska to get a constant input shape for the neural network. However, hard-coded playing algorithms can be used for varying number of players, BUT some of their coefficients could benefit from optimizing.

This is designed for simulations of moska games with different algorithms, and playing as a human can be quite tedious since playing happens through the command line.

## Rules of the game

In finnish: https://korttipeliopas.fi/moska

Shortly, Moska is a card game for 2-8 players. Each player is dealt 6 cards, and then a trump suit is selected from the deck.

The goal of the game is to not be the last player with cards in hand.

One of the players is the target until the target has no cards, or the target lifts the cards from the table. The target can't lift the cards, until every player has confirmed that they do not want to play more cards.

The initiating player is the player on the right hand side of the target. That player can play any single card, or a combination of cards with the same value. For example valid moves could be: 2H or 14A or 5S,5C or 8S,8C,8A,10S,10C.

After the initiating player has played the cards, any player can play cards to the table. However they can only play cards of the same value, that are already on the table. For example, if the table has 5S,5C, the only playable cards are 5A,5H. This also applies to the target, who can play to themselves to get rid of unwanted cards. Other players can only play cards to the target, if the target has more cards in hand, than there are un-killed cards on the table.

The target can kill cards on the table by playing a larger card of the same suit, or any card of the trump suit (unless the card to kill is also a trump suit).

The target can also take the top card from the deck, and play it to any card on the table. If the lifted card cannot be played, it is placed on the table. If the lifted card cannot be played, then another card can not be played from the deck, until the target has killed the card played from deck.

Note, that if the target kills a card from the table, with a value that is not already on the table, the other players are then also allowed to play cards of that value to the table.

The target does not fill their hand to 6 cards, until after they have ended their turn as target (either killing everything on the table or lifting cards). When the target lifts cards, they can either lift every card that hasn't been killed, or lift every card, including killed cards and the cards used to kill said card.

If the target kills all cards, then player on the targets left side is the new target, and the previous target is the new initiating player. If the target lifts cards, then the player 2 slots to the left of the target is the new target, and the player immediately to the left of the target is the new initiating player.

The last one with cards in hand is the loser.

## General techniques
There are multiple techniques to playing the game.

Generally, each player tries to get rid of bad cards (small cards, that are not trump suit), and keep good cards when there is deck left. Combinations play a large role, since they can be played to others simultaneously.


## Simulation
The simulation is a parallelizable process, with players as threads. To make a move, a player thread acquires a  (`threading.RLock` instance) on the game state, and calculates what it wants to play. It then calls a method of the Game instance with the desired arguments. The Game instance then checks if the move is valid, and if it is, it updates the game state. If the move is not valid, an assertion error is raised. The player then either tries to play again, or frees the lock and waits for the next turn.

The game was chosen to have a lock mechanic and players as threads, because in a real game the players do not have turns, but they make moves when they want to. This also enables us to measure the effect of delay for a players success.
The threading probably does not give a performance boost, since the threads can not make any calculations, if the game state is locked, because there is no way to know whether the player is going to change the status of the game or not, and some references might be different if another thread is calculating moves due to the shared memory of the threads.

Note: The changing of the game state, and checking validity of moves, are calculated in the player threads, and not in the Game (main) thread. This doesn't matter, but coding wise it would've been better to make the Game a separate API.

The games are parallelized by using the `multiprocessing` library Pool. The games are then lazily mapped to the pool with a specified chunksize. The results are returned in order of finishing, with `Pool.imap_unordered`.

The speed of the simulation depends on the parallelization, algorithms used, and whether we are collecting data vectors. The bots using neural networks are considerably slower than the other bots. A tensorflow .h5 file must be converted to a tflite file, to be used in the simulation. This is done by the `Analysis/convert-to-tflite.py` script. The tflite models are MUCH smaller and faster than using a tensorflow model. The speed reduction with the NN bots comes mainly from the very large search space, in the order of a factorial.



## Data collection
The data is created by playing games, and saving the state of the game from the perspective of the player.
After each game, the players recorded states are then labeled according to whether they lost (1 not lost, 0 lost). Many combinations and player arguments are used, to avoid a biased dataset.

The state of the game contains:
- The number of cards in the deck
- The number of cards in each players hand
- The counted cards for each player (If a player lifts cards, we know they have the card in hand until they play it away.)
- The cards on the table
- The cards that have been killed and the cards that have been used to kill them
- The players own cards
- Whether each player is ready
- Whether each player is still in the game
- Whether there is a card played from the deck on the table
- Which player is the target

The data is represented as an integer vector, where a set of cards (for example the current cards on the table) is represented as vector of length 52, where each index corresponds to a card. The cards are sorted by suit and value. Each card that is in the set, is then assigned a number representing how many cards the card can fall at the moment. For example, the triumph suit ace, would have a value of 51 if no cards have been discarded from the game. The encoded value is -1, if the card is already discarded from the game.

The data is then balanced, by taking a random sample from the measured not-losers. If we took the full dataset, the model would be biased towards not losing, since there is a 1/4 relation between the number of losers and not-losers.

Each games state vectors and labels are shuffled and saved as a randomly named .out file, and the folder of files is read as a tensorflow TextLineDataset.

## Neural network
The neural network is a simple feed forward neural network, with generally a BatchNormalization layer, 4 hidden layers with size of input number of nodes, and 3 Dropout layers. The output layer has a sigmoid activation function, and the loss function is binary crossentropy. The optimizer is Adam.

For some models, the data is normalized in the preprocessing step, by dividing the data by 52. This does not currently make sense, since there are also values, that are not in the range of 0-52.




## TODO LIST
- Create TESTS!!
- Improve play to self, play to target, deck lift fall method, choose move (Calculate a score for each play and choose the play with the highest score)
- Create ways to get all the possible moves from a state. The possible futures from each move (if the future state is random) are then randomly sampled, and the average evaluation of the sample of futures is considered the evaluation for the given move.
- Fix information leak for the NN model about the picked cards from deck
- More efficient algorithm for calculating the possible moves from a state (currently brute force for some moves)
