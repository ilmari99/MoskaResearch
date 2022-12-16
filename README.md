# moska
A repo for simulating moska games, and trying different play algorithms.

This is designed for simulations of moska games with different algorithms, and playing as a human can be quite tedious.

## TODO LIST
- Create TESTS!!
- Clean moska.py, possibly add reading the configurations from file.
- Add MoskaGameResult class
- Create a bot, that calculates the expected value for each playable combination, and chooses the best
- Create a Play -folder containing files related to playing
- Add coefficient class that can be used as a component of a MoskaPlayer
- Improve play to self, play to target, deck lift fall method, choose move (Calculate a score for each play and choose the play with the highest score)
- OpenAI gym requires: reset(), step(), __init__(), 
- Add methods to CardMonitor, to get all cards in deck