# moska
A repo for simulating moska games, and trying different play algorithms.

This is designed for simulations of moska games with different algorithms, and playing as a human can be quite tedious.

## TODO LIST
- Create TESTS!!
- Clean moska.py, possibly add reading the configurations from file.
- Fix when running multiple games, the log files are appended if usig the same name
- Add MoskaGameResult class
- Create a Play -folder containing files related to playing
- Add coefficient class that can be used as a component of a MoskaPlayer
- Improve play to self, play to target, deck lift fall method, choose move (Calculate a score for each play and choose the play with the highest score)
- Use scipy optimizing to find the coefficients, that minimize chance to lose, OR maximize average ranking.
- Maybe add a wrapper to play_move, that would log when the played hand becomes worse after playing a move
- Add methods to CardMonitor, to get all cards in deck