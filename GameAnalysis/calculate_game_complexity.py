import os
import matplotlib.pyplot as plt
import pandas as pd
import math


def read_data(filename):
    deck_list = []
    nmoves_list = []
    with open(filename, 'r') as f:
        for line in f:
            if "NMOVES" not in line:
                continue
            line = line.split("NMOVES: ")[1].strip()
            data = line.split(" , ")
            deck = int(data[0])
            nmoves = int(data[1])
            deck_list.append(deck)
            nmoves_list.append(nmoves)
    return deck_list, nmoves_list

if __name__ == "__main__":
    # In 500 games, the total number of moves was 133959
    # The average number of moves per game is hence 267.918
    # From this data, we see the average number of possible moves is 3.94
    # This means, that each game can be played in 3.94^267.918 = 3.5*10^159 ways
    # 
    # The average state-space complexity is hence branching_factor^depth = 3.94^267.918 = 3.5*10^159
    # Here the branching factor is the average number of possible moves in a state.
    # The number of reasonable moves is probably much lower.
    # The branching factor DOES account for symmetrical moves.
    #
    # The number of possible initial states is the set of different hands dealt to players,
    # and the number of different orders for the remaning cards.
    # This can be calculated ncr(52,6) + ncr(46,6) + ncr(40,6) + ncr(34,6) + 28! = 3* 10^29
    # This doesnt take symmetries, or the triumph card in to account.
    #
    #
    #
    #
    #path = "./Benchmark3/"
    complexities = []
    players = range(2,9)
    for nplayers in players:
        path = "./Benchmark3-{}PL/".format(nplayers)
        game_datas = []
        # Loop through files
        for file in os.listdir(path):
            if not os.path.isfile(path + file):
                print("Not a file:",file)
                continue
            deck_list, nmoves_list = read_data(path + file)
            if deck_list and nmoves_list:
                game_datas.append(pd.DataFrame({'deck': deck_list, 'nmoves': nmoves_list}))
        # remove all rows with just 1 move
        game_datas = [game_data[game_data.nmoves > 1] for game_data in game_datas]

        # Calculate average number made moves per game
        n_made_moves = [len(game_data) for game_data in game_datas]
        avg_n_made_moves = (sum(n_made_moves)*nplayers)/len(n_made_moves)
        print("Average number of moves per game:",avg_n_made_moves)

        # Average number of possible moves per move
        n_possible_moves = [game_data.nmoves.mean() for game_data in game_datas]
        avg_nmoves = sum(n_possible_moves)/len(n_possible_moves)
        print("Average number of possible moves per move:",sum(n_possible_moves)/len(n_possible_moves))

        # Game tree complexity
        gtc = avg_nmoves**avg_n_made_moves
        print(f"Average game tree complexity of {nplayers} player games:",gtc)
        # in log10
        complexities.append(math.log10(gtc))
    fig, ax = plt.subplots()
    ax.plot(players, complexities)
    ax.set_xlabel(f"Number of players", fontsize=14)
    ax.set_ylabel(f"Game tree complexity (log10)", fontsize=14)
    ax.set_title(f"Game tree complexity calculated from 500 games", fontsize=14)
    plt.show()

    
            