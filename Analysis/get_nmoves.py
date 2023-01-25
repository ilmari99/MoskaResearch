import matplotlib.pyplot as plt
import pandas as pd


def read_data(filename):
    deck_list = []
    nmoves_list = []
    with open(filename, 'r') as f:
        for line in f:
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
    
    filename = "nmoves.txt"
    deck_list, nmoves_list = read_data(filename)
    df = pd.DataFrame({'deck': deck_list, 'nmoves': nmoves_list})
    print("Data:\n",df.describe())
    df = df[ df["nmoves"]>1 ]
    print("Data with nmoves=1 removed:\n",df.describe())
    
    plt.hist(df["nmoves"], bins=200)
    plt.figure()
    plt.scatter(df["deck"], df["nmoves"])
    df = df.groupby('deck').mean()
    print("Moves per deck:\n",df)
    plt.figure()
    plt.scatter(df.index, df["nmoves"])
    plt.show()
            