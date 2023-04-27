
""" Analyze log files from a nested folder structure."""
import os
import re
import json
import matplotlib.pyplot as plt

class Result:
    def __init__(self, path):
        self.path = path
        print(self.path)
        self.username = self.get_username()
        self.game_id = self.get_game_id()
        self.user_rank = self.get_user_rank()
    
    def get_username(self):
        self.username = self.path.split("/")[2]
        return self.username
    
    def get_game_id(self):
        self.game_id = int(self.path.split("/")[-1].split("-")[1].split(".")[0])
        return self.game_id
    
    def get_user_rank(self):
        user_rank = None
        with open(self.path, "r") as f:
            # Read the file, until we find the ranking
            for line in f:
                if "INFO:#" in line and "NN2" not in line:
                    user_rank = int(line.split("#")[1][0])
                    break
        return user_rank


def get_log_files(folder):
    """ Return a list of all log files in the folder and its subfolders """
    log_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if "HumanGame" in file:
                log_files.append(os.path.join(root, file).replace("\\", "/"))
    return log_files

if __name__ == "__main__":
    # Get all log files
    log_files = get_log_files("./FireBaseLogs")
    # Analyze each log file
    results = []
    for log_file in log_files:
        result = Result(log_file)
        results.append(result)
    # Print the number of games with rank, and number of games without rank
    print("Games with rank:", len([r for r in results if r.user_rank is not None]))
    print("Games without rank:", len([r for r in results if r.user_rank is None]))

    # Print the average rank if there is a rank
    ranks = [r.user_rank for r in results if r.user_rank is not None]
    print("Average rank:", sum(ranks)/len(ranks))
    human_lost = len([r for r in results if r.user_rank == 4])
    # Print the portion of games, where the user has rank 4
    print("Humans lost games:", human_lost)
    print("Humans lost games (%):", human_lost/len(ranks))

    # Print the number of games per user
    users = {}
    for result in results:
        user = result.username
        if user not in users:
            users[user] = []
        users[user].append(result)
    for user, results in users.items():
        # Number of played games
        print(f"{user} played {len(results)} games")
        # Number of successful games
        suc_games = len([r for r in results if r.user_rank is not None])
        print(f"{user} finished {suc_games} games")
        if suc_games == 0:
            continue
        # Average rank
        ranks = [r.user_rank for r in results if r.user_rank is not None]
        print(f"{user} average rank: {sum(ranks)/len(ranks)}")
        # Portion of games, where the user has rank 4
        human_lost = len([r for r in results if r.user_rank == 4])
        print(f"{user} lost games: {human_lost}")
        print(f"{user} lost games (%): {human_lost/len(ranks)}")
        print()

