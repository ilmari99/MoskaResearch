import os
import matplotlib.pyplot as plt
import ast

def get_histories_from_file(path):
    """ Get the history of the game from the log file """
    histories = []
    with open(path, "r") as f:
        # Each line contains a printed tensorflow history object
        # {'loss': [0.6471672654151917, 0.6374167203903198 ... 
        for line in f:
            line = line.strip()
            if line.startswith("{"):
                history = ast.literal_eval(line)
                histories.append(history)
    return histories

if __name__ == "__main__":
    histories = get_histories_from_file("histories.txt")
    
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    # Plot 4 images: Accuracy, loss, val_accuracy, val_loss
    accuracies = [h["accuracy"][-1] for h in histories]
    losses = [h["loss"][-1] for h in histories]
    val_accuracies = [h["val_accuracy"][-1] for h in histories]
    val_losses = [h["val_loss"][-1] for h in histories]
    
    ax[0, 0].plot(accuracies)   
    ax[0, 0].set_title("Accuracy")
    ax[0, 1].plot(losses)
    ax[0, 1].set_title("Loss")
    ax[1, 0].plot(val_accuracies)
    ax[1, 0].set_title("Validation accuracy")
    ax[1, 1].plot(val_losses)
    ax[1, 1].set_title("Validation loss")
    plt.show()
    
    