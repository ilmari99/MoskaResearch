from collections import Counter
import os
import random
import tensorflow as tf

MISC_PARTS = list(range(0,5)) + list(range(161,170))
CARD_PARTS = list((n for n in range(0,430) if n not in MISC_PARTS))

def create_bitmap_dataset(paths, add_channel = False) -> tf.data.Dataset:
    """ Create a tf dataset from a folder of files"""
    global MISC_PARTS, CARD_PARTS
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    # Cards on table, vards in hand + players ready, players out, kopled

    file_paths = [os.path.join(path, file) for path in paths for file in os.listdir(path) if os.path.isdir(path)]

    print("Found {} files".format(len(file_paths)))

    dataset = tf.data.TextLineDataset(file_paths, num_parallel_reads=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: tf.strings.split(x, sep=","), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: (tf.strings.to_number(x[:-1], out_type=tf.int32), tf.strings.to_number(x[-1], out_type=tf.int32)), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    #dataset = dataset.map(lambda x,y : (tf.gather(x, CARD_PARTS),tf.gather(x,MISC_PARTS), y))

    #dataset = dataset.map(lambda cards, misc, y : _to_bitmap(cards,misc,y), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(to_bitmap_if_necessary, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.filter(lambda x,y: not tf.equal(y,-1))

    if add_channel:
        dataset = dataset.map(lambda x,y: (tf.expand_dims(x, axis=-1), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset

def to_bitmap_if_necessary(x, y):
    # If x has 430 elements, it is in the old format
    if tf.size(x) == 430:
        cards,misc,y = (tf.gather(x, CARD_PARTS),tf.gather(x,MISC_PARTS), y)
        #dataset = dataset.map(lambda x,y: ((tf.gather(x, CARD_PARTS),tf.gather(x,MISC_PARTS), y)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
        x,y = _to_bitmap(cards, misc, y)
    return x,y

@tf.function
def _to_bitmap(cards, misc, y):
    #card_tracker = cards[0:52].reshape([52//4, 4])
    # Read the cards_fall_dict in to 14x4 tensor
    card_tracker = tf.reshape(cards[0:52], [52//4, 4])
    # Filter rows where all values are -1
    card_tracker = card_tracker[tf.reduce_any(card_tracker != -1, axis=1)]
    # Get the most common highest index by getting the most common value
    index_with_highest_value = tf.argmax(card_tracker, axis=1, output_type=tf.int32)
    _, indices, counts = tf.unique_with_counts(index_with_highest_value)
    index_with_highest_value = indices[tf.argmax(counts)]
    # Create a one-hot vector
    trump_one_hot = tf.one_hot(index_with_highest_value, 4, dtype=tf.int32)
    # Also, find the target
    target_val = tf.reduce_max(misc[9:13])
    target_pid = tf.argmax(tf.equal(misc[9:13], target_val))
    tg_one_hot = tf.one_hot(target_pid, 4, dtype=tf.int32)
    # Update the misc vector, by setting the target pid to 1
    update_indices = tf.expand_dims([9 + target_pid], axis=0)
    update_values = tf.where(target_val == 2, tf.constant([1],dtype=tf.int32), tf.constant([0],dtype=tf.int32))
    misc = tf.tensor_scatter_nd_update(misc, update_indices, update_values)
    # Concatenate the one-hot vectors
    misc = tf.concat([misc, tg_one_hot, trump_one_hot], axis=0)
    # Mark each card as 1 if it is known, and 0 if it is unknown/not-in game
    cards = tf.where(cards == 0, 1, cards)
    cards = tf.where(cards == -1, 0, cards)
    cards = tf.where(cards > 0, 1, cards)

    # Deduce the player pid
    # Get the monitored (global) player cards, and the players cards
    card_rows = tf.reshape(cards, [cards.shape[0]//52, 52])[3:,:]
    # Duplicate the players cards, so the matrices can be added
    last_row_duplication = tf.reshape(tf.tile(card_rows[-1], [card_rows.shape[0]]), [card_rows.shape[0], card_rows.shape[1]])
    # Add the last row to each row by element
    card_row_add = tf.add(card_rows, last_row_duplication)
    # Now, the players pid is the index of the row that has 2,
    # because no other monitored cards row has same cards
    player_pid = tf.argmax(tf.reduce_any(tf.equal(card_row_add, 2), axis=1), output_type=tf.int32)
    # If player pid is 0, then either no row has 2, or the first row has 2
    # Check if the first row has 2 if player pid is 0
    if player_pid == 0 and tf.reduce_any(tf.equal(card_rows[0], 2)):
        tf.print("Cant deduce player pid")
        return tf.constant(-1), tf.constant(-1)
    # Add the player pid (one-hot) to the misc vector
    player_pid_one_hot = tf.one_hot(player_pid, 4, dtype=tf.int32)
    misc = tf.concat([misc, player_pid_one_hot], axis=0)
    data = tf.concat([misc, cards], axis=0)
    return data, y

if __name__ == "__main__":
    dataset = create_bitmap_dataset(["./HumanLogs/Vectors/"])#, "./Benchmark1/Vectors/"])
    #dataset = dataset.shuffle(1000)
    data_buffer = []
    file = "bitmaps.csv"
    cant_deduce_player = 0
    with open(file, "w") as f:
        for data in dataset.as_numpy_iterator():
            print(data)
            if data[-1] == -1:
                cant_deduce_player += 1
                continue
            data = data[0].tolist() + [data[1].tolist()]
            data_buffer.append(data)
            if len(data_buffer) > 1000:
                f.write("\n".join([",".join([str(d) for d in data]) for data in data_buffer]) + "\n")
                data_buffer = []
        print("Could not deduce player pid", cant_deduce_player, "times")
        f.write("\n".join([",".join([str(d) for d in data]) for data in data_buffer]))
            





