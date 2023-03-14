from collections import Counter
import os
import random
import tensorflow as tf


def create_tf_dataset(paths) -> tf.data.Dataset:
    """ Create a tf dataset from a folder of files"""
    if not isinstance(paths, (list, tuple)):
        try:
            paths = [paths]
        except:
            raise ValueError("Paths should must be a list of strings")
    # Cards on table, vards in hand + players ready, players out, kopled
    misc_parts = list(range(0,5)) + list(range(161,170))
    card_parts = list((n for n in range(0,430) if n not in misc_parts))
    print(f"Number of card parts" + str(len(card_parts)))
    print(f"Misc parts: {misc_parts}")
    file_paths = []
    for path in paths:
        if not os.path.isdir(path):
            raise ValueError(f"Path {path} is not a directory")
        file_paths += [os.path.join(path, file) for file in os.listdir(path)]
    print("Number of files: " + str(len(file_paths)))
    random.shuffle(file_paths)
    print("Shuffled files.")
    dataset = tf.data.TextLineDataset(file_paths)
    dataset = dataset.map(lambda x: tf.strings.split(x, sep=", "), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: (tf.strings.to_number(x[:-1],out_type=tf.float32), tf.strings.to_number(x[-1], out_type=tf.float32)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # Separate into card,misc, and label
    dataset = dataset.map(lambda x,y: ((tf.gather(x, card_parts),tf.gather(x,misc_parts), y)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset

if __name__ == "__main__":
    dataset : tf.data.Dataset = create_tf_dataset(["./Benchmark3/Vectors/","./Benchmark1/Vectors/"])
    #dataset : tf.data.Dataset = create_tf_dataset(["./HumanLogs/Vectors/"])
    output_file = "./HumanLogs/Bitmaps/bitmaps.csv"
    verbose = False
    cant_deduce_player = 0
    with open(output_file, "w") as f:
        data_buffer = []
        for cards,misc,y in dataset.as_numpy_iterator():
            card_tracker = cards[0:52].reshape([52//4, 4])
            # Filter rows where all values are -1
            card_tracker = card_tracker[tf.reduce_any(card_tracker != -1, axis=1)]
            index_with_highest_value = tf.argmax(card_tracker, axis=1, output_type=tf.int32)
            # Get the most common highest index by getting the most common value
            c = Counter(index_with_highest_value.numpy())
            index_with_highest_value = c.most_common(1)[0][0]
            # Create a one-hot vector
            trump_one_hot = tf.one_hot(index_with_highest_value, 4, dtype=tf.int32)
            # Also, find the target
            #target_val = max(misc[9:13])
            target_val = tf.reduce_max(misc[9:13])
            #target_pid = tf.argmax(misc[9:13], output_type=tf.int32)
            target_pid = tf.argmax(tf.equal(misc[9:13], target_val), output_type=tf.int32)
            tg_one_hot = tf.one_hot(target_pid, 4, dtype=tf.int32)
            #misc[9 + target_pid] = 1 if target_val == 2 else 1
            # Convert the misc index 9 + target_pid to 1 if target_val == 2 else 0
            #update_indices = tf.constant([[9 + target_pid]])
            update_indices = tf.expand_dims([9 + target_pid], axis=0)
            update_values = tf.where(target_val == 2, tf.constant([1],dtype=tf.int32), tf.constant([0],dtype=tf.int32))
            if verbose:
                print("Original misc",misc)
            misc = tf.tensor_scatter_nd_update(misc, update_indices, update_values)
            #misc = tf.convert_to_tensor(misc,dtype=tf.int32)
            misc = tf.concat([misc, tg_one_hot, trump_one_hot], axis=0)
            # Change all card 0 to 1
            # Change all card -1 to 0
            # change all cards > 0 to 1
            cards = tf.where(cards == 0, 1, cards)
            cards = tf.where(cards == -1, 0, cards)
            cards = tf.where(cards > 0, 1, cards)
            #card_rows = cards.reshape([cards.shape[0]//52, 52])
            card_rows = tf.reshape(cards, [cards.shape[0]//52, 52])
            # Multiply the last row card_rows.shape[0]
            last_row_duplication = tf.reshape(tf.tile(card_rows[-1], [card_rows.shape[0]]), [card_rows.shape[0], card_rows.shape[1]])
            # Add the last row to each row by element
            card_row_add = tf.add(card_rows, last_row_duplication)
            # Now, the row index between index 3 - 6 which has a value 2 somewhere, is the player
            player_pid = tf.argmax(tf.reduce_any(tf.equal(card_row_add[3:7], 2), axis=1), output_type=tf.int32)
            # argmax also returns 0, if no 2 is found, so we need to check if the player pid is 0
            # Check if the player pid is 0, if yes, check if there is a 2 in the row
            if player_pid == 0:
                has2 = tf.argmax(card_row_add[3+player_pid], output_type=tf.int32)
                if has2 == 0:
                    if card_row_add[3+player_pid][0] != 2:
                        cant_deduce_player += 1
                        continue
            player_pid_one_hot = tf.one_hot(player_pid, 4, dtype=tf.int32)
            misc = tf.concat([misc, player_pid_one_hot], axis=0)
            #print("Card rows",card_rows)
            #print("Player pid",player_pid)
            data = list(misc.numpy()) + list(cards.numpy()) + [y]
            data_buffer.append(data)
            if verbose:
                print("misc_after",misc)
                print("data_misc", data[417:417+23])
                print("misc shape", misc.shape)
                print("cards shape",cards.shape)
                print("data shape", len(data))
            print("Misc shape", misc.shape)
            if len(data_buffer) == 100:
                f.write("\n".join([",".join([str(d) for d in data]) for data in data_buffer]) + "\n")
                data_buffer = []
        print("Could not deduce player pid", cant_deduce_player, "times")
        f.write("\n".join([",".join([str(d) for d in data]) for data in data_buffer]))


        