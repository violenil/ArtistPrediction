from typing import List, Tuple


def split_data(dt: List) -> Tuple[List, List, List]:
    """
    split data into train and validation and test data. We make a 80,10,10 split.
    :param dt: The list of our retrieved and processed data
    :return: total training, validation and test data.
    """
    total_train_data_len = int(len(dt) * 0.8)
    print(len(dt), total_train_data_len)
    total_validation_data_len = int(len(dt) * 0.1)
    print(total_validation_data_len)
    total_test_data_len = len(dt) - (total_train_data_len + total_validation_data_len)
    print(total_test_data_len)
    total_train_data = dt[:total_train_data_len]
    total_validation_data = dt[total_train_data_len:total_train_data_len+total_validation_data_len]
    total_test_data = dt[total_validation_data_len+total_train_data_len:]
    return total_train_data, total_validation_data, total_test_data

def make_batches(data:List,bch_sz:int)->list:
    """
    We make batches of the training data and validation data according to the batch size. It simply is another list inside
    the list of songs where each batch size number of songs are keep in another list. We discard the excess data in the end
    that does not form a batch.
    :param data:
    :param bch_sz:
    :return: list of songs with batches. [songs[batch1[song data 1],[2],[3]],[[],[],[]],......[batch n[],[],[]]]
    """
    a = []
    list_with_batch = []
    for e in data:
        a.append(e)
        if len(a) == bch_sz:
            list_with_batch.append(a)
            a = []
    return list_with_batch



if __name__ == "__main__":
    """
    Split 80-20
    Create batches and return
    """
    dt = [[1, [5], [6]], [2, [4], [7]], [4, [6], [5]], [3, [6], [4]], [7, [8], [9]], [2, [6], [5]], [1, [2], [4]],
          [2, [4], [7]], [4, [6], [5]], [3, [6], [4]], [7, [8], [9]], [2, [6], [5]]]
    train_dt, validation_dt, test_dt = split_data(dt=dt)
    # print(train_dt, '\n', validation_dt, '\n', test_dt)
    batch_size = 2
    list_of_train_batches = make_batches(data=train_dt, bch_sz=batch_size)
    print(list_of_train_batches)