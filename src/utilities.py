from typing import List, Tuple, Dict
import random


def matrix_multiplication(vector_1=List, vector_2=List)->List:
    score=0
    for i in range(len(vector_1)):
        value=vector_1[i]*vector_2[i]
        score=score+value
    return score

def split_data(dt:List) ->Tuple[List,List,List]:
    """
    split data into train and validation and test data. We make a 90,5,5 split.
    :param dt: The list of our retrieved and processed data
    :return: total training, validation and test data.
    """
    total_train_data_len = int(len(dt) * 0.9)
    total_validation_data_len = int(len(dt) * 0.05)
    total_test_data_len = len(dt) - (total_train_data_len + total_validation_data_len)
    total_train_data = dt[:total_train_data_len]
    total_validation_data = dt[total_train_data_len:total_train_data_len + total_validation_data_len]
    total_test_data = dt[total_validation_data_len + total_train_data_len:]
    return total_train_data, total_validation_data, total_test_data

def make_batches(data:List, batch_size:int) -> List:
    per_batch = []
    list_with_batch = []
    for e in data:
        per_batch.append(e)
        if len(per_batch) == batch_size:
            list_with_batch.append(per_batch)
            per_batch = []
    return list_with_batch


if __name__ == '__main__':
    for i in range(1000):
        l1=[random.randint(1, 999) for iter in range(50)]
        l2=[random.randint(1, 999) for iter in range(50)]
        score=matrix_multiplication(l1,l2)
        a = np.array(l1)
        b = np.array(l2)
        npscore=np.matmul(a, b)
        assert score==npscore