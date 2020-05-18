from typing import List
import random


def matrix_multiplication(vector_1=List, vector_2=List)->List:
    score=0
    for i in range(len(vector_1)):
        value=vector_1[i]*vector_2[i]
        score=score+value
    return score

if __name__ == '__main__':
    for i in range(1000):
        l1=[random.randint(1, 999) for iter in range(50)]
        l2=[random.randint(1, 999) for iter in range(50)]
        score=matrix_multiplication(l1,l2)
        a = np.array(l1)
        b = np.array(l2)
        npscore=np.matmul(a, b)
        assert score==npscore