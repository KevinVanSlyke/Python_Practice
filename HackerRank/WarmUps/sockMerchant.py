#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'sockMerchant' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY ar
#

def sockMerchant(n, ar):
    '''Return number of pairs'''
    sockTypes = []
    numPairs = 0
    for i in range(len(ar)):
        if sockTypes.count(ar[i]) == 0:
            sockTypes.append(ar[i])
            numPairs = numPairs + math.floor(ar.count(ar[i])/2)
        else:
            continue
    return numPairs

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    ar = list(map(int, input().rstrip().split()))

    result = sockMerchant(n, ar)

    fptr.write(str(result) + '\n')

    fptr.close()
