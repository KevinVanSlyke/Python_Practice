#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'jumpingOnClouds' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY c as parameter.
#

def jumpingOnClouds(c):
    '''we can move 1 or 2 indices forward, only to i's where c[i] = 0'''
    i = 0
    jumps = 0
    while i < len(c)-2:

        if c[i+2] == 0:
            i = i + 2
        else:
            i = i + 1
        jumps = jumps + 1
    if i != len(c)-1:
        jumps = jumps + 1
    return jumps

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    c = list(map(int, input().rstrip().split()))

    result = jumpingOnClouds(c)

    fptr.write(str(result) + '\n')

    fptr.close()
