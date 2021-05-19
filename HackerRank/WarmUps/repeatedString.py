#!/bin/python3

import math
import os
import random
import re
import sys

#
# Complete the 'repeatedString' function below.
#
# The function is expected to return a LONG_INTEGER.
# The function accepts following parameters:
#  1. STRING s
#  2. LONG_INTEGER n
#

def repeatedString(s, n):
    '''return number of "a"s in the first n letters of an infinitely repeating s'''
    i = 0
    count = 0
    if len(s) <= n:
        aCount = s.count('a')
        count = aCount*math.floor(n/len(s))
        if n%len(s) != 0:
            count = count + s[:n%len(s)].count('a')
    else:
        count = s[:n%len(s)].count('a')
    return count

    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    n = int(input().strip())

    result = repeatedString(s, n)

    fptr.write(str(result) + '\n')

    fptr.close()
