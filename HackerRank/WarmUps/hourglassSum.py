#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the hourglassSum function below.
def hourglassSum(arr):
    '''returns the sum of hour glass (roman numeral I) elements of an array'''
    maxHourglass = -9*7
    print(len(arr[0]))
    
    for i in range(len(arr)-2):
        for j in range(len(arr)-2):
            print(arr[j][i:i+3])
            print(arr[j+1][i+1])
            print(arr[j+2][i:i+3])

            hourglass = sum(arr[j][i:i+3])+arr[j+1][i+1]+sum(arr[j+2][i:i+3])
            if hourglass > maxHourglass:
                maxHourglass = hourglass
    return maxHourglass

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    arr = []

    for _ in range(6):
        arr.append(list(map(int, input().rstrip().split())))

    result = hourglassSum(arr)

    fptr.write(str(result) + '\n')

    fptr.close()
