import sys

def duplicate_encode(word):
    """Replace single use letters with '(' and multi use with ')'"""
    lInd = []
    rInd = []
    for i in range(len(word)):
        if word[i] == '(':
            lInd.append(i)
        elif word[i] == ')':
            rInd.append(i)
    
    new_word = word
    for x in word:
        if x != ('(' or ')'):
            if (word.isalpha() and (word.count(x)+word.swapcase().count(x)) == 1) or (not word.isalpha() and (word.count(x) == 1)):
                new_word = new_word.replace(x,'(')
                new_word = new_word.replace(x.swapcase(),'(')
            else:
                new_word = new_word.replace(x,')')
                new_word = new_word.replace(x.swapcase(),')')

    new_word = list(new_word)
    if len(lInd) > 1:
        for i in lInd:
            new_word[i] = ')'

    if len(rInd) == 1:
        new_word[rInd[0]] = '('

    return str(new_word)

if __name__ == "__main__":
    duplicate_encode(sys.argv[1])