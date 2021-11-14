import random
import re

ending = [".", "...", ";", "!", "?"]

def read_corpus(filename):
    f = open(filename, 'r',encoding='UTF-8')
    lines = f.readlines()

    n = len(lines)
    i = 0

    word = []
    POS = []

    while i < n:
        if (lines[i][2:6] == 'text' and lines[i-1][2:6] == 'sent') or\
             (lines[i][2:6] == 'sent' and lines[i-1][2:8] == 'newdoc'):

            i += 1
            while i < n:
                line = lines[i].strip('\n').split('\t')
                if line[0] == '':
                    word.append(".")
                    POS.append("PUNCT")
                    i += 1
                    break
                else:
                    word.append(line[1])
                    POS.append(line[3])
                    i += 1
        else:
            i += 1

    if word[-1] not in ending:  # ajout à la dernière phrase
        word.append(".")
        POS.append("PUNCT")

    return word, POS


def make_repetition(train_word, test_word, test_POS):

    word_rep = []
    POS_rep = []

    word_no_rep = []
    POS_no_rep = []

    train_set = set(train_word)

    for i, word in enumerate(test_word):
        if word in train_set:
            word_rep.append(word)
            POS_rep.append(test_POS[i])
        else:
            word_no_rep.append(word)
            POS_no_rep.append(test_POS[i])

    word_no_rep.append(".")
    POS_no_rep.append("PUNCT")

    print(len(test_POS))
    print(len(word_rep))
    print(len(word_no_rep))

    write("repeated_gsd_test.txt", word_rep, POS_rep)
    write("non_repeated_gsd_test.txt", word_no_rep, POS_no_rep)

def write(filename, word, POS):
    f = open(filename,'w',encoding='UTF-8')
    n = len(word)
    for i in range(n):
        f.write(str(word[i]))
        f.write('\t')
        f.write(str(POS[i]))
        f.write('\n')
    f.close()


if __name__ == "__main__":

    fr_gsd_train = "fr_gsd-ud-train.txt"
    fr_gsd_test = "fr_gsd-ud-test.txt"

    fr_spoken_train = "fr_spoken-ud-train.txt"
    fr_spoken_test = "fr_spoken-ud-test.txt"

    fr_old_train = "fro_srcmf-ud-train.txt"
    fr_old_test = "fro_srcmf-ud-test.txt"

    #old_test = "old_test.txt"

    filename = fr_gsd_train
    word_train, POS_train = read_corpus(filename)

    filename = fr_gsd_test
    word_test, POS_test = read_corpus(filename)

    make_repetition(word_train, word_test, POS_test)

    #write("fr_spoken_train.txt", word_train, POS_train)
    #write("fr_spoken_test.txt", word_test, POS_test)
