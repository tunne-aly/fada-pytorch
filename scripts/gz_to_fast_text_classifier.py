import sys, csv, re, gzip
from os import listdir, sep

splitter = re.compile(r"[?!.]")

data_size = 0

def read_sentences(file_path):
  global data_size
  print('reading sentences from {}'.format(file_path))
  g = gzip.open(file_path, 'r')
  for l in g:
    dicti =  eval(l)
    grade = dicti['overall']
    line = dicti['reviewText']
    for para in line.splitlines():
      if para:
        data_size += 1
        yield grade, [re.sub("[^a-öA-Ö ']", '', l).strip().lower() for l in para.split()]

def get_sentences(data_path):
    ls = []
    for p in listdir(data_path):
        if p.endswith(".gz"):
            for sent in read_sentences("{}{}{}".format(data_path, sep, p)):
              yield sent

if len(sys.argv) < 2:
  print('Please give path to data file as argument')
  exit()

train_file = "amazon_trinary_train.txt"
test_file = "amazon_trinary_test.txt"

train = open(train_file, 'w')
test = open(test_file, 'w')

i = 0
for s in get_sentences(sys.argv[1]):
  grade, sentence = s
  if (sentence == ''):
    continue
  if grade == 3:
    grade = 0
  elif grade < 3:
    grade = -1
  else:
    grade = 1
  label = '__label__' + str(grade)
  if i <= 0.9 * 2114197:
    train.write(label + ' ' + ' '.join(sentence) + "\n")
  else:
    test.write(label + ' ' + ' '.join(sentence) + "\n")
  i += 1
print(data_size)
test.close()
train.close()
