import sys, csv, re, gzip
from os import listdir, sep

splitter = re.compile(r"[?!.]")

def read_finnish_reviews_from_file(file_path):
    with open(file_path) as infile:
        reader = csv.reader(infile)
        for line in reader:
          yield [re.sub("[^a-öA-Ö ']", '', w).strip().lower() for w in line[1].split()]


def read_sentences(file_path):
  print('reading sentences from {}'.format(file_path))
  g = gzip.open(file_path, 'r')
  for l in g:
    dicti =  eval(l)
    grade = dicti['overall']
    line = dicti['reviewText']
    for para in line.splitlines():
      if para:
          yield [re.sub("[^a-öA-Ö ']", ' ', l).strip().lower() for l in para.split()]

def read_sentences_from_json(file_path):
  print('reading sentences from {}'.format(file_path))
  f = open(file_path, 'r')
  for l in f:
    dicti =  eval(l)
    grade = dicti['overall']
    line = dicti['reviewText']
    for para in line.splitlines():
      if para:
          yield [re.sub("[^a-öA-Ö ']", ' ', l).strip().lower() for l in para.split()]

def get_sentences(data_path):
    for p in listdir(data_path):
        if p.endswith(".gz"):
            for sent in read_sentences("{}{}{}".format(data_path, sep, p)):
                yield sent
        elif p.endswith(".csv"):
            for sent in read_finnish_reviews_from_file("{}{}{}".format(data_path, sep, p)):
                yield sent
        elif p.endswith(".json"):
            for sent in read_sentences_from_json("{}{}{}".format(data_path, sep, p)):
                yield sent


if len(sys.argv) < 2:
  print('Please give path to data file as argument')
  exit()

sentences = "amazon_electronics_fast_text_sentences"
words = "amazon_electronics_fast_text_words"
sentence_file = open(sentences, 'w')
word_file = open(words, 'w')

print("Storing vectors to {}".format(sentences))
print("Storing words to {}".format(words))
s = 0
w = 0
word_set = set()
for review in get_sentences(sys.argv[1]):
    s += 1
    sentence_file.write(" ".join(review) + "\n")
    for word in review:
        if not word == '':
            w += 1
            word_set.add(word)
word_file.write("\n".join(word_set))
print('Wrote {} sentences with {} distinct words'.format(s, w))

