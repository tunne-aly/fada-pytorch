import sys, csv, re, gzip
from os import listdir, sep

splitter = re.compile(r"[?!.]")

def read_sentences(file_path):
  print('reading sentences from {}'.format(file_path))
  g = gzip.open(file_path, 'r')
  for l in g:
    dicti =  eval(l)
    grade = dicti['overall']
    line = dicti['reviewText']
    for para in line.splitlines():
      if para:
          yield grade, [re.sub("[^a-öA-Ö ']", ' ', l).strip().lower() for l in para.split()]

def get_sentences(data_path):
    for p in listdir(data_path):
        if p.endswith(".gz"):
            for sent in read_sentences("{}{}{}".format(data_path, sep, p)):
                yield sent

if len(sys.argv) < 2:
  print('Please give path to data file as argument')
  exit()

output_file_name = "amazon_reviews.txt"
output_file = open(output_file_name, 'w')

print("Storing reviews to {}".format(output_file))
i = 0
for review in get_sentences(sys.argv[1]):
    grade = review[0]
    text = review[1]
    data = (grade, " ".join(text))
    output_file.write(str(data) + "\n")
    i += 1
print('Wrote {} sentences to {}'.format(i, output_file_name))
