import csv
import sys
from sklearn.model_selection import StratifiedShuffleSplit
ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)


with open("vcb_segmented.txt", "r") as f:
  reader = csv.reader(f, delimiter="\t")
  lines = []
  for line in reader:
      if sys.version_info[0] == 2:
          line = list(unicode(cell, 'utf-8') for cell in line)
      lines.append(line)


data = np.array([line[0] for line in lines])
intents = np.array([line[1] for line in lines])

for train_index, test_index in ss.split(data, intents):
  train_data = data[train_index]
  test_data = data[test_index]
  train_intent = intents[train_index]
  test_intent = intents[test_index]

ss = StratifiedShuffleSplit(n_splits=1, test_size=0.125, random_state=0)
for train_index, test_index in ss.split(train_data, train_intent):
  val_data = train_data[test_index]
  train_data = data[train_index]
  val_intent = train_intent[test_index]
  train_intent = train_intent[train_index]


with open('train.tsv', 'w') as f:
  f.write('text\tlabel\n')
  for t in train:
    f.write(t[0] + '\t' + t[1] + '\n')

with open('test.tsv', 'w') as f:
  f.write('text\tlabel\n')
  for t in test:
    f.write(t[0] + '\t' + t[1] + '\n')

with open('val.tsv', 'w') as f:
  f.write('text\tlabel\n')
  for t in val:
    f.write(t[0] + '\t' + t[1] + '\n')
