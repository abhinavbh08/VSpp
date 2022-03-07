import os
from collections import Counter
from tqdm import tqdm

# root_path =
root_path = "/home/abhinav/Documents/data/train"


def load_content(filepath):
    """Given a filepath, returns (content, classname), where content = [list of lines in file]"""
    with open(filepath, "r") as file:
        data = file.read().split("\n")
    data = data[:-1]
    _, file_extension = os.path.splitext(filepath)

    return data, file_extension[1:]


def load_data(root_path, nworkers=10):
    """Returns each data sample as a tuple (x, y), x = sequence of strings (i.e., syscalls), y = malware program class"""
    raw_data_samples = []
    for file_name in tqdm(os.listdir(root_path)):
        filepath = os.path.join(root_path, file_name)
        data, label = load_content(filepath)
        lst_initials = []
        for line in data:
            if line[0] == "#":
                continue
            line = line.split(" ")
            line.remove("|")
            if len(line) >= 4:
                lst_initials.extend(line[:4])
            else:
                lst_initials.extend(line[: len(line)])
        raw_data_samples.append((" ".join(lst_initials), label))

    return raw_data_samples


# load_data(root_path)

from collections import Counter

counts = Counter()
lines = [
    "This is the greatest show",
    "Yes this is",
    "Wow man you are on the greatest show",
    "Cool show this is",
]
for line in lines:
    counts.update(line.split(" "))

print(counts)

from sklearn.model_selection import train_test_split

data = [(1, "a"), (2, "b"), (3, "c"), (4, "d"), (5, "e")]
train, test = train_test_split(data, test_size=0.2)
print("abc")
print(train, test)
