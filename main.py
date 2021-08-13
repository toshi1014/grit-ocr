from handle_image import HandleImage
from read_contents import ReadContents
import numpy as np

img_path = "sample.png"
min_line_length = 40
row, column = [7, 3]
read_contents = ReadContents(img_path, min_line_length, row, column)

test_label_list = np.array([
    ["Hello", "Bonjour", "foo"],
    ["World", "Hola", "bar"],
    ["Nice", "Sawadee", "baz"],
    ["to", "Respect", "spam"],
    ["Meet", "my", "ham"],
    ["You", "Authoritah", "eggs"],
    ["2021", "12:00", "4649"],
])

ignore_idx_list = [
]

read_contents.test(test_label_list, ignore_idx_list)
print(read_contents.read())