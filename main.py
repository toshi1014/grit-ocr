from grit_ocr import *
import numpy as np

img_path = "img/sample.png"
min_line_length = 40        ## minimum length of line to be detected
row, column = [7, 3]        ## 7x3 grid
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

print("\n\ttest")
read_contents.test(test_label_list, check_all_vertices=True, export_img=True)

print("==============================")
print("\n\tcontents\n")
print(read_contents.read())
print()