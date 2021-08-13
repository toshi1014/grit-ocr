from handle_image import HandleImage
from read_contents import ReadContents

img_path = "calender.png"
min_line_length = 40
row, column = [6, 7]
read_contents = ReadContents(img_path, min_line_length, row, column)

test_label_list = [
7, 14, 21, 28, 1, 8, 15, 22, 29, 2, 9, 16, 23, 30, 3, 10, 17, 24, 31,
4, 11, 18, 25, 5, 12, 19, 26, 6, 13, 20, 27
]

ignore_idx_list = [
    0, 1, 6, 12, 18, 24, 29, 30, 35, 36, 41
]

read_contents.test(test_label_list, ignore_idx_list)