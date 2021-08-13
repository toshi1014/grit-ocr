import cv2
import pyocr
import pyocr.builders
from PIL import Image
from handle_image import  HandleImage
import numpy as np
import os


class ReadContents(HandleImage):
    def __init__(self, img_path, min_length_, row_, column_):
        super().__init__(img_path, min_length_, row_, column_)


    def read_grid(self, grid_img):
        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            raise Exception("No ORC Found")
        tool = tools[0]

        pil_img = Image.fromarray(grid_img)     ## from np.array (opencv) to PIL.Image
        content = tool.image_to_string(
            pil_img, lang="eng", builder=pyocr.builders.TextBuilder(tesseract_layout=6)
        )

        return content


    def test(self, test_label_arr, export_grid_img=False):
        if export_grid_img:
            try:
                os.mkdir("grid_img")
            except:
                pass

        correct = 0
        sum = 0
        print("\nidx\tpredict\tlabel\tis_correct")

        for idx, grid in enumerate(self.grid_list):
            transformed_grid_img = self.get_transformed_grid_img(grid)
            content = self.read_grid(transformed_grid_img)

            row = sum % self.row
            column = sum // self.row
            label = str(test_label_arr[row][column])

            if export_grid_img:
                filename = "grid_img/" + str(row) + "-" + str(column) + ".png"
                cv2.imwrite(filename, transformed_grid_img)

            if content == label:
                correct += 1
                is_correct = "o"
            else:
                is_correct = "x"
            sum += 1

            print(str(idx) + "\t" + content + "\t" + label + "\t" + is_correct)

        print("\naccuracy:", correct/sum, "\n")


    def read(self):
        content_list = []
        cnt = 0
        for grid in self.grid_list:
            transformed_grid_img = self.get_transformed_grid_img(grid)
            cnt += 1
            content = self.read_grid(transformed_grid_img)
            content_list.append(content)

        clm_row_arr = np.reshape(content_list, (self.column, self.row))
        return np.transpose(clm_row_arr)