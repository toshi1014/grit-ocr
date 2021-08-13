import cv2
import pyocr
import pyocr.builders
from PIL import Image
from handle_image import  HandleImage


class ReadContents(HandleImage):
    def __init__(self, img_path, min_length_, row_, column_):
        super().__init__(img_path, min_length_, row_, column_)
        self.read()


    def read_grid(self, grid_img):
        tools = pyocr.get_available_tools()
        if len(tools) == 0:
            raise Exception("No ORC Found")
        tool = tools[0]

        pil_img = Image.fromarray(grid_img)
        content = tool.image_to_string(
            pil_img, lang="eng", builder=pyocr.builders.TextBuilder(tesseract_layout=6)
        )

        return content


    def read(self):
        test_label_list = [
            7, 14, 21, 28, 1, 8, 15, 22, 29, 2, 9, 16, 23, 30, 3, 10, 17, 24, 31,
            4, 11, 18, 25, 5, 12, 19, 26, 6, 13, 20, 27
        ]
        correct = 0
        sum = 0

        for idx, grid in enumerate(self.grid_list):
            if (idx % 6) == 0:
                continue
            if idx in [1, 29, 35, 41]:
                continue

            transformed_grid_img = self.get_transformed_grid_img(grid)
            content = self.read_grid(transformed_grid_img)

            if content == str(test_label_list[sum]):
                correct += 1
                is_correct = "o"
            else:
                is_correct = "x"
            sum += 1

            print(str(idx) + "\t" + content + "\t" + is_correct)

        print("accuracy:", correct/sum)