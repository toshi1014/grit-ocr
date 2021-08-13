import cv2
import pyocr
import pyocr.builders
from PIL import Image
from handle_image import  HandleImage


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


    def test(self, test_label_list, ignore_idx_list=[]):
        correct = 0
        sum = 0
        print("\nidx\tpredict\tlabel\tis_correct")

        for idx, grid in enumerate(self.grid_list):
            if idx in ignore_idx_list:
                continue

            transformed_grid_img = self.get_transformed_grid_img(grid)
            content = self.read_grid(transformed_grid_img)

            label = str(test_label_list[sum])

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
        for grid in self.grid_list:
            transformed_grid_img = self.get_transformed_grid_img(grid)
            content = self.read_grid(transformed_grid_img)
            content_list.append(content)

        return content_list