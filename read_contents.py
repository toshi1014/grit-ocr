import cv2
import pyocr
import pyocr.builders
from PIL import Image
from handle_image import  HandleImage
import numpy as np
import os
import matplotlib.pyplot as plt


DIRNAME = "dst"


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


    def test(self, test_label_arr, check_all_vertices=False, export_img=False):
        if check_all_vertices:
            fig = plt.figure()
            ax1 = fig.add_subplot(1,1,1)
            for xy in self.ordered_cluster_center_list:
                ax1.scatter(xy[0], xy[1], color="white")
            for i in range((self.row+1)*(self.column+1)):
                ax1.annotate(i, self.ordered_cluster_center_list[i])
            for cluster_now in self.cluster_list:
                ax1.scatter([xy[0] for xy in cluster_now], [xy[1] for xy in cluster_now])
            ax1.xaxis.tick_top()
            ax1.set_aspect("equal")
            plt.gca().invert_yaxis()
            plt.show()

        if export_img:
            try:
                os.mkdir(DIRNAME)
            except:
                pass

            cv2.imwrite(DIRNAME + "/detected_line.png", self.detected_line_img)

            vertices_img = self.img.copy()
            for grid in self.grid_list:
                for vertex in grid:
                    cv2.circle(vertices_img, vertex, radius=0, color=(0,0,255))
            cv2.imwrite(DIRNAME + "/vertices.png", vertices_img)

        correct = 0
        sum = 0
        print("\nidx\tpredict\tlabel\tis_correct\n")

        for idx, grid in enumerate(self.grid_list):
            transformed_grid_img = self.get_transformed_grid_img(grid)
            content = self.read_grid(transformed_grid_img)

            row = sum % self.row
            column = sum // self.row
            label = str(test_label_arr[row][column])

            if export_img:
                filename = DIRNAME + "/" + str(row) + "-" + str(column) + ".png"
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