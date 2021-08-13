import cv2
import numpy as np
import matplotlib.pyplot as plt
from pylsd.lsd import lsd
import itertools
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


class HandleImage():
    def __init__(self, img_path, min_length_, row_, column_):
        self.img = cv2.imread(img_path)
        self.min_length = min_length_
        self.row = row_
        self.column = column_

        self.get_dot_of_line()
        self.get_cluster_list()
        self.get_grid_list()


    def get_dot_of_line(self):
        grayed = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        grayed_inv = cv2.bitwise_not(grayed)        ## color inversion: make easy to find lines
        line_list = lsd(grayed_inv)

        self.dot_list = []
        cnt = 0
        img = self.img.copy()

        for line in line_list:
            x1, y1, x2, y2, _ = list(map(lambda x: int(x), line))
            length = np.sqrt(sum((np.array([x1, y1]) - np.array([x2, y2]))**2))
            if length > self.min_length:
                self.dot_list += [[x1, y1], [x2, y2]]
                cnt += 1
                img = cv2.line(img, (x1, y1), (x2, y2), (0,0,255), 1)

        cv2.imwrite("detected_line.png", img)
        print("line:", cnt)
        print("dot:", len(self.dot_list))


    def get_cluster_list(self):
        kmeans = KMeans(n_clusters=(self.row+1)*(self.column+1))
        kmeans.fit(self.dot_list)
        labels = kmeans.labels_

        raw_cluster_list = []
        for i in range((self.row+1)*(self.column+1)):
            cluster_now = []
            for idx, label in enumerate(labels):
                if label == i:
                    cluster_now.append(np.array(self.dot_list[idx]))
            raw_cluster_list.append(cluster_now)

        cluster_center_list = [sum(cluster)/len(cluster) for cluster in raw_cluster_list]
        clm_ordered_cluster_center_list = sorted(cluster_center_list, key=lambda x: x[0])
        clm_list = [clm_ordered_cluster_center_list[i:i+(self.row+1)] for i in range(len(clm_ordered_cluster_center_list))[::(self.row+1)]]
        self.ordered_cluster_center_list = \
            list(itertools.chain.from_iterable(
                [sorted(clm, key=lambda clm: clm[1]) for clm in clm_list]
            ))

        ordered_cluster_list = []
        pred_label_list = kmeans.predict(self.ordered_cluster_center_list)

        self.cluster_list = [raw_cluster_list[i] for i in pred_label_list]

        # DEBUG:
        # fig = plt.figure()
        # ax1 = fig.add_subplot(1,1,1)
        # for c in self.ordered_cluster_center_list:
        #     ax1.scatter(c[0], c[1], color="white")
        # for i in range(56):
        #     ax1.annotate(i, self.ordered_cluster_center_list[i])
        # for cluster_now in self.cluster_list:
        #     ax1.scatter([xy[0] for xy in cluster_now], [xy[1] for xy in cluster_now])
        # ax1.xaxis.tick_top()
        # ax1.axis("square")
        # ax1.set_aspect("equal")
        # plt.gca().invert_yaxis()
        # plt.show()


    def get_grid_center(self, idx1, idx2, idx3, idx4):
        sum = np.array([0.,0.])
        for idx in [idx1, idx2, idx3, idx4]:
            sum += self.ordered_cluster_center_list[idx]
        return sum/4


    def get_nearest(self, cluster):
        abs_list = list(map(lambda x: sum((abs(x - self.grid_center))**2), cluster))
        min_idx = abs_list.index(min(abs_list))
        return min_idx


    ## return nearest 2 dots from grid_center
    def get_nearest_2(self, cluster):
        min_idx = self.get_nearest(cluster)
        nearest = cluster[min_idx]
        cluster.pop(min_idx)

        snd_min_idx = self.get_nearest(cluster)
        snd_nearest = cluster[snd_min_idx]

        lower, upper = sorted([nearest, snd_nearest], key=lambda x: x[1])
        return lower, upper


    def get_rectangle_sides(self):
        top_left_lower, top_left_upper = self.get_nearest_2(self.top_left_cluster)
        top_right_lower, top_right_upper = self.get_nearest_2(self.top_right_cluster)
        bottom_left_lower, bottom_left_upper = self.get_nearest_2(self.bottom_left_cluster)
        bottom_right_lower, bottom_right_upper = self.get_nearest_2(self.bottom_right_cluster)

        top_side = [top_left_lower, top_right_lower]
        left_side = [top_left_upper, bottom_left_lower]
        right_side = [top_right_upper, bottom_right_lower]
        bottom_side = [bottom_left_upper, bottom_right_upper]

        return top_side, left_side, right_side, bottom_side


    def get_linear_function(self, line):        ## y = ax + b
        diff = line[1] - line[0]
        ## if y = const
        if diff[0] == 0:
            a, b = None, None
        else:
            a = diff[1] / diff[0]
            b = line[0][1] - a*line[0][0]

        return a, b

    def get_cross_point(self, line1, line2):
        a1, b1 = self.get_linear_function(line1)
        a2, b2 = self.get_linear_function(line2)

        if a1 == None:
            x = line1[0][0]
            y = a2*x + b2
        elif a2 == None:
            x = line2[0][0]
            y = a1*x + b1
        else:
            x = (b2 - b1) / (a1 - a2)
            y = (a1*b2 - b1*a2) / (a1 - a2)
        return int(x), int(y)


    def get_vertices(self, sides):
        top_side, left_side, right_side, bottom_side = sides

        top_left = self.get_cross_point(top_side, left_side)
        top_right = self.get_cross_point(top_side, right_side)
        bottom_left = self.get_cross_point(bottom_side, left_side)
        bottom_right = self.get_cross_point(bottom_side, right_side)

        return [top_left, top_right, bottom_left, bottom_right]


    def get_grid_list(self):
        self.grid_list = []

        for i in range((self.row+1)*self.column):
            ## if lowest row
            if ((i+1) % (self.row+1)) == 0:
                continue

            self.top_left_cluster = self.cluster_list[i]
            self.top_right_cluster = self.cluster_list[i+(self.row+1)]
            self.bottom_left_cluster = self.cluster_list[i+1]
            self.bottom_right_cluster = self.cluster_list[i+(self.row+1)+1]

            self.grid_center = self.get_grid_center(i, (i+(self.row+1)), (i+1), (i+(self.row+1)+1))

            self.grid_list.append(self.get_vertices(self.get_rectangle_sides()))

        # DEBUG:
        # img = self.img.copy()
        # for grid in self.grid_list:
        #     for vertex in grid:
        #         cv2.circle(img, vertex, radius=0, color=(0,0,255))
        # cv2.imwrite("vertex.png", img)


    def get_transformed_grid_img(self, grid):
        original_vertices = np.float32(grid)

        x_list = [xy[0] for xy in grid]
        y_list = [xy[1] for xy in grid]
        x_min, x_max = min(x_list), max(x_list)
        y_min, y_max = min(y_list), max(y_list)

        transfered_vertices = np.float32([[0, 0], [x_max-x_min, 0], [0, y_max-y_min], [x_max-x_min, y_max-y_min]])

        img = self.img.copy()
        M = cv2.getPerspectiveTransform(original_vertices, transfered_vertices)
        transformed_grid_img = cv2.warpPerspective(img, M, (x_max-x_min, y_max-y_min))

        return transformed_grid_img