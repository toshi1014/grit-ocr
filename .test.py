import cv2
import pyocr
import pyocr.builders
from PIL import Image, ImageFilter
import glob, itertools


def get_number_area(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_area_list = []
    contour_x_list = []
    contour_y_list = []

    for i in range(0, len(contours)):
        if len(contours[i]) > 0:

            # remove small objects
            if cv2.contourArea(contours[i]) < 500:
                continue

            rect = contours[i]
            x, y, w, h = cv2.boundingRect(rect)
            contour_area_list.append(w*h)
            contour_x_list.append([x, x+w])
            contour_y_list.append([y, y+h])

    if len(contours) > 1:
        largest_contours_idx = contour_area_list.index(max(contour_area_list))
        contour_x_list.pop(largest_contours_idx)
        contour_y_list.pop(largest_contours_idx)

    x_list = list(itertools.chain.from_iterable([contour_x for contour_x in contour_x_list]))
    y_list = list(itertools.chain.from_iterable([contour_y for contour_y in contour_y_list]))

    half_width = int(max(x_list) - min(x_list)/2)
    half_height = int(max(y_list) - min(y_list)/2)

    rate = 1.1
    new_half_width = int(rate*half_width)
    new_half_heiht = int(rate*half_height)

    x_center = min(x_list) + half_width
    y_center = min(y_list) + half_height

    return img[y_center-new_half_heiht : y_center+new_half_heiht, x_center-new_half_width : x_center+new_half_width]


def read_text(img_path, dim):
    img = cv2.imread("img/" + img_path + ".png")
    grayed = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inv = cv2.bitwise_not(grayed)

    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        raise Exception("no ocr found")
    tool = tools[0]

    resized = cv2.resize(inv, dim, interpolation=cv2.INTER_AREA)

    ret, threshed = cv2.threshold(resized, 127, 255, cv2.THRESH_BINARY)

    num_area = get_number_area(threshed)

    pil_img = Image.fromarray(num_area)
    pil_img = pil_img.filter(ImageFilter.ModeFilter(size=18))

    pil_img.save("in_pil.png")
    text = tool.image_to_string(
        pil_img, lang="eng", builder=pyocr.builders.TextBuilder(tesseract_layout=6)
    )

    return text

img_path_list = [int(img_path[4:-4]) for img_path in glob.glob("img/*.png")]
img_path_list.sort()

dim = (900,900)
cnt = 0
sum = 0
for img_path in img_path_list:
    if (img_path-1) % 6 == 0:
        continue
    elif img_path in [2, 30, 36, 42]:
        continue
    sum += 1
    text = read_text(str(img_path), dim)
    try:
        _ = int(text)
        cnt += 1
    except:
        pass
    print(str(img_path) + "\t" + text)

print("\naccuracy: " + str(cnt/sum))