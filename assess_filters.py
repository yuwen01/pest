import cv2
import numpy as np
import utils
import xml.etree.ElementTree as ET


def read_content(num):
    im_path = "stickytraps/" + str(num) + ".jpg"
    xml_path = "stickytraps/" + str(num) + ".xml"

    img = cv2.imread(im_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    list_with_all_boxes = []

    for boxes in root.iter('object'):

        #filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)
            # if str(boxes.find('name').text) == "WF":
            #     cv2.rectangle(img, (ymin, xmin), (ymax, xmax), (255, 0, 255), 8)
            # if str(boxes.find('name').text) == "MR":
            #     cv2.rectangle(img, (ymin, xmin), (ymax, xmax), (0, 0, 0), 8)
            # if str(boxes.find('name').text) == "NC":
            #     cv2.rectangle(img, (ymin, xmin), (ymax, xmax), (0, 0, 255), 8)

        
    cv2.rectangle(img, (242, 1083), (416, 1184), (0, 0, 255), 8)

    scale_percent = 20  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img  = cv2.resize(img, dim)
    cv2.imshow("Labelled", img)

    #return filename, list_with_all_boxes


if __name__ == '__main__':
    read_content(1013)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
