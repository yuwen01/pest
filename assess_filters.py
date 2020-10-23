import cv2
import numpy as np
import utils
import xml.etree.ElementTree as ET
import bug_detection

'''
parameters:
img_num: the number of the image to look at
detector: the detector function, which takes in an image path, and outputs the number of bugs on that image
display_boxes: whether or not to display the legit bounding boxes
display_bugs: whether or not to display the detected bugs.
'''
def assess_detector(img_num, detector, display_boxes=False, display_bugs=False):
    im_path = "stickytraps/" + str(img_num) + ".jpg"
    xml_path = "stickytraps/" + str(img_num) + ".xml"

    img = cv2.imread(im_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    list_with_all_boxes = []
    bugs = {'WF': 0, 'MR': 0, 'NC': 0}

    for boxes in root.iter('object'):

        ymin, xmin, ymax, xmax = None, None, None, None

        for box in boxes.findall("bndbox"):
            ymin = img.shape[1] - int(box.find("ymin").text)
            xmin = int(box.find("xmin").text)
            ymax = img.shape[1] - int(box.find("ymax").text)
            xmax = int(box.find("xmax").text)
            name = str(boxes.find('name').text)
            if name == "WF":
                cv2.rectangle(img, (ymin, xmin), (ymax, xmax), (255, 0, 255), 8)
            if name == "MR":
                cv2.rectangle(img, (ymin, xmin), (ymax, xmax), (0, 0, 0), 8)
            if name == "NC":
                cv2.rectangle(img, (ymin, xmin), (ymax, xmax), (0, 0, 255), 8)
            bugs[name] += 1

    if display_boxes:
        cv2.rectangle(img, (242, 1083), (416, 1184), (0, 0, 255), 8)
        scale_percent = 20  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        img  = cv2.resize(img, dim)
        cv2.imshow("Labelled", img)
    
    num_detected = detector(im_path, display=display_bugs)

    print('Img number:\t', img_num)
    print("Legit WF:\t", bugs['WF'], '\nLegit MR:\t', bugs['MR'], '\nLegit NC:\t', bugs['NC'])
    print("Legit total:\t", bugs['WF'] + bugs['MR'] + bugs['NC'])
    print("Legit sans WF:\t", bugs['MR'] + bugs['NC'])
    print('Detected total:\t', num_detected)
    print()
    print()

    #return filename, list_with_all_boxes


if __name__ == '__main__':
    for i in range(1000, 1030):
        assess_detector(i, bug_detection.detector)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
