import xml.etree.ElementTree as ET
import numpy as np
import cv2

test_num = 1


# image = plt.imread('data/JPEGImages/BloodImage_00003.jpg')
# plt.imshow(image)
# plt.show()

def show_img_annotation(img, annotations, withname=True):
    '''

    :param img: img path
    :param anotations: list, anatations
    :return:
    '''
    image = cv2.imread(img)
    for b in annotations:
        if withname:
            name, xmin, ymin, xmax, ymax = b
        else:
            xmin, ymin, xmax, ymax = b
            name = ' '
        if name == 'RBC':
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
            cv2.putText(image, name, (xmin + 10, ymin + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 255, 0), 1)
        elif name == "WBC":
            cv2.rectangle(image, (xmin, ymin),
                          (xmax, ymax), (0, 0, 255), 1)
            cv2.putText(image, name, (xmin + 10, ymin + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (0, 0, 255), 1)
        elif name == "Platelets":
            cv2.rectangle(image, (xmin, ymin),
                          (xmax, ymax), (255, 0, 0), 1)
            cv2.putText(image, name, (xmin + 10, ymin + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1e-3 * image.shape[0], (255, 0, 0), 1)
        else:
            clr = list(np.random.random(size=3) * 256)
            cv2.rectangle(image, (xmin, ymin),
                          (xmax, ymax), clr, 1)
    cv2.imshow("display", image)
    # cv2.imwrite("display",image)
    cv2.waitKey()


def get_img_annotation(num, root=''):
    img_path = root + 'data/JPEGImages/' + str(num) + '.jpg'
    ana_path = root + 'data/Annotations/' + str(num) + '.xml'
    tree = ET.parse(ana_path)
    root = tree.getroot()
    annotations = []
    for obj in root.iter('object'):
        cls_name = obj.find('name').text
        xml_box = obj.find('bndbox')
        xmin = int(xml_box.find('xmin').text)
        ymin = int(xml_box.find('ymin').text)
        xmax = int(xml_box.find('xmax').text)
        ymax = int(xml_box.find('ymax').text)
        annotations.append([cls_name, xmin, ymin, xmax, ymax])
    return img_path, annotations


if __name__ == '__main__':
    img, anno = get_img_annotation(test_num)
    show_img_annotation(img, anno)
