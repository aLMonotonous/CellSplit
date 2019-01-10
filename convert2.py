import random
import xml.etree.ElementTree as ET
from glob import glob

import pandas as pd

annotations = glob('data/Annotations/*.xml')

df = []
cnt = 0
ratio = 0.66
classes = ["RBC", "WBC", "Platelets"]
ratio = 0.66
path = 'data'


#train/test split
idx_random = list(range(file_num))
random.shuffle(idx_random)
idx_train = idx_random[:int(file_num*ratio)+1]
idx_val = idx_random[int(file_num*ratio)+1:]


#covert
for file in annotations:
    prev_filename = file.split('/')[-1].split('.')[0] + '.jpg'
    filename = str(cnt) + '.jpg'
    row = []
    parsedXML = ET.parse(file)
    for node in parsedXML.getroot().iter('object'):
        blood_cells = node.find('name').text
        if blood_cells not in classes:
            continue
        blood_cells =
        xmin = int(node.find('bndbox/ymin').text)
        xmax = int(node.find('bndbox/ymax').text)
        ymin = int(node.find('bndbox/xmin').text)
        ymax = int(node.find('bndbox/xmax').text)
        row = [prev_filename, filename, blood_cells, xmin, xmax,
        ymin, ymax]
        df.append(row)
    cnt += 1
data = pd.DataFrame(df, columns=['prev_filename', 'filename', 'cell_type',
'xmin', 'xmax', 'ymin', 'ymax'])
data[['prev_filename', 'cell_type', 'xmin', 'xmax', 'ymin', 'ymax']].to_csv('data/list/blood_cell_detection.csv', index=False)
print(data.head())





