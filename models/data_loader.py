import os
import cv2
import numpy as np

IMG_SIZE = 128

def load_data(data_dir):
    data = []
    labels = []

    for category in ["genuine", "morphed"]:
        path = os.path.join(data_dir, category)
        label = 0 if category == "genuine" else 1

        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

                data.append(img)
                labels.append(label)
            except:
                pass

    data = np.array(data) / 255.0
    labels = np.array(labels)

    return data, labels