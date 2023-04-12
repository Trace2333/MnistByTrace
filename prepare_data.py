import os
from PIL import Image
import pickle as pkl
from tqdm import tqdm
import logging
import mnist_reader


logging.info("process the data...")

dataset_name = 'fashion_mnist'
# store the processed pkl file
dataset_path = 'f_mnist/'
X_train, y_train = mnist_reader.load_mnist('f_mnist/', kind='train')
X_test, y_test = mnist_reader.load_mnist('f_mnist/', kind='t10k')
# list of pngs, such as: xxxx/xxx/1.png
data_obj_train = {
    "X": X_train,
    "Y": y_train,
}
data_obj_test = {
    "X": X_test,
    "Y": y_test,
}
store_path = os.path.join(dataset_path, dataset_name)
store_path = store_path.replace("\\", "/")

with open(store_path + "_images_train.pkl", 'wb') as f1:
    pkl.dump(data_obj_train, f1)

with open(store_path + "_images_test.pkl", 'wb') as f1:
    pkl.dump(data_obj_test, f1)
logging.info("Data prepare complete!")

