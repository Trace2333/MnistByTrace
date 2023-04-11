import os
from PIL import image
import pickle as pkl
from tqdm import tqdm
import logging


logging.info("process the data...")

dataset_name = 'fashion_mnist'
# store the raw image files
data_filename_or_path = './dataset/fashion_mnist'
# store the processed pkl file
data_target_path = './dataset/'

png_list = os.listdir(data_filename_or_path)
# list of pngs, such as: xxxx/xxx/1.png
image_objs = [Image.open(path) for path in tqdm(png_list, desc="runing on images")]
with open(os.path.join(data_target_path, dataset_name + "_images.png"), 'wb') as f1:
    pkl.dump(image_objs, f1)
logging.info("Data prepare complete!")

