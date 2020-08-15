import numpy as np
import h5py
import os
from PIL import Image

class HDF5_write:
    """
    Write training and testing HDF5 files.
    
    Parameters
    ----------
    path: string
        The path of the file directory.
    name: string
        Name of the hdf5 file.
    dims: list
        Size of dataset in hdf5 file.
        Should be [number of images, row of images, column of images].
    bufSize: integer
        The size of buffer set.
        Initialize as 1000.
    """
    # Initialise an empty hdf5 file and a buffer set.
    def __init__(self, path, name, dims, bufSize=1000): 
        name = '.'.join([name,'hdf5'])
        self.path = path
        self.db = h5py.File(name , 'w')
        self.dims = dims[1:]
        self.data =  self.db.create_dataset('images', dims, dtype='float64')
        self.labels = self.db.create_dataset('labels', (dims[0],))
        
        self.bufSize = bufSize
        self.buffer = {'data': [], 'labels': []}
        self.idx = 0
    
    # Function add append images and labels into buffer set, then flush the buffer set into hdf5 dataset
    # if reach the buffer size.
    def add(self, image, label):
        self.buffer['data'].append(image)
        self.buffer['labels'].append(int(label))
        
        # check if we need to refresh the buffer.
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()
    
    # Function flush used to put the data into hdf5 dataset.
    def flush(self):
        # refresh buffer.
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}
        
    # Function close flush the rest of buffer set data into hdf5 dataset.
    def close(self):
        if len(self.buffer["data"]) > 0:
            self.flush()

        self.db.close()
    
    # Function img_processing will process
    def img_processing(self, img_info):
        img_arr = np.array(img_info)
        # Turn the array-like image into image type then resize it to self.dims
        img_resized = Image.fromarray(img_arr).resize(self.dims[:-1])
        self.img_arr_resized = np.array(img_resized)

    # Function Write_it 
    def Write_it(self, kind_of_img='labeled'):
        if kind_of_img == 'labeled':
            # This part is related with the structure of directory.
            for child_dir in os.listdir(self.path):
                child_path = os.path.join(self.path, child_dir)
                for label_dir in os.listdir(child_path):
                    label_path = os.path.join(child_path, label_dir)
                    for dir_image in os.listdir(label_path):
                        if dir_image.endswith('.png'):
                            img = Image.open(os.path.join(label_path, dir_image))
                            self.img_processing(img)
                            if self.img_arr_resized.shape == self.dims:
                                self.add(self.img_arr_resized, child_dir)
            self.close()
        elif kind_of_img == 'unlabeled':
            # This part is related with the structure of directory.
            for child_dir in os.listdir(self.path):
                child_path = os.path.join(self.path, child_dir)
                for dir_image in os.listdir(child_path):
                    img = Image.open(os.path.join(child_path, dir_image))
                    self.img_processing(img)
                    self.add(self.img_arr_resized, child_dir)
            self.close()