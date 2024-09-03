import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt

ds = tfds.load("fashion_mnist", split="train")

#Shuffle the data to look at different images each time the code is run.
ds = ds.shuffle(buffer_size=10000)

#This allows to import over a new batch everytime the iterator is called. A new batch in this pipeline is a single image and its label.
data_iterator = ds.as_numpy_iterator()

#Calls the iterator for a new image and label and displays a group of four of them along with their label.
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for id in range(4):
    batch = next(data_iterator)
    ax[id].imshow(np.squeeze(batch["image"]))
    ax[id].title.set_text(str(batch["label"]))

plt.show()