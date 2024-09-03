import tensorflow as tf
import os
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model

model_dir = "models"
abspath_model_dir = os.path.abspath(model_dir)
model = load_model(os.path.join(abspath_model_dir, "FashionGAN_Generator.keras"))

images = model.predict(tf.random.normal((16, 128, 1)))

fig, ax = plt.subplots(nrows=4, ncols=4, figsize=(20,20))
ax = ax.flatten()
for id in range(16):
    ax[id].imshow(images[id])