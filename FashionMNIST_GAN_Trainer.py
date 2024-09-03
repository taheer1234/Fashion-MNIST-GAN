import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from matplotlib import pyplot as plt
import os

ds = tfds.load("fashion_mnist", split="train")

#A function that allows our data to be scaled between 0 and 1.
def scale_data(data):
    data["image"] = tf.cast(data["image"], tf.float32)
    data["image"] = data["image"] / 255.0
    return data

#A Data Pipeline thats makes use of Map, Cahce, Shuffle, Batch, Prefetch.
#Map uses our scale_data function to normalize every image in our dataset.
ds = ds.map(scale_data)
#Cache caches the dataset into memory after its loaded the first time to speed up processes. Its called after Map so the dataset stored to memory is the one with normalized values.
ds = ds.cache()
#Shuffle creates a buffer (in this case of '60000') and reorders all the elements in the dataset within the buffer and then brings it back to memory.
ds = ds.shuffle(60000)
#Batch creates a sample of 128 (in our case) elements from the dataset to use.
ds = ds.batch(128)
#Prefetch preloads the next batches ('64' in our case) in memory while the preluding batch is being processed.
ds = ds.prefetch(64)

pipeline_batch = ds.as_numpy_iterator().next()
# print(pipeline_batch["image"].shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Reshape, LeakyReLU, Dropout, UpSampling2D

def build_generator():
    model = Sequential()

    #Input layer of the generator that makes the dimenisons of an image from an input layer of 128 random values.
    model.add(Dense(7*7*128, input_dim=128))
    model.add(LeakyReLU(0.2))
    model.add(Reshape((7,7,128)))
    
    #Upsampling the image to make it (14, 14, 128).
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding="same"))
    model.add(LeakyReLU(0.2))

    #Upsampling the image to make it the desired shape for an image (28, 28).
    model.add(UpSampling2D())
    model.add(Conv2D(128, 5, padding="same"))
    model.add(LeakyReLU(0.2))

    #Convolutional layers to add more complexity for the generator to build better images.
    model.add(Conv2D(128, 4, padding="same"))
    model.add(LeakyReLU(0.2))

    model.add(Conv2D(128, 4, padding="same"))
    model.add(LeakyReLU(0.2))

    #Final layer so that the output image of the desired form (28, 28, 1).
    model.add(Conv2D(1, 4, padding="same", activation="sigmoid"))

    return model

generator = build_generator()

#Testing

# image = generator.predict(np.random.randn(8, 128, 1))
# fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
# for id, img in enumerate(image):
#     ax[id].imshow(np.squeeze(img))
#     ax[id].title.set_text(id)
# plt.show()

def build_discriminator():
    model = Sequential()

    #The discriminator inputs an image from the generator and tells us whether its real or not.
    #The Dropout() function allows us to slow down the learning of the discriminator at every step.
    model.add(Conv2D(32, 5, input_shape=(28, 28, 1)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    #Additional layers to make a better discriminator model.
    model.add(Conv2D(64, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(128, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    model.add(Conv2D(256, 5))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))

    #Flatten so that it can input into a dense layer, a layer which at the end gives us a 0 or 1.
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(1, activation="sigmoid"))

    return model

discriminator = build_discriminator()

#Testing

# value = discriminator.predict(image)
# print(value)


from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

generator_optimizer = Adam(learning_rate=0.0001)
discriminator_optimizer = Adam(learning_rate=0.00001)
generator_loss = BinaryCrossentropy()
discriminator_loss = BinaryCrossentropy()

class FashionGAN(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #Make the generator and discriminator available in our class (Make them attributes of our class).
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.g_opt = generator_optimizer
        self.g_loss = generator_loss
        self.d_opt = discriminator_optimizer
        self.d_loss = discriminator_loss

    def train_step(self, batch):
        #Fetching/Creating images for training.
        real_images = batch["image"]
        fake_images = generator(tf.random.normal((128, 128, 1)), training=False)

        #Training the Discriminator.
        with tf.GradientTape() as d_tape:
            #Extarct the output of both fake and real batches from the discriminator.
            predicted_val_real = self.discriminator(real_images, training=True)
            predicted_val_fake = self.discriminator(fake_images, training=True)
            predicted_val_realfake = tf.concat([predicted_val_real, predicted_val_fake], axis=0)

            #Create labels for the real and fake images.
            real_labels = tf.zeros_like(predicted_val_real)  # Label real images as 0
            fake_labels = tf.ones_like(predicted_val_fake)  # Label fake images as 1
            actual_val_realfake = tf.concat([real_labels, fake_labels], axis=0)

            #Create and add noise to the actual values of the discriminator.
            real_noise = 0.15*tf.random.uniform(tf.shape(predicted_val_real))
            fake_noise = 0.15*tf.random.uniform(tf.shape(predicted_val_fake))
            actual_val_realfake += tf.concat([real_noise, fake_noise], axis=0)

            #Calculate loss.
            total_d_loss = self.d_loss(actual_val_realfake, predicted_val_realfake)

        #Apply Backpropogation.
        d_grad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables)
        self.d_opt.apply_gradients(zip(d_grad, self.discriminator.trainable_variables))

        #Training the Generator.
        with tf.GradientTape() as g_tape:
            #Generate a batch of images with training=True.
            gen_images = self.generator(tf.random.normal((128, 128, 1)), training=True)

            #Get the labels from the discriminator with training=False.
            predicted_labels = self.discriminator(gen_images, training=False)

            #Calculate loss by feeding the actual labels as zeros (meaning that they are real (even though they are not)) and the predicted values as those calculated above.
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels)

        #Apply Backpropogation.
        g_grad = g_tape.gradient(total_g_loss, self.discriminator.trainable_variables)
        self.g_opt.apply_gradients(zip(g_grad, self.discriminator.trainable_variables))
        
        return {"d_loss": total_d_loss, "g_loss": total_g_loss}

#Training the model.
fashgan = FashionGAN(generator, discriminator)
fashgan.compile(generator_optimizer, discriminator_optimizer, generator_loss, discriminator_loss)
fashgan.fit(ds, epochs=20)

#Saving the models.
model_dir = "models"
abspath_model_dir = os.path.abspath(model_dir)
generator.save(os.path.join(abspath_model_dir, "FashionGAN_Generator.keras"))
discriminator.save(os.path.join(abspath_model_dir, "FashionGAN_Discriminator.keras"))