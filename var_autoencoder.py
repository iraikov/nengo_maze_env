from tqdm import tqdm
from PIL import Image
import time, glob
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, regularizers, models


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the latent representation vector."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    

        



# Define encoder model.
def create_encoder(xdim,ydim,latent_dim):
    intermediate_dim = latent_dim * 2
    encoder_inputs = tf.keras.Input(shape=(xdim,ydim), name="encoder_input")
    x = layers.Flatten()(encoder_inputs)
    h = layers.Dense(intermediate_dim, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(h)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(h)
    z = Sampling()([z_mean, z_log_var])
    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return encoder

# Define decoder model.
def create_decoder(xdim, ydim, latent_dim):
    intermediate_dim = latent_dim * 2
    latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
    x = layers.Dense(intermediate_dim, input_dim=latent_dim, activation="relu")(latent_inputs)
    x = layers.Dense(xdim*ydim, activation="sigmoid")(latent_inputs)
    decoder_outputs = layers.Reshape((xdim, ydim))(x)
    decoder = tf.keras.Model(inputs=latent_inputs, outputs=decoder_outputs, name="decoder")
    decoder.summary()
    return decoder

# Define encoder model.
def create_encoder_conv(xdim,ydim,latent_dim):
    intermediate_dim = latent_dim * 4
    encoder_inputs = tf.keras.Input(shape=(xdim,ydim), name="encoder_input")
    x = layers.Reshape((xdim, ydim, 1))(encoder_inputs)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(intermediate_dim, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return encoder

# Define decoder model.
def create_decoder_conv(xdim, ydim, latent_dim):
    latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
    x = layers.Dense(int(xdim/4) * int(ydim/4) * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((int(xdim/4), int(ydim/4), 64))(x)
    x = layers.Conv2DTranspose(filters=64, kernel_size=3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(filters=32, kernel_size=3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2DTranspose(filters=1, kernel_size=3, activation="sigmoid", padding="same")(x)
    decoder_outputs = layers.Reshape((xdim, ydim))(x)
    decoder = tf.keras.Model(inputs=latent_inputs, outputs=decoder_outputs, name="decoder")
    decoder.summary()
    return decoder

def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """

    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return tf.reduce_sum(losses.binary_crossentropy(y_true, y_pred))


            

class VAE(models.Model):

    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def get_config(self):
        return {"encoder": self.encoder, "decoder": self.decoder}
        
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
            
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            #reconstruction_loss = tf.reduce_mean(
            #    losses.mean_squared_error(data, reconstruction)
            #    )
            reconstruction_loss = nll(data, reconstruction)
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            return {
                "loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
                }
        
# Train.
def make_vae(xdim, ydim, latent_dim, use_conv=False):
    if use_conv:
        encoder = create_encoder_conv(xdim, ydim, latent_dim)
        decoder = create_decoder_conv(xdim, ydim, latent_dim)
    else:
        encoder = create_encoder(xdim, ydim, latent_dim)
        decoder = create_decoder(xdim, ydim, latent_dim)
        
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True), loss=nll)
    return vae

    

    
