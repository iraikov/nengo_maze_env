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
def create_encoder(xdim,ydim,latent_dim,intermediate_dim=512):
    encoder_inputs = tf.keras.Input(shape=(xdim,ydim), name="encoder_input")
    x = layers.Flatten()(encoder_inputs)
    h = layers.Dense(intermediate_dim, activation="relu", name="intermediate_encoder", kernel_regularizer='l2')(x)
    h = layers.Dense(latent_dim, activation="relu", name="latent_encoder", kernel_regularizer='l2')(h)
    z_mean = layers.Dense(latent_dim, name="z_mean", kernel_regularizer='l2')(h)
    z_log_var = layers.Dense(latent_dim, name="z_log_var", kernel_regularizer='l2')(h)
    z = Sampling()([z_mean, z_log_var])
    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return encoder

# Define decoder model.
def create_decoder(xdim, ydim, latent_dim, intermediate_dim=512):
    latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
    x = layers.Dense(intermediate_dim, input_dim=latent_dim,
                     activation="relu", name="intermediate_decoder",
                     kernel_regularizer='l2')(latent_inputs)
    x = layers.Dense(xdim*ydim, activation="sigmoid", kernel_regularizer='l2')(x)
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

    def __init__(self, encoder, decoder, beta=4., **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta

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
            total_loss = reconstruction_loss + self.beta * kl_loss
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            return {
                "loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
                }
        
# Train.
def make_vae(xdim, ydim, latent_dim, intermediate_dim=512, beta=4.):

    encoder = create_encoder(xdim, ydim, latent_dim, intermediate_dim=intermediate_dim)
    decoder = create_decoder(xdim, ydim, latent_dim, intermediate_dim=intermediate_dim)
        
    vae = VAE(encoder, decoder, beta=beta)
    vae.compile(optimizer=tf.keras.optimizers.Adam(), loss=None)
    return vae

    

    
