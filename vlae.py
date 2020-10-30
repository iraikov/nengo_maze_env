import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, regularizers, models
from tensorflow.keras.layers import Input, Flatten, Reshape, Concatenate, Dense, Conv2D, BatchNormalization, ReLU, SpatialDropout2D, UpSampling2D, Activation, LeakyReLU, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
import tensorflow_probability as tfp


def flatten_binary_crossentropy(x,xhat):
    return 10 * tf.losses.binary_crossentropy(Flatten()(x), Flatten()(xhat))

class NormalVariational(tf.keras.layers.Layer):

    def __init__(self, size, mu_prior=0., sigma_prior=1., add_kl=True, coef_kl = 1.0, add_mmd=False, lambda_mmd=1.0, kernel_f=None, name=None, show_posterior=True):
        super().__init__(name=name)
        self.mu_layer = tf.keras.layers.Dense(size)
        self.sigma_layer = tf.keras.layers.Dense(size)
        self.add_kl = add_kl
        self.mu_prior = tf.constant(mu_prior, dtype=tf.float32, shape=(size,))
        self.sigma_prior = tf.constant(sigma_prior, dtype=tf.float32, shape=(size,))
        self.show_posterior = show_posterior
        self.coef_kl = tf.Variable(coef_kl, trainable=False, name='coef_kl')
        self.add_mmd = add_mmd
        if kernel_f is None:
            self.kernel_f = self._rbf
        else:
            self.kernel_f = kernel_f
        self.lambda_mmd = lambda_mmd
            
    def _rbf(self, x, y):
        x_size = tf.shape(x)[0]
        y_size = tf.shape(y)[0]
        dim = tf.shape(x)[1]
        tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
        tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
        return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

    def _linear(self, x,y):
        return tf.reduce_sum(tf.multiply(x,y))

    def add_kl_divergence(self, mu1, sigma1, mu2, sigma2):
        logsigma1, logsigma2 = tf.math.log(sigma1), tf.math.log(sigma2)
        mu_diff = mu1 - mu2
        kl = self.coef_kl * \
          tf.reduce_sum(logsigma1 - logsigma2 - 1. + (sigma2 + tf.square(mu_diff)) / sigma1, axis=1)
        kl = tf.reduce_mean(kl)
        self.add_loss(kl)
        self.add_metric(kl, 'kl_divergence')


    def call(self, inputs):
        mu = self.mu_layer(inputs)
        log_sigma =  self.sigma_layer(inputs)
        sigma_square = tf.exp(log_sigma)
        if self.add_kl:
            self.add_kl_divergence(mu, sigma_square, self.mu_prior, self.sigma_prior)
        if self.show_posterior:
            self.add_metric(mu, 'mu_posteror')
            self.add_metric(sigma_square, 'sigma^2_posterior')
        z = mu + sigma_square * tf.random.normal(tf.shape(sigma_square))
        if self.add_mmd:
            z_prior = tfp.distributions.MultivariateNormalDiag(self.mu_prior, self.sigma_prior).sample(tf.shape(z)[0])
            #print(z_prior)
            #print(z)
            k_prior = self.kernel_f(z_prior, z_prior)
            k_post = self.kernel_f(z, z)
            k_prior_post = self.kernel_f(z_prior, z)
            mmd = tf.reduce_mean(k_prior) + tf.reduce_mean(k_post) - 2 * tf.reduce_mean(k_prior_post)
            mmd = tf.multiply(self.lambda_mmd,  mmd, name='mmd')
            self.add_loss(mmd)
            self.add_metric(mmd, 'mmd')
        return z

    
def make_encoder(xdim, ydim, latent_dim1, latent_dim2, latent_dim3, dropout_rate = 0.05):
    
    inputs = Input((xdim,ydim,1))
    with tf.name_scope('h_1'):
        h_1_layers = Sequential([ 
            Input((xdim, ydim, 1)),
            Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"),
            BatchNormalization(trainable=False),
            ReLU(),
            MaxPooling2D((2, 2), padding='same'),
            ], name='h_1')
        h_1 = h_1_layers(inputs)
        h_1_flatten = Flatten()(h_1)
    with tf.name_scope('h_2'):
        h_2_layers = Sequential([ 
            Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"),
            BatchNormalization(trainable=False),
            ReLU(),
            MaxPooling2D((2, 2), padding='same'),
            ], name='h_2')
        h_2 = h_2_layers(h_1)
        h_2_flatten = Flatten()(h_2)
    with tf.name_scope('h_3'):
        h_3_layers = Sequential([ 
            Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"),
            BatchNormalization(trainable=False),
            ReLU(),
            MaxPooling2D((2, 2), padding='same'),
            ], name='h_3')
        h_3 = h_3_layers(h_2)
        h_3_flatten = Flatten()(h_3)
    z_1 = NormalVariational(latent_dim1, add_kl=False, coef_kl=0.0, add_mmd=True,
                            lambda_mmd=1., name='z_1_latent')(h_1_flatten)
    z_2 = NormalVariational(latent_dim2, add_kl=False, coef_kl=0.0, add_mmd=True,
                            lambda_mmd=1., name='z_2_latent')(h_2_flatten)
    z_3 = NormalVariational(latent_dim3, add_kl=False, coef_kl=0.0, add_mmd=True,
                            lambda_mmd=1., name='z_3_latent')(h_3_flatten)

    return Model(inputs, [z_1, z_2, z_3], name='encoder')
        
def make_decoder(latent_dim1, latent_dim2, latent_dim3, xdim, ydim):
    scale_factor = 2
    intermediate_dim = int(xdim/scale_factor) * int(ydim/scale_factor)
    z_1_input, z_2_input, z_3_input = Input((latent_dim1,), name='z_1'), Input((latent_dim2,), name='z_2'), Input((latent_dim3,), name='z_3')
    
    with tf.name_scope('z_tilde_3'):
        z_3 = Dense(intermediate_dim, activation='relu')(z_3_input)
        z_tilde_3_layers = Sequential([
            Dense(intermediate_dim),
            BatchNormalization(trainable=False),
            ReLU()] * 3, name='z_tilde_3')
        z_tilde_3 = z_tilde_3_layers(z_3)
        
    with tf.name_scope('z_tilde_2'):
        z_2 = Dense(intermediate_dim, activation='relu')(z_2_input)
        z_tilde_2_layers = Sequential([
            Dense(intermediate_dim),
            BatchNormalization(trainable=False),
             ReLU()] * 3, name='z_tilde_2')
        input_z_tilde_2 = Concatenate()([z_tilde_3, z_2])
        z_tilde_2 =  z_tilde_2_layers(input_z_tilde_2)
    
    with tf.name_scope('z_tilde_1'):
        z_1 = Dense(intermediate_dim, activation='relu')(z_1_input)
        z_tilde_1_layers = Sequential([
            Dense(intermediate_dim),
            BatchNormalization(trainable=False),
             ReLU()] * 3, name='z_tilde_1')
        input_z_tilde_1 = Concatenate()([z_tilde_2, z_1])
        z_tilde_1 =  z_tilde_1_layers(input_z_tilde_1)
        
    with tf.name_scope('decoder'):
       x = layers.Dense(intermediate_dim, activation="relu", )(z_tilde_1)
       x = BatchNormalization(trainable=False)(x)
       x = layers.Dense(xdim*ydim, activation="sigmoid", )(x)
       decoder_output = layers.Reshape((xdim, ydim))(x)
       
    return Model([z_1_input, z_2_input, z_3_input], decoder_output, name='decoder')

def make_vlae(xdim, ydim, latent_dim):
    with tf.name_scope('encoder'):
        encoder = make_encoder(xdim, ydim, latent_dim, latent_dim, latent_dim)
    with tf.name_scope('decoder'):
        decoder = make_decoder(latent_dim, latent_dim, latent_dim, xdim, ydim)
    inputs = Input((xdim,ydim,1))
    z_1, z_2, z_3 = encoder(inputs)
    decoded = decoder([z_1, z_2, z_3])
    vlae = Model(inputs, decoded, name='vlae')
    return vlae
