import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, losses, regularizers
from tensorflow.keras.models import Model
try:
    import tensorflow.contrib.eager as tfe
    #tf.enable_eager_execution()
    tf.compat.v1.enable_eager_execution()
except:
    pass

class Autoencoder(Model):
  def __init__(self, encoding_dim, xdim, ydim):
    super(Autoencoder, self).__init__()
    self.encoding_dim = encoding_dim
    self.xdim = xdim
    self.ydim = ydim
    self.encoder = tf.keras.Sequential()
    self.decoder = tf.keras.Sequential()

    self.encoder.add(layers.InputLayer(input_shape=(xdim, ydim)))
    self.encoder.add(layers.Flatten())
    self.encoder.add(layers.Dense(encoding_dim, activation='relu'))
                                  
    
    self.decoder.add(layers.InputLayer(input_shape=(encoding_dim,)))
    self.decoder.add(layers.Dense(xdim*ydim, activation='sigmoid'))
    self.decoder.add(layers.Reshape((xdim, ydim)))
    self.compile(loss='mean_squared_logarithmic_error',
                 metrics=['mse'], optimizer='adam')
    
  def train(self, train_data, test_data, epochs=10, shuffle=True):
    history = self.fit(train_data, train_data, epochs=epochs, shuffle=True, validation_data=(test_data, test_data))
    print(self.summary())
    return history
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
