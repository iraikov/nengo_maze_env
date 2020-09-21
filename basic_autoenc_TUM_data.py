from PIL import Image
import glob
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
try:
    import tensorflow.contrib.eager as tfe
    tf.enable_eager_execution()
except:
    pass

list_data = glob.glob("./rgbd_dataset_freiburg1_xyz/rgb/*.png")

X_train, X_test = train_test_split(list_data, test_size=0.33, random_state=27)

n_train = len(X_train)
train_data = np.zeros((n_train,480,640))
for i in range(n_train):
    train_data_temp = np.asarray(Image.open(X_train[i]).convert('LA'))/255
    train_data[i,:,:] = train_data_temp[:,:,0]
    
n_test = len(X_test)
test_data = np.zeros((n_test,480,640))
for i in range(n_test):
    test_data_temp = np.asarray(Image.open(X_test[i]).convert('LA'))/255
    test_data[i,:,:] = test_data_temp[:,:,0]

latent_dim = 64

class Autoencoder(Model):
  def __init__(self, encoding_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim   
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),
      layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
      layers.Dense(307200, activation='sigmoid'),
      layers.Reshape((480, 640))
    ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim) 
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(train_data, train_data, epochs=10, shuffle=True, validation_data=(test_data, test_data))

low_dim_data = autoencoder.encoder(test_data).numpy()
