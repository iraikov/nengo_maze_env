
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from vlae import make_vlae, flatten_binary_crossentropy
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from urllib.request import urlretrieve
es = EarlyStopping(monitor='loss', min_delta=0.001, patience=3)

# With this routine you can encode inputs and then check the corresponding
# reconstructions
def generate_and_save_images(vae, epoch_label, test_sample):
    predictions = vae(test_sample)
    _, xdim, ydim = test_sample.shape
    for i in range(predictions.shape[0]):
        fig, axes = plt.subplots(nrows=2, ncols=1)
        test_sample_i = np.asarray(test_sample[i, :, :]).reshape ((xdim, ydim))
        prediction_i = np.asarray(predictions[i, :, :]).reshape ((xdim, ydim))
        axes[0].imshow(test_sample_i.T, interpolation='nearest', cmap='gray')
        axes[1].imshow(prediction_i.T, interpolation='nearest', cmap='gray')
        fig.tight_layout()
        plt.savefig(f'image_at_epoch_{epoch_label}_{i}.png')
        plt.show()

xdim, ydim = 28, 28 
latent_dim = 32

epochs = 10
batch_size = 8

#download mnist dataset
(train_data, _), (test_data, _) = tf.keras.datasets.mnist.load_data()

#flatten images
train_data = train_data.reshape((train_data.shape[0], -1))
test_data = test_data.reshape((test_data.shape[0], -1))

train_data = train_data / 255
test_data = test_data / 255

vlae = make_vlae(xdim, ydim, latent_dim)
vlae.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True), loss=flatten_binary_crossentropy)
vlae.summary()
history = vlae.fit(train_data, epochs=epochs, batch_size=batch_size, shuffle=False,
                  validation_data=(test_data, None), callbacks=[es])

vlae.save("vlae_mnist")
test_sample_size = 4
test_sample = test_data[:test_sample_size, :]

encoder = vlae.get_layer("encoder")
encoded_input = encoder.predict(test_data)

#plt.figure(figsize=(6, 6))
#plt.scatter(z_test[:, 0], z_test[:, 1], c=y_test,
#            alpha=.4, s=3**2, cmap='viridis')
#plt.colorbar()
#plt.show()
