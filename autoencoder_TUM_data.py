
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from autoencoder import Autoencoder
import tensorflow as tf

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
        plt.savefig('image_at_epoch_{}_{}.png'.format(epoch_label,i))
        plt.show()

list_data = glob.glob("./rgbd_dataset_freiburg1_xyz/rgb/*.png")

X_train, X_test = train_test_split(list_data, test_size=0.2, random_state=27)

xdim = 128
ydim = 96

n_train = len(X_train)
train_data = np.zeros((n_train,xdim,ydim))
for i in range(n_train):
    train_data_temp = np.array(Image.open(X_train[i]).resize((xdim,ydim)).convert('LA'))/255.
    train_data[i,:,:] = train_data_temp[:,:,0].T
    
n_test = len(X_test)
test_data = np.zeros((n_test,xdim,ydim))
for i in range(n_test):
    test_data_temp = np.array(Image.open(X_test[i]).resize((xdim,ydim)).convert('LA'))/255.
    test_data[i,:,:] = test_data_temp[:,:,0].T

encoding_dim = 256

autoencoder = Autoencoder(encoding_dim, xdim, ydim)
history = autoencoder.train(train_data, test_data, epochs=30)

low_dim_data = autoencoder.encoder(test_data).numpy()
encoded = autoencoder.encoder.predict(test_data)
decoded = autoencoder.decoder.predict(encoded)

batch_size = 8
test_dataset = tf.data.Dataset.from_tensor_slices(test_data).batch(batch_size)
test_sample_size = 2
for test_batch in test_dataset.take(1):
    test_sample = test_batch[0:test_sample_size, :, :]


plt.subplot(211)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plot mse during training
#plt.subplot(212)
#plt.title('Mean Squared Error')
#plt.plot(history.history['mean_squared_error'], label='train')
#plt.plot(history.history['val_mean_squared_error'], label='test')
#plt.legend()
#plt.show()
