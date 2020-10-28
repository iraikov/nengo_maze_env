
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from var_autoencoder import make_vae
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

xdim = 128
ydim = 96
original_dim = xdim*ydim
latent_dim = 256

epochs = 10
batch_size = 8

list_data = glob.glob("./rgbd_dataset_freiburg1_xyz/rgb/*.png")
n_input = len(list_data)
all_data = np.zeros((n_input,xdim,ydim))
for i in range(n_input):
    data_frame = np.array(Image.open(list_data[i]).resize((xdim,ydim)).convert('LA'))/255.
    all_data[i] = data_frame[:,:,0].astype('float32').T.reshape((xdim,ydim))


X_train, X_test = train_test_split(range(n_input), test_size=0.2, random_state=27)

n_train = len(X_train)
n_test = len(X_test)
train_data = np.zeros((n_train,xdim,ydim))
test_data = np.zeros((n_test,xdim,ydim))
for i in range(n_train):
    train_data[i] = all_data[X_train[i]]
for i in range(n_test):
    test_data[i,] = all_data[X_test[i]]


vae = make_vae(xdim, ydim, latent_dim, beta=5.)
history = vae.fit(train_data, 
                   epochs=epochs, batch_size=batch_size, shuffle=True,
                   validation_data=(test_data, None))
vae(all_data)
vae.save("vae_dataset_freiburg1")
test_sample_size = 4
test_sample = test_data[0:test_sample_size, :, :]

_ , _, encoded_input = vae.encoder.predict(all_data) 
#plt.figure(figsize=(6, 6))
#plt.scatter(z_test[:, 0], z_test[:, 1], c=y_test,
#            alpha=.4, s=3**2, cmap='viridis')
#plt.colorbar()
#plt.show()
