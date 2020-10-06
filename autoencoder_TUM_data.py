
from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from autoencoder import Autoencoder

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
