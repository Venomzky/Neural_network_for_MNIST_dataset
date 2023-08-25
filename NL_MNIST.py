import sys, numpy as np
from keras.datasets import mnist

relu = lambda x: (x>=0)*x
relu2deriv = lambda x: x>=0

(x_train, y_train), (x_test, y_test) = mnist.load_data()

images = (x_train[0:1000].reshape(1000,28*28))/255     #pixel value of images reshaped into 28*28 vectors, each pixel value divided by 255
labels = y_train[0:1000]                               #label for each image

hidden_size = 40
alpha = 0.005
pixels_per_image = 784
num_labels = 10
iteration = 50 #not too many iterations to not overfit 
np.random.seed(1)
weights_0_1 = 0.2*np.random.random((pixels_per_image, hidden_size)) - 0.1
weights_1_2 = 0.2*  np.random.random((hidden_size,num_labels)) - 0.1


one_hot_labels = np.zeros((len(labels), 10))
#convert labels into 
for i, j in enumerate(labels):  
    one_hot_labels[i][j]=1
labels = one_hot_labels  


test_images = x_test.reshape(len(x_test),28*28) / 255
test_labels = np.zeros((len(y_test),10))
for i,l in enumerate(y_test):
    test_labels[i][l] = 1



for j in range(iteration):
    correct_cnt = 0
    error = 0.0
    for i in range(len(images)):
        layer0 = images[i:i+1]
        layer1 = relu(np.dot(layer0,weights_0_1))
        layer2 = np.dot(layer1,weights_1_2)
        error += np.sum((labels[i:i+1]-layer2)**2)
        correct_cnt += int(np.argmax(layer2)==np.argmax(labels[i:i+1]))

        layer2_delta = (labels[i:i+1]-layer2)
        layer1_delta = layer2_delta.dot(weights_1_2.T)*relu2deriv(layer1)
        
        weights_1_2 += alpha * layer1.T.dot(layer2_delta)
        weights_0_1 += alpha * layer0.T.dot(layer1_delta)
print("training results")
print(f"error: {error/len(images)} acc:{correct_cnt/len(images)}")



error = 0.0
correct_cnt = 0
for i in range(len(test_images)):
    layer_0 = test_images[i:i+1]
    layer_1 = relu(np.dot(layer_0,weights_0_1))
    layer_2 = np.dot(layer_1,weights_1_2)
    error += np.sum((test_labels[i:i+1] - layer_2) ** 2)
    correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i+1]))
print("results on unseen dataset")
print(f"error: {error/len(test_images)} acc:{correct_cnt/len(test_images)}")