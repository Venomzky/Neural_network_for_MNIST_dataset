import sys, numpy as np
from keras.datasets import mnist

relu = lambda x: (x>=0)*x
relu2deriv = lambda x: x>=0

(x_train, y_train), (x_test, y_test) = mnist.load_data()

images = (x_train[0:1000].reshape(1000,28*28))/255     #pixel value of images reshaped into 28*28 vectors, each pixel value divided by 255
labels = y_train[0:1000]                               #label for each image

hidden_size = 100
alpha = 0.001
pixels_per_image = 784
num_labels = 10
iteration = 300 #not too many iterations to not overfit 
batch_size = 100
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
    for i in range(int(len(images)/batch_size)):
        batch_start, batch_end = ((i*batch_size), ((i+1)*batch_size))

        layer0 = images[batch_start:batch_end]
        layer1 = relu(np.dot(layer0,weights_0_1))
        dropout_mask = np.random.randint(2, size=layer1.shape)
        layer1 *= dropout_mask * 2 #50% of nodes are turned off, value of rest nodes are increased by 1/percent of turned on nodes
        layer2 = np.dot(layer1,weights_1_2)
        error += np.sum((labels[batch_start:batch_end]-layer2)**2)
        for k in range(batch_size):
            correct_cnt += int(np.argmax(layer2[k:k+1])==np.argmax(labels[batch_start+k:batch_start+k+1]))
            layer_2_delta = (labels[batch_start:batch_end]-layer2)/batch_size
            layer_1_delta = layer_2_delta.dot(weights_1_2.T)* relu2deriv(layer1)
            layer_1_delta *= dropout_mask

            weights_1_2 += alpha * layer1.T.dot(layer_2_delta)
            weights_0_1 += alpha * layer0.T.dot(layer_1_delta)
    if (j%10==0):
        test_error = 0.0
        test_correct_cnt = 0
        for i in range(len(test_images)):
            layer0 = test_images[i:i+1]
            layer1 = relu(np.dot(layer0,weights_0_1))
            layer2 = np.dot(layer1,weights_1_2)
            test_error += np.sum((test_labels[i:i+1] - layer2) ** 2)
            test_correct_cnt += int(np.argmax(layer2) == np.argmax(test_labels[i:i+1]))
        print(f"I: {j} Train error: {error/len(images)} Train acc: {correct_cnt/len(images)} Test error: {test_error/len(test_images)} Test acc: {test_correct_cnt/len(test_images)}")
