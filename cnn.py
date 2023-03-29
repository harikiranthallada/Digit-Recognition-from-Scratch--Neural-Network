import numpy as np
import mnist

from conv import Conv
from maxPool import MaxPool
from softmax import SoftMax

train_images = mnist.train_images()[:10000]
train_labels = mnist.train_labels()[:10000]
test_images = mnist.test_images()[:2000]
test_labels = mnist.test_labels()[:2000]

conv = Conv(8)
mp = MaxPool()
sMax = SoftMax(13 * 13 * 8, 10)


def forward(image, label):

    output = conv.forward((image / 255) - 0.5)
    output = mp.forward(output)
    output = sMax.forward(output)

    loss = -np.log(output[label])


    num_correct = 1 if label == np.argmax(output) else 0
    return output, loss, num_correct


def train(im, label, lr=0.005):

    output, loss, n = forward(im, label)

    gradient = np.zeros(output.shape)
    gradient[label] = -1 / output[label]

    gradient = sMax.backprop(gradient, lr)
    gradient = mp.backprop(gradient)
    conv.backprop(gradient, lr)

    return loss, n

for epoch in range(3):
    print("---- EPOCH : ", epoch," ----")
    loss = 0
    num_correct = 0
    permutations = np.random.permutation(10000)
    train_images = train_images[permutations]
    train_labels = train_labels[permutations]

    for i in range(len(train_images)):
        l, n = train(train_images[i], train_labels[i])
        loss += l
        num_correct += n
        if i % 100 == 99:
            print("step: ", i + 1)
            print("past 100 loss: ", loss / 100, " accuracy: ", num_correct )
            loss = 0
            num_correct = 0


print("---------------- TESTING OF CNN ---------------- ")
loss = 0
num_correct = 0

for i in range(len(test_images)):
    l, n = train(test_images[i], test_labels[i])
    loss += l
    num_correct += n
num_tests = len(test_images)
print("test loss: ", loss/num_tests, "\naccuracy: ", (num_correct/num_tests) * 100)

