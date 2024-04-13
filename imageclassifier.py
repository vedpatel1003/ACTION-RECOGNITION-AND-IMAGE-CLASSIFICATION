import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

#batch_size = batch_size refers to the number of training examples you feed to your model at once.
#shuffle = True means that the data will be shuffled after each epoch .
#num_workers specifies the number of helper processes to load and preprocess data in parallel.
#More workers can improve training speed.

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
#here all this 5 lines of code represent a function imshow that displays an image.


# get some random training images
dataiter = iter(trainloader)
#here dataiter is an iterator object that iterates over the training data in batches.
#The iterator object is created by calling the DataLoader function with the trainloader as an argument.
#The iterator object is used to access the training data in batches.
images, labels = next(dataiter)
#here next(dataiter) is a function that returns the next batch of images and labels from the training set.
#The images are represented by tensors with dimensions [batch_size, 3, 32, 32] and the labels are tensors with dimensions [batch_size].
#the returned images are tensors with a range of values between 0 and 1.

# show images
imshow(torchvision.utils.make_grid(images))
#here torchvision.utils.make_grid(images) is a function that takes a batch of images and returns a grid of those images.
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()

#Imagine you have a bunch of categories (like different types of animals: dogs, cats, birds, etc.), and you want a computer program to figure out which category each input belongs to. Cross entropy loss helps the program learn how well it's doing that.

#Predictions: The program makes predictions about which category each input belongs to. For example, it might say there's a 70% chance the input is a dog, 20% chance it's a cat, and 10% chance it's a bird.

#Actual Labels: You already know the correct category for each input. This is the ground truth. For instance, you know that the input is actually a dog.

#omparing Predictions with Reality: Cross entropy loss measures how different the predictions are from reality. If the program said there's a high chance it's a dog, and it turns out it actually is a dog, the loss would be low. But if it confidently predicted it's a cat when it's actually a dog, the loss would be high.

#Calculating the Loss: The loss is calculated by looking at the difference between the predicted probability distribution and the actual distribution (which is usually just a 1 for the correct category and 0s for others). It's a way to quantify how well the predictions match reality.

#Training the Model: During training, the program adjusts its parameters to minimize this loss. It learns from its mistakes and tries to make better predictions next time.


optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#here optimizer is an object that will help us update the weights and biases of our model.
#net.parameters() is a list of all the weights and biases in the network.
#We then pass this to the SGD optimizer, which will update the weights and biases based on the loss function.
#The learning rate (lr) used to determine how big the steps are that the optimizer takes to update the parameters.

#Imagine you're trying to find the best route to a destination. An optimizer is like the GPS system that guides you. Here's how it works:

#Starting Point: You begin with your model, which is like your starting location.

#Destination: Your destination is to minimize the error or loss of your model, making it perform better on tasks like classification or prediction.

#Adjusting Parameters: Just like different routes you can take to reach your destination, your model has parameters that it can tweak to improve its performance. These parameters might represent things like the weights in a neural network.

#Optimization Algorithm: The optimizer is the algorithm that decides how to adjust these parameters. In your example, optim.SGD stands for Stochastic Gradient Descent, which is a popular optimization algorithm. It's like your GPS deciding which roads to take based on traffic conditions and distance.

#Learning Rate and Momentum: The learning rate (lr) is like the speed at which you travel. It determines how big the steps are that the optimizer takes to update the parameters. Momentum (momentum) is like the inertia of your car. It helps the optimizer to keep moving in the same direction when it's finding a good path to the destination.

#Iterative Process: Training a model is like taking a journey. You repeatedly use the optimizer to update the parameters based on the data you have, moving your model closer and closer to its destination of lower error or better performance.

for epoch in range(2):  # loop over the dataset multiple times
#Epoch: This is like showing them your entire (all your training data) once.
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
#This loop iterates over the training data which is likely pre-divided into 
#smaller batches (trainloader is assumed to be a data loader object).
#A single batch is simply a group of things that are done together at the same time.
#i is an index variable keeping track of the current batch number.
#data is a single batch of training data, likely containing a list with two elements: [inputs, labels].

#         # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

#         # zero the parameter gradients
        optimizer.zero_grad()

#small story :-
#The line optimizer.zero_grad() is crucial for training the neural network effectively. Here's why:
#Imagine you're training a student (the neural network) on multiple problems (data points) one after another.
#After each problem, the student learns from its mistakes (the gradients).
#However, if you don't clear the previous mistakes (gradients) before the next problem, the student might get confused and mix up the learning from different problems.
#optimizer.zero_grad() essentially acts like an eraser, clearing the gradients accumulated from the previous data point, ensuring the network learns cleanly from each new example.



        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        #In essence, these five lines represent a single iteration (or training step) in training a neural network. 
        #The process loops through many iterations, feeding the network with batches of data, calculating the loss, 
        #updating the gradients, and adjusting the weights and biases. Over time, the network learns from the data and improves its ability to make accurate predictions.

#         # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')