# !pip install torch_snippets torch_summary

import torch
#torch is used for deep learning  computations, while torchsnippets provides small utility functions .
from torch import nn, optim
#nn is used for neural network modules, and optim is used for optimization algorithms.
from torchvision import transforms, models
#torchvision is used for data preprocessing, model architectures, and datasets and transforms is used for image transformations.
#models is used for pre-trained models. The transforms module is used to normalize the data.
from torch_snippets import *
# * is used to import all the functions and classes from the torch_snippets module.
#torch_snippets is a collection of small utility functions for PyTorch.
from torch.utils.data import DataLoader, Dataset
#dataloader refers to a class that is used to load data into PyTorch datasets and data loaders.
#torch.utils.data is used for loading data into PyTorch datasets and data loaders.
from torchsummary import summary 
#torchsummary is a utility function for printing model summary.
import seaborn as sns
#seaborn is used for data visualization.
import matplotlib.pyplot as plt
#matplotlib.pyplot is used for plotting.
from sklearn.model_selection import train_test_split
#sklearn.model_selection is used for splitting data into training and testing sets.
#train_test_split is a function that splits the data into training and testing sets. It returns X_train, y_train for training and X_test, y_test for testing.
from PIL import Image
#pil full form is python image library. It is used for image manipulation and processing.
import numpy as np
#numpy is used for numerical computation.
import cv2
#cv2 is used for computer vision and image processing. here cv2 is used for augmentation.
from glob import glob
#glob is used for searching for files matching a specified pattern.
import pandas as pd
#pandas is used for data manipulation and analysis.

sns.set_theme()
#set_theme is used to set the default theme for seaborn plots.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device is used to specify the device on which the computations will be performed.

DIR = "../input/human-action-recognition-har-dataset/Human Action Recognition/"
#in this dataset image resolution is 240x160.
#here DIR is the directory where the dataset is stored.
TRAIN_DIR=f"{DIR}train"
#here we use f"{DIR}train because we want to concatenate the directory DIR with the string "train".
#Train_dir refers to the directory where the training data is stored.
TEST_DIR=f"{DIR}test"
#Test_dir refers to the directory where the training data is stored.
TRAIN_VAL_DF = "../input/human-action-recognition-har-dataset/Human Action Recognition/Training_set.csv"
#df means dataframe.
#val means validation.
#Train_val_df refers to the csv file containing the training data labels.

train_val_data=glob(TRAIN_DIR+'/*.jpg')
    # remove duplicate
train_val_data.remove('../input/human-action-recognition-har-dataset/Human Action Recognition/train/Image_10169(1).jpg')
train_data, val_data = train_test_split(train_val_data, test_size=0.15,
                                           shuffle=True)
print('Train Size', len(train_data))
print('Val Size', len(val_data))
        
df=pd.read_csv(f"{DIR}Training_set.csv")
#here we read the csv file containing the training data labels.
df.head()
#print the first 5 rows of the dataframe.
    
agg_labels = df.groupby('label').agg({'label': 'count'})
#agg_labels is variable that contains the count of each label in the dataset.
#groupby is used to group the dataframe by the 'label' column and agg is used to aggregate the counts.
#aggregate means summarize the data.
agg_labels.rename(columns={'label': 'count'})
#here rename is used to rename the column 'label' to 'count'.
#we rename the column name because is easier to understand.
ind2cat = sorted(df['label'].unique().tolist())
#here ind2cat is a variable that contains the unique labels in the dataset.
#sorted is used to sort the labels alphabetically.
#unique is used to get the unique labels.
#tolist method is used to convert the unique labels to a list.
#we convert the unique labels to a list because it is easier to work with lists.
#we will use this list to convert the labels to indices.

cat2ind = {cat: ind for ind, cat in enumerate(ind2cat)}
#here cat2ind is a dictionary that maps each category to its index.
#This dictionary maps each label to its index. We use the enumerate function to loop through 
#ind2cat and assign each label an index starting from 0. 
#So, for our example, cat2ind would be {'A': 0, 'B': 1, 'C': 2}

#till now we have done the following:
#1.  created a train_val_data variable that contains the file paths of the training data.
#2.  created a train_data and val_data variable that contains the file paths of the training and validation data.
#3.  created a df variable that contains the dataframe containing the labels of the training data.
#4.  created a agg_labels variable that contains the count of each label in the dataset.
#5.  created a ind2cat variable that contains the unique labels in the dataset.
#6.  created a cat2ind variable that maps each category to its index.

class HumanActionData(Dataset):
    #here we create a custom dataset class.
    #this class will be used to load the data from the dataset.
    def __init__(self, file_paths, df_path, cat2ind):
        #init is a special method that is called when an instance of the class is created.
        #an instance of class is called an object.
        #here we initialize the class with the file_paths, df_path, and cat2ind variables.
        #df_path is the path to the csv file containing the labels.
        #cat2ind is a dictionary that maps each category to its index.
        super().__init__()
        #super is method that is used to call the constructor of the parent class.
        #and parent class is Dataset.
        self.file_paths = file_paths
        #here we initialize the file_paths variable with the file_paths argument.
        self.cat2ind = cat2ind
        self.df = pd.read_csv(df_path)

        #here we initialize the df variable with the df_path argument.
        self.transform = transforms.Compose([ 
            #here we transform the image to a tensor.
            transforms.Resize([224, 244]), 
            #here we resize the image to a specific size.
            transforms.ToTensor(),
            #here we have image of 224,224 now we need to normalize the image.
            # std multiply by 255 to convert img of [0, 255]
            # to img of [0, 1]
            transforms.Normalize((0.485, 0.456, 0.406), 
                                 (0.229*255, 0.224*255, 0.225*255))]
        )
    
    def __len__(self):
        #here we return the length of the dataset.
        return len(self.file_paths)

    
    def __getitem__(self, ind):
        #here getitem is a method that is used to get an item from the dataset.
        #here we return the item at the given index.
        file_path = self.file_paths[ind]
        #here we get the file path of the image at the given index.
        itarget = int(fname(file_path)[6:-4])
        target = self.df.iloc[itarget-1]['label']
        target = self.cat2ind[target]
        img = Image.open(file_path).convert('RGB')
        return img, target
 #Takes an index as input: This means you give the function a number, 
 #which represents the position of the image you want to access.

#Retrieves the file path of the image: Each image has a location or path where it's stored on your computer. 
#The function looks up this path based on the index you provided.

#Extracts the action category label from the filename: 
#The filename usually contains some information about the image. 
#The function pulls out a specific piece of information from the filename that describes 
#what action is happening in the image. 
#For example, if the filename is "running_dog.jpg", it would extract "running" as the action category.

#Converts the category label to a numerical index: 
#Instead of using words to represent categories, 
#it's often easier for computers to work with numbers. 
#So, the function converts the action category label (like "running") into a number. 
#For example, "running" might be converted to 0, "jumping" to 1, and so on. 
#This conversion is done using a dictionary called cat2ind.

#Opens the image file and converts it to RGB format: 
#Images can be stored in different formats, and sometimes we need to convert them to a standard format for processing. 
#RGB format is a common way to represent colors in images. 
#So, this step ensures that the image is in the right format for further processing.

#Returns a tuple containing the transformed image and its corresponding numerical label: 
#Finally, the function puts together the transformed image (in RGB format) along with 
#its numerical label (representing the action category). 
#It bundles them together in a tuple and hands them back to you.
    
    def collate_fn(self, data):
        imgs, targets = zip(*data)
        imgs = torch.stack([self.transform(img) for img in imgs], 0)
        imgs = imgs.to(device)
        targets = torch.tensor(targets).long().to(device)
        return imgs, targets
    # this function takes a bunch of image-label pairs, 
    #transforms the images to make them suitable for training, 
   #and hands back the transformed images and their corresponding labels for further processing.
    
    def choose(self):
        return self[np.random.randint(len(self))]
    #this method randomly selects and returns an item from the object it's called on. 
    #For example, if self is a list, it randomly chooses one item from that list and returns it.
        
train_ds = HumanActionData(train_data, TRAIN_VAL_DF, cat2ind)
#here we initialize the train_ds variable with the train_data and cat2ind variables.
train_dl = DataLoader(train_ds, batch_size=128, shuffle=True,
                      collate_fn=train_ds.collate_fn,
                      drop_last=True)
#here dataloader refers to a class that is used to load the data from the dataset.
#Imagine you have a giant pile of training data. 
#train_dl acts like a helper that creates study sessions (batches) for your model. Here's how:

#It grabs data (flashcards) from your pile (train_ds).
#It groups 128 flashcards together (batch_size).
#It shuffles the pile first (shuffle=True) so your model learns from a random order.
#collate_fn is a function that takes a bunch of flashcards and puts them together in a batch.
#If there are leftover flashcards after making sessions, it skips them (drop_last).

val_ds = HumanActionData(val_data, TRAIN_VAL_DF, cat2ind)
val_dl = DataLoader(val_ds, batch_size=128, shuffle=True,
                    collate_fn=val_ds.collate_fn,
                    drop_last=True)
                    
img, target = train_ds.choose()
#here we choose a random image and its label from the train_ds dataset.
show(img, title=ind2cat[int(target)])
#here we show the image and its label.

inspect(*next(iter(train_dl)), names='image, target')
#here we inspect the first batch of the train_dl dataset.

#till now we have done the following: 
#1.  created a custom dataset class.
#2.  created a dataloader class.
#3.  created a collate_fn function.
#4.  created a train_ds and val_ds dataset.
#5.  created a train_dl and val_dl dataloader.
#6.  inspected the first batch of the train_dl and val_dl dataloader.

class ActionClassifier(nn.Module):
    #here we create a custom classifier class that extends the nn.Module class.
    def __init__(self, ntargets):
        #here init is a special method that is called when an instance of the class is created.
        #ntargets is an integer argument that specifies the number of action classes your model needs to predict.
        super().__init__()
        resnet = models.resnet50(pretrained=True, progress=True)
        #here create a resnet50 model that is pretrained on ImageNet.
        #pretrained=True means that the model will use the weights from the ImageNet dataset.
        #progress=True means that the download and processing of the model will be shown as progress bars.
        modules = list(resnet.children())[:-1] # delete last layer
        #here we delete the last layer of the resnet model because we will be using our own fully connected layer.
        #here we use list that contains all the layers of the resnet model except the last layer.
        #resnet.children() returns an iterator that yields all the layers of the resnet model.
        self.resnet = nn.Sequential(*modules)
        #this resnet model is now a sequential model that contains all the layers of the resnet model except the last layer.
        for param in self.resnet.parameters():
            #this iterates through all the parameters of the resnet model and sets their requires_grad attribute to False. 
            #because we want to freeze the parameters of the resnet model.
            #param is a parameter of the resnet model.
            #requires_grad is a boolean attribute that specifies whether the parameter is trainable or not.
            #False means that the parameter is not trainable.
            param.requires_grad = False
        self.fc = nn.Sequential(
            #here we create a fully connected layer that takes the output of the resnet model as input.
            #nn.Sequential is a module that allows you to define a sequence of layers in a single line of code.
            nn.Flatten(),
            #flatten is a function that flattens a given tensor into a 1D tensor.
            nn.BatchNorm1d(resnet.fc.in_features),

            #batchnorm1d means that the batch normalization layer will be applied to a 1D tensor.
            #here we create a batch normalization layer that normalizes the input tensor.
            #resnet.fc.in_features is the number of input features of the resnet model.
            nn.Dropout(0.2),
            #dropout is a function that randomly sets some elements of a tensor to zero, with a given probability.
            #0.2 means that 20% of the elements will be set to zero.   
            nn.Linear(resnet.fc.in_features, 256),
            #here we create a linear layer that takes the input tensor of size 
            #resnet.fc.in_features and outputs a tensor of size 256.
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, ntargets)
        )
    

    #till now we have done the following:

    #1.  created a custom classifier class that extends the nn.Module class.
    #2.  created a resnet50 model that is pretrained on ImageNet.
    #3.  deleted the last layer of the resnet model.
    #4.  created a fully connected layer that takes the output of the resnet model as input.
    #5.  froze the parameters of the resnet model.
    #6.  froze means that the parameters are not trainable.
    #7.  created a batch normalization layer that normalizes the input tensor.
    #8.  created a dropout layer that randomly sets some elements of a tensor to zero, with a given probability.
    #9.  created a linear layer that takes the input tensor of size resnet.fc.in_features and outputs a tensor of size 256.
    #10. created a relu layer that applies the rectified linear unit activation function to the input tensor.  
    def forward(self, x):
        x = self.resnet(x)
        x = self.fc(x)
        return x
    
    #This function defines the forward pass of a neural network model, likely for image classification.
#It takes an input x, which is typically a tensor representing an image (e.g., with shape (batch_size, channels, height, width)).
#Inside the function:
#x = self.resnet(x): The input x is passed through a pre-trained ResNet model (self.resnet). 
#This model is likely responsible for extracting high-level features from the image.
#x = self.fc(x): The output from the ResNet (x) is then passed through a fully-connected layer (self.fc).
#This layer typically has a large number of neurons and is responsible for classifying the image based on the extracted features.
#return x: The final output x from the fully-connected layer represents the
#model's prediction scores for different classes (e.g., probabilities for each action category).

classifier = ActionClassifier(len(ind2cat))
_ = summary(classifier, torch.zeros(32,3,224,224).to(device))

def train(data, classifier, optimizer, loss_fn):
    classifier.train()
    imgs, targets = data
    outputs = classifier(imgs)
    loss = loss_fn(outputs, targets)
    preds = outputs.argmax(-1)
    acc = (sum(preds==targets) / len(targets))
    classifier.zero_grad()
    loss.backward()
    optimizer.step()
    return loss, acc
    
@torch.no_grad()
def validate(data, classifier, loss_fn):
    classifier.eval()
    imgs, targets = data
    outputs = classifier(imgs)
    loss = loss_fn(outputs, targets)
    preds = outputs.argmax(-1)
    acc = (sum(preds==targets) / len(targets))
    return loss, acc
    
    
n_epochs = 50
log = Report(n_epochs)
classifier = ActionClassifier(len(ind2cat)).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters())
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                     gamma=0.5)
                                     
for epoch in range(n_epochs):
    n_batch = len(train_dl)
    for i, data in enumerate(train_dl):
        train_loss, train_acc = train(data, classifier, 
                                      optimizer, loss_fn)
        pos = epoch + ((i+1)/n_batch)
        log.record(pos=pos, train_loss=train_loss, 
                   train_acc=train_acc, end='\r')
        
    n_batch = len(val_dl)
    for i, data in enumerate(val_dl):
        val_loss, val_acc = validate(data, classifier, loss_fn)
        pos = epoch + ((i+1)/n_batch)
        log.record(pos=pos, val_loss=val_loss, val_acc=val_acc, 
                   end='\r')
    
    scheduler.step()
    log.report_avgs(epoch+1)
    
log.plot_epochs(['train_loss', 'val_loss'])
log.plot_epochs(['train_acc', 'val_acc'])
# !mkdir saved_model
torch.save(classifier.state_dict(), './saved_model/classifier_weights.pth')