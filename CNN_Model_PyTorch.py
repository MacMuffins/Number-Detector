import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#this is the transformer that is converting and normalizing the mnist dataset from grayscale to RGB
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((.1307,.1307,.1307),(.3081,.3081,.3081))
    ])


#Needed to import the MNIST data set, the root is where it is locally, train is for if its the train set, and download is to download it locally if its not there
#mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
train_set = datasets.MNIST(root= './data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root = './data' , train=False, download=True, transform=transform)

#Need Dataloaders to get labels from the dataset variables
train_loader = DataLoader(dataset=train_set, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_set, batch_size=1000, shuffle=True)

##########################################################
#update this actually does nothing, I need to use a transformer on the dataset to achieve what I'm trying to do here
#########################################################

#the class takes a greyscale image and converts the set into an RGB one
#then it is normalized to 1, since there are 255 pixel values (0-1 instead of 0-255)

#def black_and_white_to_RGB(set_to_convert):
#    set_to_convert = np.stack([set_to_convert]*3, axis=-1)
#    set_to_convert = set_to_convert/255
#    return(set_to_convert)  
#these are now in RGB and not greyscale
#black_and_white_to_RGB(test_set)
#black_and_white_to_RGB(train_set)


class Model(nn.Module):
    def __init__(self):
        super().__init__() 
        self.conv1 = nn.Conv2d(3, 32, 7) #([input size i,e XxXx_], filter , filter size)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256*8*8, 128) #([Flatten output size,which is previous filter x ([input size-kernal size +2 x padding/1] +1)]x[same thing]], new dense layer) AKA just brush up on pooling and output sizes
        self.fc2 = nn.Linear(128, 10)

#This forward definition is the execution of the NN (^) right above
#I choose what to execute and how to execute (v)
#The reason I need init to be defined is to be able to execute this below
#this below essentially is how I want the input data to be manipulated and put into the NN
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        # or x = self.relu(x) after x = self.conv1(x)
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = torch.relu(self.conv4(x))
        x = self.flatten(x)
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim = 1)
        return x
    
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#training loop

epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

#this is the evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
