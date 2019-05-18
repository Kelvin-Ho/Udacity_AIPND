#import torchvision.transforms as transforms
#import torchvision.models as models
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models

#resnet18 = models.resnet18(pretrained=True)
#alexnet = models.alexnet(pretrained=True)
#vgg16 = models.vgg16(pretrained=True)

#models = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}



def load_datas(data_dir):
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define your transforms for the training, validation, and testing sets
    training_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    testing_transforms = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    training_datasets = datasets.ImageFolder(train_dir,transform=training_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir,transform=validation_transforms)
    testing_datasets = datasets.ImageFolder(test_dir,transform=testing_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
#dataloaders = 
    training_dataloaders = torch.utils.data.DataLoader(training_datasets,batch_size=64,shuffle=True)
    validation_dataloaders = torch.utils.data.DataLoader(validation_datasets,batch_size=64,shuffle=True)
    testing_dataloaders = torch.utils.data.DataLoader(testing_datasets,batch_size=64,shuffle=True)

    class_to_idx = training_datasets.class_to_idx
    
    return training_dataloaders,validation_dataloaders,testing_dataloaders,class_to_idx

def pretrained_model(arch):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    return model

def classifier(model,hidden_units):
    for param in model.parameters():
        param.requires_grad = False
    
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([('fc1',nn.Linear(in_features=25088,out_features=hidden_units,bias=True)),
                                            ('relu',nn.ReLU()),
                                            ('drop',nn.Dropout(p=0.5)),
                                            ('fc2',nn.Linear(in_features=hidden_units,out_features=102,bias=True)),
                                            ('output', nn.LogSoftmax(dim = 1))]))
    model.classifier = classifier
    return model

def train_model(model, training_dataloaders, validation_dataloaders,learning_rate,epochs,processing_unit):
    # Train a model with a pre-trained network 
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    #epochs = 2
    print_every = 40
    steps = 0 

    device = torch.device('cuda:0' if (processing_unit=='gpu') else 'cpu')
    print(device)
    
    # change to cuda/cpu
    model.to(device)

    for e in range(epochs):
    # initialize loss value in each epoch
        running_loss = 0
    
        # Model in training mode, dropout is on
        model.train()    

        for ii, (inputs, labels) in enumerate(training_dataloaders):
            steps +=1

            inputs, labels = inputs.to(device), labels.to(device)

            # initialize optimizer with zero gradient
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            #print(inputs.size())
            loss = criterion(outputs, labels)
            loss.backward() # calculate gradients
            optimizer.step() # use gradient to update the weights

            running_loss += loss.item()

            if steps % print_every == 0:

                # Model in inference mode, dropout is off
                model.eval()

                accuracy = 0
                test_loss = 0

                for ii, (inputs, labels) in enumerate(validation_dataloaders):
                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model.forward(inputs)
                    test_loss += criterion(output, labels).item()

                    ## Calculating the accuracy
                    # Model's output is log-softmax, take exponential to get the probabilities
                    ps = torch.exp(output).data
                    # Class with highest probability is predicted class, compare with true label
                    equality = (labels.data == ps.max(1)[1])
                    # Accuracy is number of correct predction divided by all predictions, just take the mean
                    accuracy += equality.type_as(torch.FloatTensor()).mean()

                print("Epoch: {}/{}...".format(e+1, epochs), 
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}".format(test_loss/len(validation_dataloaders)),
                      "Test Accuracy: {:.3f}".format(accuracy/len(validation_dataloaders)))

                # initialize running_loss in each printing
                running_loss = 0

                # Dropout is on again for training
                model.train()
    return model
            

def valid_model(model,testing_dataloaders,processing_unit):
    model.eval()
    
    correct = 0
    total = 0
    
    device = torch.device('cuda:0' if (processing_unit=='gpu') else 'cpu')
    #print(device)
    
    with torch.no_grad():
        for ii, (inputs, labels) in enumerate(testing_dataloaders):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 819 test images: %d %%' % (100 * correct / total))
    
    
def save_checkpoint(model,save_dir,class_to_idx):
    checkpoint = {'classifier': model.classifier,
              'state_dict': model.state_dict(), 
              'class_to_idx': class_to_idx}
    torch.save(checkpoint, save_dir)
    