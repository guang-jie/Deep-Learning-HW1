# HW1 Image Classification for Deep learning by Prof.Hsu #

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from os.path import join
from dataset import CustomImageDataset
from function import onehot_encoding, turn_to_score, accurate_num
from model import CNN
import config as cfg
from torch.autograd import Variable

epoch_num = 500
batch_size = 500

device = cfg.device
current_path = cfg.current_path
checkpoint = join(current_path, "checkpoint")
classifier1 = CNN().to(device)
optimizer1 = torch.optim.Adam(classifier1.parameters(), lr=0.01)


criterion = torch.nn.MSELoss()


TrainData = CustomImageDataset("train.txt")
ValidData = CustomImageDataset("val.txt")
TestData = CustomImageDataset("test.txt")
trainloader = DataLoader(TrainData, batch_size=batch_size, shuffle=True, num_workers=0)
validloader = DataLoader(ValidData, batch_size=batch_size, shuffle=False, num_workers=0)
testloader = DataLoader(TestData, batch_size=batch_size, shuffle=False, num_workers=0)


for epoch in tqdm(range(epoch_num)):

    # Training Part #
    classifier1.train()
    train_accuracy = 0
    loss_list = []
    for i, (image, label) in enumerate(trainloader):
        optimizer1.zero_grad()

        image = image.permute(0, 3, 1, 2) #shape = (batch, channels, height, width)
        image = image.to(device)
        encoded_label = onehot_encoding(label).to(device)

        score_vector = classifier1(image.to(torch.float32)).to(device)
        score_vector = score_vector.cpu().detach().numpy() # (450, 50)
    
        label = np.array(label).astype(int)
        
        score_list = []
        for idx in range(len(score_vector)):
            score = turn_to_score(score_vector[idx, :])
            score_list.append(score)
        score_list = np.array(score_list)
        
        accurate_batch = accurate_num(label, score_list) # calculate accurate number of a batch
        train_accuracy = train_accuracy + accurate_batch # sum accurate number for every batch
        
        score_vector = torch.tensor(score_vector).to(device)
        loss = criterion(score_vector, encoded_label)
        loss = Variable(loss, requires_grad = True)
        
        loss.backward()
        optimizer1.step()

        loss_list.append(loss.item())
        
    train_accuracy = train_accuracy / TrainData.__len__()
    train_loss = np.array(loss_list).mean()



    # Valid Part #
    classifier1.eval()
    valid_accuracy = 0
    loss_list = []
    for i, (image, label) in enumerate(validloader):

        image = image.permute(0, 3, 1, 2) #shape = (batch, channels, height, width)
        image = image.to(device)
        encoded_label = onehot_encoding(label).to(device)

        score_vector = classifier1(image.to(torch.float32)).to(device)
        score_vector = score_vector.cpu().detach().numpy() # (450, 50)
    
        label = np.array(label).astype(int)
        
        score_list = []
        for idx in range(len(score_vector)):
            score = turn_to_score(score_vector[idx, :])
            score_list.append(score)
        score_list = np.array(score_list)
        
        accurate_batch = accurate_num(label, score_list) # calculate accurate number of a batch
        valid_accuracy = valid_accuracy + accurate_batch # sum accurate number for every batch 
        
        score_vector = torch.tensor(score_vector).to(device)
        loss = criterion(score_vector, encoded_label)
        loss_list.append(loss.item())
        
    valid_accuracy = valid_accuracy / ValidData.__len__()
    valid_loss = np.array(loss_list).mean()



    # Test Part #
    classifier1.eval()
    test_accuracy = 0
    loss_list = []
    for i, (image, label) in enumerate(testloader):

        image = image.permute(0, 3, 1, 2) #shape = (batch, channels, height, width)
        image = image.to(device)
        encoded_label = onehot_encoding(label).to(device)

        score_vector = classifier1(image.to(torch.float32)).to(device) 
        score_vector = score_vector.cpu().detach().numpy() # (450, 50)
    
        label = np.array(label).astype(int)
        
        score_list = []
        for idx in range(len(score_vector)):
            score = turn_to_score(score_vector[idx, :])
            score_list.append(score)
        score_list = np.array(score_list)
        
        accurate_batch = accurate_num(label, score_list) # calculate accurate number of a batch
        test_accuracy = test_accuracy + accurate_batch # sum accurate number for every batch 
        
        score_vector = torch.tensor(score_vector).to(device)
        loss = criterion(score_vector, encoded_label)
        loss_list.append(loss.item())

    test_accuracy = test_accuracy / TestData.__len__()
    test_loss = np.array(loss_list).mean()

    # print accuracy for every epoch
    msg = "Epoch {}, Train accuracy: {}, Valid accuracy: {}, Test accuracy: {}, Test loss: {}".format(epoch, train_accuracy, valid_accuracy, test_accuracy, test_loss)
    with open("accuracy.txt", "a+") as f:
        f.write(msg + '\n')
    print(msg)
    
    if epoch == 0:
        min = test_loss
    elif test_loss < min:
        min = test_loss
    

    torch.save({'model': classifier1.state_dict()}, join(checkpoint, "model_epoch_{}".format(epoch)))







