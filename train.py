#train.py

import sys
import os
import json
import train_args
import torch

from collections import OrderedDict
from torchvision import models
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

#parsing

def main():
   
    parser = train_args.get_args()
    parser.add_argument('--version',
                        action='version',
                        version='%(prog)s ' + __version__ + ' by ' + __author__)
    cli_args = parser.parse_args()
    #  directory
	#First check
    if not os.path.isdir(cli_args.data_directory):
        print(f'Data directory {cli_args.data_directory} not found.')
        exit(1)

    # Then save directory
    if not os.path.isdir(cli_args.save_dir):
        print(f'Directory {cli_args.save_dir} does not exist. Creating...')
        os.makedirs(cli_args.save_dir)
		
    with open(cli_args.categories_json, 'r') as f:
        cat_to_name = json.load(f)
		
    output_size = len(cat_to_name)
   
    expected_means = [0.485, 0.456, 0.406]
    expected_std = [0.229, 0.224, 0.225]
    max_image_size = 224
    batch_size = 32
#train_transform
    tr_transform = transforms.Compose([transforms.RandomHorizontalFlip(p=0.25),
                                           transforms.RandomRotation(25),
                                           transforms.RandomGrayscale(p=0.02),
                                           transforms.RandomResizedCrop(max_image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(expected_means, expected_std)])
#train_dataset
    tr_dataset = datasets.ImageFolder(cli_args.data_directory, transform=tr_transform)
#tr_dataloader
    tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
	
	
# model
    if not cli_args.arch.startswith("vgg") and not cli_args.arch.startswith("densenet"):
        print("Only supporting VGG and DenseNet")
        exit(1)

    print(f"Using a pre-trained {cli_args.arch} network.")
    my_model = models.__dict__[cli_args.arch](pretrained=True)

    densenet_input = {
        'densenet121': 1024,
        'densenet169': 1664,
        'densenet161': 2208,
        'densenet201': 1920
    }

    input_size = 0

    if cli_args.arch.startswith("vgg"):
        input_size = my_model.classifier[0].in_features

    if cli_args.arch.startswith("densenet"):
        input_size = densenet_input[cli_args.arch]
		
    for param in my_model.parameters():
        param.requires_grad = False

    od = OrderedDict()
    hidden_sizes = cli_args.hidden_units

    hidden_sizes.insert(0, input_size)

    print(f"Building a {len(cli_args.hidden_units)} hidden layer classifier with inputs {cli_args.hidden_units}")

    for i in range(len(hidden_sizes) - 1):
        od['fc' + str(i + 1)] = nn.Linear(hidden_sizes[i], hidden_sizes[i + 1])
        od['relu' + str(i + 1)] = nn.ReLU()
        od['dropout' + str(i + 1)] = nn.Dropout(p=0.15)

    od['output'] = nn.Linear(hidden_sizes[i + 1], output_size)
    od['softmax'] = nn.LogSoftmax(dim=1)

    classifier = nn.Sequential(od)

    # Replace the classifier
    my_model.classifier = classifier

      my_model.zero_grad()

    criterion = nn.NLLLoss()

    # Adam:
    
    optimizer = optim.Adam(my_model.classifier.parameters(), lr=cli_args.learning_rate)

    # CPU
    device = torch.device("cpu")

    # GPU
    if cli_args.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("GPU is not available. Using CPU.")

    print(f"Sending model to device {device}.")
    my_model = my_model.to(device)

    data_set_len = len(tr_dataloader.batch_sampler)

    chk_every = 30


    for e in range(cli_args.epochs):
        e_loss = 0
        prev_chk = 0
        total = 0
        correct = 0
       
        for ii, (images, labels) in enumerate(tr_dataloader):
            
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # forward and backward Propigation
            outputs = my_model.forward(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            e_loss += loss.item()

            # calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

         #loss
            itr = (ii + 1)
            if itr % chk_every == 0:
                avg_loss = f'avg. loss: {e_loss/itr:.4f}'
                acc = f'accuracy: {(correct/total) * 100:.2f}%'
                print(f'  Batches {prev_chk:03} to {itr:03}: {avg_loss}, {acc}.')
                prev_chk = (ii + 1)


    my_model.class_to_idx = training_dataset.class_to_idx
    model_state = {
        'epoch': cli_args.epochs,
        'state_dict': my_model.state_dict(),
        'optimizer_dict': optimizer.state_dict(),
        'classifier': my_model.classifier,
        'class_to_idx': my_model.class_to_idx,
        'arch': cli_args.arch
    }

    save_location = f'{cli_args.save_dir}/{cli_args.save_name}.pth'
    print(f"Saving checkpoint to {save_location}")

    torch.save(model_state, save_location)


if __name__ == '__main__':
#end