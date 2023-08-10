import os
import sys
import json
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import *


def print_model_params(params, train_loss,val_loss, time, best_acc):
    '''
    print model training log
    '''
    with open('model_training_log.txt', 'a') as f:
        f.write('/' + '*' * 80 + '/\n')
        f.write('training date:\t' + str(datetime.datetime.now()) + '\n')
        f.write('model name:\t' + 'resnet_'+params['save_path']+  '\n')
        f.write('model hyper-parameters:\t')
        #params.pop('dataset_name', None)
        js = json.dumps(params)
        f.write(js)
        f.write('\n')
        f.write('best_accuracy:\t%.03f\n' % best_acc)
        f.write('train_loss:\t%.06f\n' % train_loss)
        f.write('val_loss:\t%.06f\n' % val_loss)
        f.write('training_time:\t%.03f\n' % time)
        f.write('/' + '*' * 80 + '/\n')
    return 0


def main(parameter):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    #set hyper-parameters
    batch_size = parameter['batch_size']
    lr = parameter['learning_rate']
    epochs = parameter['epochs']
    model_type = parameter['model_type']
    
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])}

    train_path = r'../data/train0.9'
    val_path = r'../data/val0.9'
    train_dataset = datasets.ImageFolder(root=train_path,
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    dog_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in dog_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=119)
    with open('class_indices1.json', 'w') as json_file:
        json_file.write(json_str)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=val_path,
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    if model_type == '34':
        net = resnet34()
    elif model_type == '50':
        net = resnet50()
    elif model_type == '101':
        net = resnet101()
    elif model_type == '50_32x4d':
        net = resnext50_32x4d()
    else:
        net = resnext101_32x8d()
       
    # load pretrain weights
    model_weight_path = './resnet'+model_type+'-pre.pth'
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    # change fc layer structure
    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 120)
    
    # load trained model to fine-tune
    # model_weight_path = './resNet101_32x8d_new1.pth'
    # assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    # net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))
    
    
    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)

    best_acc = 0.0
    save_path = './resNet'+model_type+'_new2.pth'
    parameter['save_path'] = save_path
    train_steps = len(train_loader)
    train_loss = []
    val_loss = []
    val_acc = []
    t = time.time()
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        running_loss_v = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        

        # validate
        net.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                v_loss = loss_function(outputs, val_labels.to(device))
                running_loss_v += v_loss.item()
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
        val_steps = len(validate_loader)
        train_loss.append(running_loss / train_steps)
        val_loss.append(running_loss_v / val_steps)
        val_acc.append(val_accurate)
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    plt.figure(0)
    plt.plot(range(len(train_loss)), train_loss,label='train')
    plt.plot(range(len(val_loss)), val_loss, label = 'val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss_'+model_type+'6.png')
    plt.show()
    t = time.time() - t
    print_model_params(parameter, train_loss[-1],val_loss[-1], t ,best_acc)
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
    plt.figure(1)
    plt.plot(range(len(val_acc)),val_acc)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.savefig('accuracy_'+model_type+'6.png')
    plt.show()
    print('Finished Training')


if __name__ == '__main__':
    #set hyper-parameters
    parameter = {}
    #batch_size:16,32,64,128
    parameter['batch_size'] = 64
    #lr:0.00001,0.00005,0.0001,0.001
    parameter['learning_rate'] = 0.0001
    #epoch:15,20,30,40
    parameter['epochs'] = 30
    #34,50,101,50_32x4d,101_32x8d
    parameter['model_type'] = '101'
    main(parameter)