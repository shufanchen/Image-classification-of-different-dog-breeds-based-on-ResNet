import os
import json
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import pandas
from model import *
import time
import datetime


def predict(image_path,model_type):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    # load image
    img_path = image_path
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)


    if model_type == '34':
        model = resnet34(num_classes=120).to(device)
    elif model_type == '50':
        model = resnet50(num_classes=120).to(device)
    elif model_type == '101':
        model = resnet101(num_classes=120).to(device)
    elif model_type == '50_32x4d':
        model = resnext50_32x4d(num_classes=120).to(device)
    elif model_type == '101_32x8d':
        model = resnext101_32x8d(num_classes=120).to(device)
    # load model weights
    #weights_path = './resNet'+model_type+'.pth'
    weights_path = 'resNet101_32x8d_new2'+'.pth'
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    # prediction
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        #predict_cla = torch.argmax(predict).numpy()
    return predict.detach().numpy()



def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
            
if __name__ == '__main__':
    json_path = './class_indices1.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)
    columns = ['id']
    for i in range(120):
        columns.append(class_indict[str(i)])
    prediction = pd.DataFrame(columns=columns)
    #prediction.to_csv('result'+'_newhope1.csv')
    path = '../data/test'
    #predict('00a3edd22dc7859c487a64777fc8d093.jpg')
    test_list = []
    listdir(path, test_list)
    model_type = '101_32x8d'
    t = time.time()
    for i in test_list:
        temp = []
        id = i.split('/')[-1][:-4]
        temp.append(id)
        result = predict(i,model_type)
        result = list(result.flatten())
        temp = temp + result
        temp = pd.DataFrame(data = [temp],columns=columns)
        prediction = pd.concat([prediction,temp], ignore_index=True)
        print(id+'\t'+'finished'+'\t'+str(datetime.datetime.now()))
    prediction.to_csv('result_'+model_type+'_newhope2.csv')
    t = time.time()-t
    print('time cost:'+str(t))
    
        
        