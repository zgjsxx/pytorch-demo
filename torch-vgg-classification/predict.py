import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from model import vgg
import json
from PIL import Image

if __name__ == '__main__':

    with open('class_indices.json', 'r') as f:
        json_str = f.read()
    json_str = json.loads(json_str)

    classes = list()
    for key in json_str:
        classes.append(json_str[key])
    print(classes)
    classes.sort()
    print(classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = "vgg16"
    net = vgg(model_name=model_name, num_classes=102, init_weights=True)
    net.to(device)

    checkpoint = torch.load('pre_train_models/vgg16-0.ptn')  # 加载模型
    net.load_state_dict(checkpoint)

    model = net.to(device)
    model.eval()  # 把模型转为test模式

    img = Image.open('101_ObjectCategories_val/ewer/image_0004.jpg')
    #img = Image.open('test.jpg')

    transform = transforms.Compose(
        [transforms.Resize((224, 224)),  # 跟train.py预处理步骤不同的地方 将图片大小固定
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    img = transform(img)
    img = img.to(device)
    img = img.unsqueeze(0)
    output = model(img)
    prob = F.softmax(output, dim=1)  # prob是10个分类的概率
    print(prob)
    value, predicted = torch.max(output.data, 1)
    print(predicted.item())
    print(value)
    pred_class = classes[predicted.item()]
    print(pred_class)















