# pytorch-demo
## 1.1 using pyrotch to train lenet with cifar-10
**train**:
the dataset will be downloaded automately. And the model will be
saved into folders models.


**test**:
if you want test a picture, you just need to input your file location in predict.py
```py
img = Image.open('bird.jpg')
```



## 1.2 using vgg net to train 101_ObjectCategories
the dataset can be download here http://www.vision.caltech.edu/Image_Datasets/Caltech101/

**train**:
After downloading it successfully, just extract it into torch-vgg-classification/101_ObjectCategories
and run train.py, the train process will be started.
Also you should make a val dataset.
The folder structure is like:
torch-vgg-classification
|--101_ObjectCategories
|--101_ObjectCategories_val
If you do not have a pre-trained model, you should revise the below code in train.py
```py
test_flag = True
```

**test**:
if you want test a picture, you just need to input your file location in predict.py
```py
img = Image.open('101_ObjectCategories_val/ewer/image_0004.jpg')
```
and you will get the result

