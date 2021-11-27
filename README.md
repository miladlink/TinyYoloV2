# Pytorch Tiny YoloV2 implementation from scratch

## Description

* YOLO or `You Only Look Once`, is a popular **real-time object detection** algorithm.

* YOLO combines what was once a multi-step process, using a **single** neural network to perform both `classification` and `prediction` of bounding boxes for detected objects.

* network divides the image into regions and predicts bounding boxes and probabilities for each region.

![image](https://user-images.githubusercontent.com/81680367/143594618-c5afe17c-004b-4206-bbd1-5215fd05c935.png)

* In this repo implemented one of the **simple** and **fast** version of YOLO from scratch.

## Dataset

* Pascal-VOC dataset is the one of the collections for object detection

* Pretrained weights in this implemetation are based on training yolo team on VOC dataset

## Pretrained Weights

* You can check the [yolo website](https://pjreddie.com/darknet/yolov2/) for defferent variation pretrained weights for different sizes or datasets.

* Download used weights in this project from [here](https://pjreddie.com/media/files/yolov2-tiny-voc.weights)

* Loading this weights is not like conventional methods (like loading .pth, .pt, ... formats) so they should put on the model's body

## Result images

* Predictions and bounding boxes not so much accurate but it's fast

![person](https://user-images.githubusercontent.com/81680367/143678243-d5578b06-6b96-4739-9c0f-eba6d9d7b74b.png)

![dog](https://user-images.githubusercontent.com/81680367/143678251-fa51ab97-662d-4677-ba5b-435d4ced0ab0.png)

![horses](https://user-images.githubusercontent.com/81680367/143678256-29fa574a-523a-4c00-8a3f-e791516c7004.png)


**Note**: predicted objects is around defined class names so it can not predict out of this such as below image

![giraffe](https://user-images.githubusercontent.com/81680367/143621916-9376b500-cb09-4ec4-a6c8-2f97c826b7a2.jpg)