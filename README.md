# BASICS OF COMPUTER VISION WITH PyTorch

This repository contains various computer vision projects implemented using Python and the popular library PyTorch. Below is a brief overview of each project:

## Image Classification using CNN
<a src="https://colab.research.google.com/drive/1n4RcAcw0uMODD4nHE8P2QF9Lr_SVf1QF?usp=sharing">
   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" alt="Open In Colab"/>
</a>

Implemented a convolutional neural network (CNN) from scratch using PyTorch to classify images. The model architecture includes convolutional layers, pooling layers, and fully connected layers. The CIFRA10 dataset was used for training and evaluation.

![image_classification_from_scratch.png](https://miro.medium.com/max/1100/1*SZnidBt7CQ4Xqcag6rd8Ew.png)

## Image Classification using VGG16
<a src="https://drive.google.com/file/d/10D1i75sMBp7aaNdg9VpGlNbNXzz88EY4/view?usp=sharing">
   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" alt="Open In Colab"/>
</a>

In this project, a classification model was trained using transfer learning techniques. A pre-trained VGG16 model was used as the base model, and it was fine-tuned on a custom dataset (CIFAR10) to classify images.

![image_classification_from_scratch.png](https://miro.medium.com/v2/resize:fit:1400/1*NNifzsJ7tD2kAfBXt3AzEg.png)

## Evaluation of an object detection model
<a src="https://drive.google.com/file/d/10D1i75sMBp7aaNdg9VpGlNbNXzz88EY4/view?usp=sharing">
   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" alt="Open In Colab"/>
</a>

In this notebook, we implement the evaluation of YOLO (You Only Look Once) models using the mean Average Precision (mAP) metric. mAP is a widely used metric in object detection tasks, including YOLO models. It provides a comprehensive measure of the model's performance by considering both precision and recall across all classes and varying levels of confidence thresholds. By computing mAP, we gain insights into how well our YOLO model detects objects in an image, making it a crucial tool for evaluating and improving object detection systems.

![YOLOv5 architecture](https://deci.ai/wp-content/uploads/2022/11/yolov6-yolov5-yolox-blog-header.jpg)

## Face Denoising
<a src="https://colab.research.google.com/drive/1tyLJVYecmgGOZcxWjyv-gbal7ZMv3P7t?usp=sharing">
   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" alt="Open In Colab"/>
</a>

Implemented a face denoising algorithm using OpenCV and deep learning. The project aims to remove noise from facial images while preserving important features.

![Face denoising image](https://media.springernature.com/m685/springer-static/image/art%3A10.1007%2Fs42979-022-01042-y/MediaObjects/42979_2022_1042_Fig12_HTML.png)

## Vehicle Reidentification System

<a src="https://drive.google.com/file/d/1G0Xe140_26WGiYzRV7Z3YCVPIfYhXkvW/view?usp=sharing">
   <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" alt="Open In Colab"/>
</a>

Here, we are developping a re-identification system for vehicles using image retrieval. The system is capable of recognizing and tracking vehicles across different camera feeds.

![alt text](https://production-media.paperswithcode.com/tasks/vehicleReID_KT5l9ol.jpg)

## Installation

Clone the repository:

   ```bash
   git clone https://github.com/beria-kalpelbe/basic-computer-vision-tasks.git