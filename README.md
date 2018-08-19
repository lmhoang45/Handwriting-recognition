# Handwriting-recognition
The code for handwritten numbers recognition using PyTorch, written in Python. There are 2 .py files in the repo, 
the first one create_model.py is to create and save the model in .pt file to a directory.
The second one handwriting_mnist.py would call the .pt model and predict the input image captured from the camera. \
Prediction output will be showed on the terminal.\
MNIST dataset is used (http://yann.lecun.com/exdb/mnist/)  \
When run handwriting_mnist.py: press 'a' to capture the image from the camera and 'z' to exit the program.    
\
![alt get](https://github.com/lmhoang45/Handwriting-recognition/blob/master/hinh.jpg)  


# Prerequisites
Required frameworks and libraries:
torch - torchvision - matplotlib - numpy - cv2 - scipy - pylab - skimage

# Reference
https://github.com/pytorch/examples/blob/master/mnist/main.py
