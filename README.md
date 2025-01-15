Emotion Detector using *CNN*
============================== 
----
Dataset
----------

#### FER 2013

> The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image.
>> the facial expression into one of seven categories <ul> <li>0=Angry</li> <li>1=Disgust</li> <li>2=Fear</li> <li>3=Happy</li> <li>4=Sad</li> <li>5=Surprise</li> <li>6=Neutral</li></ol>
-----

#### CNN ARCHITECTURE

> **Model Input**: 48x48 grayscale images (48x48x1).
<br> **Convolutional Layers**: 4 convolutional layers with ReLU activation.
<br> **Pooling Layers**: 3 max-pooling layers to reduce spatial dimensions.
<br> **Dropout Layers**: To reduce overfitting.
<br> **Fully Connected Layers**: 2 dense layers with ReLU activation and dropout.
<br> **Output Layer**: 7 neurons with softmax activation for multi-class classification.
---

#### MODULES

> - **cv2** 
> - **keras**
> - **numpy**
> - **tensorflow**
---

#### TRAIN

> **Command** -- python TrainingModel.py
---

