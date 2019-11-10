# CAD-using-CXRs
Artificial Intelligence and Machine Learning have influenced medicine in myriad ways and medical imaging is at the forefront of technological transformation. We intend to transform the medical field by providing our solution. Our solution uses the in-trend technologies to create an accurate system which will provide an accurate diagnosis. Our solution will bring change in the field of diagnostic radiology.

The Problem
Chest diseases being common, may also require a radiological diagnosis when the symptoms are critical. This calls for urgent tests, results and their diagnosis. From printing out the chest X-rays to generating a report is a time-consuming process. The time period may even be of criticality if the disease is serious and life-threatening. Sometimes it may even happen that even the most experienced doctor may miss some details and give a wrong diagnosis. In real-life what people do in such cases is they take a second opinion, which again takes a lot of precious time, which could've been used in proceeding with the treatment. Our solution aims at precision and quickness of this process.

Objective
We intend to automate the manual procedure of detecting abnormalities in the chest X-rays. This will decrease the time required to analyse the reports. Furthermore, it will also increase the accuracy of the detection of abnormality which will save a lot of cost required for other tests. The idea is to feed chest radio-graphs to the system and generate a medical report. This medical report will consist of the detected abnormality and an effective solution for the patient. The report will consist of the tests further required to be taken and the medicines will be prescribed.

Technology Requirements
 Hardware Requirements:
•	Central Processing Unit (CPU): i5 6th Generation or higher (1.4GHz (64 bits)) or an equivalent AMD processor.
•	Graphics Processing Unit (GPU): CUDA-enabled Nvidia Titan 312 GB graphics processing unit (Nvidia). 
•	RAM: 8 GB minimum, 16 GB or higher is preferable.
•	Operating System: Microsoft Windows 10 (64-bit) Pro or Home.
Software Requirements:
•	Anaconda for creating virtual environment.
•	Python IDE/Text Editor-PyCharm/Visual Studio Code.
 Machine Learning Tools:
•	CSVKit for data wrangling.
•	We will require Python libraries used for deep learning, specifically-Theano, TensorFlow.
•	Scikit-learn: It is used to implement machine learning library. Specifically, the following libraries will be used:
o	SciPy
o	Pandas
o	NumPy
o	Matplotlib


Intended Approach
We intend to approach the solution in the following order:
1.)	Pre-processing:
Pre-processing is required to avoid data imbalance, so as to create a uniformity in the data. The images available in the real world will not be suitable for our system. Hence, the first step will be to standardize the images. The images will be converted into a standard dimension. We will be converting the images into a 224x224 pixel RGB images, for the sake of reducing computational load. Furthermore, many sources of variance may also be present in the images. To lessen the variance, the contrast of the images will be increased using histogram equalization techniques, so as to make the relevant information more prominent. Pre-processing will be carried out using the python sci-kit library and csvkit.

2.)	Split the dataset:
Next, we will split the dataset into training, testing and validation datasets. As the names suggest, the respective datasets will be used for training, testing and validation purposes. 70% of the dataset will be used as a training dataset and 30% will be used for testing and validation.

3.)	Building the model:
The model will consist of a neural network for classification of the images. We intend to use Convolutional Neural Networks, as they provide the most accuracy. The network will consist of Convolution layers with filters, pooling layers, fully connected layers, and SoftMax function for classification. Here we make use of VGG16 in the Keras library. 

4.)	Training the model:
On building the model, it will be trained using 70% of the dataset. The main goal at this point will be to reduce the error- rate and increasing the accuracy. For the same purposes, weighted cost function, CNN regularisation, and data augmentation will be introduced.

5.)	Testing and Validation of the model:
Testing and validation will be done using the rest 30% of the dataset. The model will first be tested for calculating the accuracy of it. Call-backs will be introduced for the sake of saving the most optimal model. Specifically, ModelCheckpoint and EarlyStopping will be used, for saving the best model and stopping at the point of noticing a worsening generalisation gap. “categorical_crossentropy” will also be used for loss function. Finally, validation will be performed using new images.

 
