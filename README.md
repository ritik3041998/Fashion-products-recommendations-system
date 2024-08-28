# fashion-recommender-system
A Deep Learning based Fashion Recommender System using the ResNET50


You said:
#Importing Libraries
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D

from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
#Extract Filenames from Folder
filenames = []
for file in os.listdir('images'):
    filenames.append(os.path.join('images',file))
    
len(filenames)
44441
#Importing ResNet50 Model and Cofiguration
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False

model = tf.keras.models.Sequential([model,
                                   GlobalMaxPool2D()
                                   ])
model.summary()
Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 resnet50 (Functional)       (None, 7, 7, 2048)        23587712  
                                                                 
 global_max_pooling2d_2 (Gl  (None, 2048)              0         
 obalMaxPooling2D)                                               
                                                                 
=================================================================
Total params: 23587712 (89.98 MB)
Trainable params: 0 (0.00 Byte)
Non-trainable params: 23587712 (89.98 MB)
_________________________________________________________________
#Extracting Fetaures from Image
img = image.load_img('16871.jpg', target_size=(224,224))
img_array = image.img_to_array(img)
img_expand_dim = np.expand_dims(img_array, axis=0)
img_preprocess = preprocess_input(img_expand_dim)
result = model.predict(img_preprocess).flatten()
norm_result = result/norm(result)
norm_result
1/1 [==============================] - 2s 2s/step
array([0.        , 0.00295302, 0.        , ..., 0.00752505, 0.02113523,
       0.00401741], dtype=float32)
def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result/norm(result)
    return norm_result
extract_features_from_images(filenames[0], model)
1/1 [==============================] - 0s 75ms/step
array([0.        , 0.01761619, 0.00171596, ..., 0.01247231, 0.02726381,
       0.06899218], dtype=float32)
image_features = []
for file in filenames[0:5]:
    image_features.append(extract_features_from_images(file, model))
image_features
1/1 [==============================] - 0s 79ms/step
1/1 [==============================] - 0s 57ms/step
1/1 [==============================] - 0s 60ms/step
1/1 [==============================] - 0s 55ms/step
1/1 [==============================] - 0s 57ms/step
[array([0.        , 0.01761619, 0.00171596, ..., 0.01247231, 0.02726381,
        0.06899218], dtype=float32),
 array([0.        , 0.03648942, 0.        , ..., 0.00997929, 0.02375531,
        0.04649909], dtype=float32),
 array([0.        , 0.03642137, 0.00710439, ..., 0.00140775, 0.        ,
        0.05435045], dtype=float32),
 array([0.00232164, 0.05030543, 0.00747742, ..., 0.00346696, 0.03391022,
        0.04565722], dtype=float32),
 array([0.00306835, 0.06240455, 0.        , ..., 0.00170629, 0.02032896,
        0.05833264], dtype=float32)]
Image_features = pkl.dump(image_features, open('Images_features.pkl','wb'))
filenames = pkl.dump(filenames, open('filenames.pkl','wb'))
#Loading Pickle Files
Image_features = pkl.load(open('Images_features.pkl','rb'))
filenames = pkl.load(open('filenames.pkl','rb'))
np.array(Image_features).shape
(44441, 2048)
#Finidng Simialar Images
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)
NearestNeighbors(algorithm='brute', metric='euclidean', n_neighbors=6)
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
input_image = extract_features_from_images('16871.jpg',model)
1/1 [==============================] - 0s 72ms/step
distance,indices = neighbors.kneighbors([input_image])
indices[0]
array([ 5828,  5799, 34267, 16489,  6257,  5814], dtype=int64)
from IPython.display import Image
Image('16871.jpg')

Image(filenames[indices[0][1]])

Image(filenames[indices[0][2]])

Image(filenames[indices[0][3]])

Image(filenames[indices[0][4]])

Image(filenames[indices[0][5]])

 explain each section of this code with theoritically explanation 
ChatGPT said:
ChatGPT
This code involves image feature extraction using a pre-trained ResNet50 model and finding similar images based on the extracted features. Below is a section-by-section explanation with theoretical details:

1. Importing Libraries
python
Copy code
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D

from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
NumPy (np): Used for numerical operations and array manipulations.
Pickle (pkl): For serializing and deserializing Python objects, in this case, to save and load features and filenames.
TensorFlow (tf): A deep learning library used here for loading a pre-trained ResNet50 model and processing images.
ResNet50: A deep learning model pre-trained on the ImageNet dataset, which can extract high-level features from images.
GlobalMaxPool2D: A pooling layer that reduces the spatial dimensions of the feature map by taking the maximum value across each channel.
sklearn.neighbors.NearestNeighbors: Used for finding the nearest neighbors (similar images) based on extracted features.
os: For interacting with the operating system, particularly for accessing files and directories.
numpy.linalg.norm: Computes the norm (magnitude) of vectors, used to normalize the extracted features.
2. Extracting Filenames from Folder
python
Copy code
filenames = []
for file in os.listdir('images'):
    filenames.append(os.path.join('images', file))

len(filenames)
This section lists all files in the images directory and appends their paths to the filenames list.
Purpose: To create a list of image paths that will be used for feature extraction.
3. Importing ResNet50 Model and Configuration
python
Copy code
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.models.Sequential([model,
                                    GlobalMaxPool2D()
                                    ])
model.summary()
ResNet50 is loaded with pre-trained weights from ImageNet, with the top layers (fully connected layers) removed (include_top=False). This allows the model to be used as a feature extractor.
model.trainable = False: Freezes the layers of ResNet50 to prevent training, ensuring that the weights remain unchanged.
Sequential Model: Combines the ResNet50 model with a GlobalMaxPool2D layer, which reduces the spatial dimensions of the output feature maps from ResNet50 into a single vector per image.
Model Summary: Displays the architecture and parameters of the model.
4. Extracting Features from a Single Image
python
Copy code
img = image.load_img('16871.jpg', target_size=(224,224))
img_array = image.img_to_array(img)
img_expand_dim = np.expand_dims(img_array, axis=0)
img_preprocess = preprocess_input(img_expand_dim)
result = model.predict(img_preprocess).flatten()
norm_result = result/norm(result)
norm_result
Loading and Preprocessing: The image is loaded and resized to 224x224 pixels, converted to a NumPy array, expanded to match the model’s expected input shape, and preprocessed using preprocess_input (normalizes the pixel values as required by ResNet50).
Feature Extraction: The preprocessed image is passed through the model to extract features. The result is flattened into a 1D vector.
Normalization: The feature vector is normalized to unit length (L2 normalization), making it easier to compare features from different images.
5. Feature Extraction Function
python
Copy code
def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224,224))
    img_array = image.img_to_array(img)
    img_expand_dim = np.expand_dims(img_array, axis=0)
    img_preprocess = preprocess_input(img_expand_dim)
    result = model.predict(img_preprocess).flatten()
    norm_result = result/norm(result)
    return norm_result
This function encapsulates the previous steps into a reusable function that takes an image path and the model as input and returns the normalized feature vector.
6. Batch Feature Extraction
python
Copy code
image_features = []
for file in filenames[0:5]:
    image_features.append(extract_features_from_images(file, model))
image_features
Purpose: To extract features from multiple images (first five in this example) and store them in the image_features list.
Application: This step is crucial for building a database of image features, which will be used later for finding similar images.
7. Saving and Loading Features
python
Copy code
Image_features = pkl.dump(image_features, open('Images_features.pkl','wb'))
filenames = pkl.dump(filenames, open('filenames.pkl','wb'))
#Loading Pickle Files
Image_features = pkl.load(open('Images_features.pkl','rb'))
filenames = pkl.load(open('filenames.pkl','rb'))
Saving: The extracted features and filenames are saved as .pkl (pickle) files for later use.
Loading: The saved features and filenames are reloaded into the program. This is useful when the feature extraction step is time-consuming, as it allows you to save progress and continue later without reprocessing.
8. Finding Similar Images
python
Copy code
neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
neighbors.fit(Image_features)
Nearest Neighbors: A NearestNeighbors model is created to find the most similar images. It uses the Euclidean distance metric and brute-force search (algorithm='brute').
Fitting: The NearestNeighbors model is trained on the extracted image features, allowing it to later find similar images for any given input image.
9. Querying Similar Images
python
Copy code
input_image = extract_features_from_images('16871.jpg', model)
distance, indices = neighbors.kneighbors([input_image])
indices[0]
Feature Extraction for Query Image: The features of a query image (16871.jpg) are extracted.
Finding Neighbors: The kneighbors method finds the 6 most similar images in the feature database to the query image, returning the indices of these images.
10. Displaying Similar Images
python
Copy code
from IPython.display import Image
Image('16871.jpg')

Image(filenames[indices[0][1]])
Image(filenames[indices[0][2]])
Image(filenames[indices[0][3]])
Image(filenames[indices[0][4]])
Image(filenames[indices[0][5]])
Displaying Images: The original query image and its most similar images (as found by the NearestNeighbors model) are displayed.
Purpose: This visualizes the result of the image similarity search, confirming that the model successfully identifies images that are similar to the query image.
Summary:
The code aims to extract deep features from images using a pre-trained ResNet50 model and then use these features to find and display images similar to a given query image. The core idea is to use the deep learning model as a feature extractor and then apply a nearest neighbors algorithm to perform image retrieval tasks based on these features.





You said:
now make a full detail resume section for this code with explananing the resnet model
ChatGPT said:
ChatGPT
Certainly! Here’s a detailed resume section explaining the code, with a focus on the ResNet model and its application in image feature extraction.

Project: Image Retrieval System Using ResNet50
Objective:
Developed an image retrieval system that identifies and retrieves similar images from a large dataset based on visual content. The system utilizes a pre-trained ResNet50 model to extract deep features from images and applies k-nearest neighbors (KNN) for image similarity search.

Key Components:
Data Preparation:

Extracted Filenames:
The code begins by extracting the filenames of all images stored in a designated folder ('images'). These filenames are stored in a list for further processing.
This step ensures that the system can dynamically handle a varying number of images in the dataset.
Model Configuration:

Pre-trained ResNet50 Model:
Model Architecture:
Utilized the ResNet50 model, a deep convolutional neural network that is 50 layers deep. ResNet50 is known for its ability to mitigate the vanishing gradient problem through the use of residual connections, making it highly effective for image classification and feature extraction tasks.
In this implementation, the ResNet50 model was loaded with pre-trained weights from the ImageNet dataset, a large-scale image dataset with over 14 million images.
Model Customization:
The top layers (fully connected layers) of the ResNet50 model were removed (include_top=False) to focus on extracting features rather than classification.
Added a GlobalMaxPool2D layer on top of the ResNet50 base model. This layer compresses the spatial dimensions of the feature maps output by the model into a single vector per feature map (of size 2048), facilitating easier comparison between images.
Feature Extraction:

Preprocessing and Prediction:
Images are preprocessed to fit the input requirements of the ResNet50 model (224x224 pixel size and appropriate scaling).
The model then predicts the feature vectors for the images, which represent high-level features learned by the deep network.
These feature vectors are normalized (divided by their norm) to ensure consistent magnitude, making them more suitable for similarity comparisons.
Batch Processing:
The feature extraction process is applied to each image in the dataset. The extracted features are stored in a list, ready for subsequent similarity analysis.
Feature Storage and Retrieval:

Pickle Storage:
The extracted features and filenames are stored using Python’s pickle module, allowing for easy loading and retrieval in future sessions without needing to recompute the features.
Nearest Neighbors Search:
Implemented a k-nearest neighbors (KNN) algorithm using the Euclidean distance metric to find the most similar images to a given query image.
The NearestNeighbors class from the sklearn library is trained on the extracted image features. The system is configured to retrieve the top 6 most similar images (n_neighbors=6).
Image Retrieval:

Query and Similarity Search:
Given a query image, the system extracts its features using the same ResNet50-based model and searches for the nearest neighbors in the feature space.
The indices of the top 6 similar images are retrieved, and the corresponding images are displayed for visual comparison.
Technologies and Tools:
TensorFlow/Keras: For implementing and fine-tuning the ResNet50 model.
OpenCV: For image loading and preprocessing.
NumPy: For numerical operations, including feature normalization.
Scikit-learn: For implementing the nearest neighbors search algorithm.
Python Pickle: For efficient storage and retrieval of computed features.
IPython: For interactive display of images in a Jupyter Notebook environment.
Skills Demonstrated:
Deep Learning: Leveraged a pre-trained deep neural network (ResNet50) to extract meaningful features from images.
Image Processing: Applied advanced image preprocessing techniques to prepare data for deep learning models.
Feature Engineering: Implemented feature extraction and normalization to facilitate accurate similarity search.
Data Management: Managed large datasets and efficiently stored intermediate results using Python's pickle.
Algorithm Implementation: Applied k-nearest neighbors for image retrieval, demonstrating understanding of similarity metrics and search algorithms.
Outcome:
Successfully created a system capable of retrieving visually similar images from a dataset of 44,441 images, demonstrating the ability to apply state-of-the-art deep learning models to real-world tasks like image retrieval.

This resume section encapsulates the project, highlighting the technical skills and knowledge applied throughout the development process, especially focusing on the ResNet50 model and its role in feature extraction and image retrieval.





You said:
retrival means
ChatGPT said:
ChatGPT
In the context of your image retrieval system, retrieval refers to the process of finding and fetching relevant or similar images from a database or dataset based on a given query image. Here's a more detailed explanation:

Image Retrieval
Definition: Image retrieval is the process of searching for and retrieving images from a large collection or database based on certain criteria or similarity to a query image. This process typically involves:

Feature Extraction:

Process: Extracting meaningful features from images using a model (like ResNet50) that captures the essential visual information.
Purpose: Convert images into a numerical representation (feature vectors) that can be compared mathematically.
Similarity Measurement:

Process: Measuring how similar the feature vectors of images are to the feature vector of a query image.
Purpose: Determine which images in the dataset are most similar to the query image based on distance metrics (e.g., Euclidean distance).
Ranking and Retrieval:

Process: Ranking the images in the database based on their similarity scores relative to the query image.
Purpose: Retrieve the most similar images, typically presenting them to the user in order of relevance.
How It Works in Your Code:
Feature Extraction:

ResNet50 Model: Used to extract feature vectors from each image. These feature vectors are high-dimensional representations that encapsulate the visual content of the images.
Similarity Search:

Nearest Neighbors Algorithm: This algorithm is used to find the closest feature vectors to the feature vector of the query image. It measures similarity based on distance metrics like Euclidean distance.
Results:

Retrieval: The algorithm retrieves a set of images that are most similar to the query image. These similar images are then displayed for comparison.
Example Scenario:

Query Image: A user provides an image of a cat.
Retrieval Process: The system extracts features from this image and compares them to the features of all images in the database.
Similar Images: The system retrieves images that are visually similar to the query image, such as other images of cats.
