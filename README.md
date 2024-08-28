# fashion-recommender-system
A Deep Learning based Fashion Recommender System using the ResNET50

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
