# fashion-recommender-system
A Deep Learning based Fashion Recommender System using the ResNET50


Certainly! Here’s a summary of the image retrieval system project using ResNet50, organized into sections:

---

### Project Summary: Image Retrieval System Using ResNet50

**1. Objective:**
Developed an image retrieval system that efficiently identifies and retrieves similar images from a large-scale dataset using ResNet50 for feature extraction and k-nearest neighbors (KNN) for similarity search.

**2. Data Preparation:**
- **Dataset:** Managed a dataset with 44,441 images.
- **Image Handling:** Extracted and stored image filenames from a directory for processing.

**3. Model Configuration:**
- **ResNet50 Model:** Utilized a pre-trained ResNet50 model for extracting image features.
  - **Architecture:** Leveraged a deep convolutional network with 50 layers, optimized for visual feature extraction.
  - **Feature Extraction:** Removed the classification layers to obtain a 1D feature vector for each image.

**4. Similarity Search:**
- **k-Nearest Neighbors (KNN):** Implemented KNN to find and retrieve similar images based on the extracted feature vectors.

**5. Tools & Technologies:**
- **Libraries:** TensorFlow/Keras for model handling and feature extraction, Scikit-learn for implementing KNN.
- **Environment:** Python programming language used for development and implementation.

**6. Achievements:**
- Successfully developed a scalable and efficient image retrieval system.
- Enhanced the accuracy of image searches by leveraging deep learning techniques for feature extraction.

