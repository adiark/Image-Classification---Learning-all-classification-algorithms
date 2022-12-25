# Classification: The Fashion MNIST from Sklearn and TensorFlow

Welcome to our machine learning web application for classifying images of fashion items! This application is built using streamlit and aims to help users understand the performance of different classification algorithms on a fashion dataset.

The application is divided into four main sections, which can be accessed from the sidebar menu:

1. Introduction: In this section, users can get an overview of the dataset, including information about the type of normalization that has been performed and sample data images with their label

2. SKlearn Algorithms: This section contains implementation of several classification algorithms from the SKlearn library, including Decision Tree, LightGBM, and XGBoost. Users can choose to tune the hyperparameters of the selected algorithm using sliders or use the grid search method to find the optimal parameters. Alternatively, users can apply Principal Component Analysis (PCA) to the data and then use a classification algorithm, to demonstrate the importance of dimensionality reduction in the performance of the algorithm

3. Deep Neural Networks: In this section, users can classify images using deep neural network architectures implemented in TensorFlow

4. Source Code: This section contains the source code for the application, allowing users to view and understand how the various algorithms and features have been implemented

We hope that this application helps users understand the strengths and limitations of different classification algorithms, as well as the impact of dimensionality reduction on classification performance. Thank you for using our application!

## How to run this demo
```
pip install streamlit
streamlit run streamlit_app.py

To access the hosted link - https://adiark-image-classification---learning-all-streamlit-app-ruben7.streamlit.app/

```
