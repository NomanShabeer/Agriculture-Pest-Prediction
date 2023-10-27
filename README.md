# Pest Prediction in Agriculture

**Abstract:** The "Pest Prediction in Agriculture" project aims to leverage machine learning techniques to predict and manage pest infestations in agricultural fields. Pests can cause significant damage to crops, leading to reduced yields and economic losses. This project addresses this challenge by developing a predictive model that can forecast pest occurrences based on various environmental and crop-related factors. By identifying potential pest outbreaks in advance, farmers can implement targeted interventions, such as pesticide application or crop rotation, to mitigate the impact of pests and optimize crop production.

**Getting Started:**

Before delving into the predictive prowess of our pest prediction model, ensure you follow the sequence of cells from start to finish, tapping into the full potential of a GPU accelerator for swift computations. It's worth noting that a simple user interface has been implemented, offering you the opportunity to test the model at your convenience. Moreover, your insights, feedback, and suggestions to elevate this notebook are not only welcomed but wholeheartedly cherished. Let's embark on this agricultural journey together, armed with innovation and collaboration. Here's to a thriving and resilient agricultural landscape ahead! Cheers!

**Libraries and Tools:**

The project code utilizes the following libraries and tools:
- `numpy`: For numerical operations and array handling.
- `pandas`: For data manipulation and analysis.
- `tensorflow`: For machine learning and deep learning.
- `matplotlib` and `seaborn`: For data visualization.
- `cv2`: OpenCV library for image processing.
- Various TensorFlow components such as `keras`, `ImageDataGenerator`, and more.
- `scikit-learn` for metrics and data preprocessing.

**Data Loading:**

The project loads data from the [Agricultural-pests-image-dataset](https://www.kaggle.com/datasets/vencerlanz09/agricultural-pests-image-dataset) and prepares it for model training and evaluation.

**Data Visualization:**

The code includes data visualization using matplotlib and seaborn to help you understand the dataset and its characteristics.
https://github.com/NomanShabeer/Agriculture-Pest-Prediction/blob/main/Capture.PNG
![image](https://github.com/NomanShabeer/Agriculture-Pest-Prediction/assets/144251597/ce1c2349-7631-42b6-960f-7d1f344288b7)


**Model:**

The model architecture includes a convolutional neural network based on EfficientNetB4, which has been pre-trained on ImageNet. This is followed by fully connected layers for classification. The code provides a summary of the model's layers and parameters.

**Training and Evaluation:**

The code involves training the model on the provided dataset. It uses data augmentation techniques and CutMix for improved performance. The model's performance is evaluated using various metrics such as accuracy and top-k accuracy.

**Prediction on Test Data:**

The code predicts labels for test data and evaluates the model's performance using metrics like accuracy and a confusion matrix.

**Results:**
https://github.com/NomanShabeer/Agriculture-Pest-Prediction/blob/main/Result.PNG
![image](https://github.com/NomanShabeer/Agriculture-Pest-Prediction/assets/144251597/4e31aac8-0060-4c6c-8f40-c8ae50567b05)


**Data Visualization and User Interface:**

The code provides data visualization for test data and offers a user interface (UI) for running predictions using the Gradio library.

**Requirements:**

Make sure to install the required libraries and dependencies by running the following:

```
!pip install numpy pandas tensorflow matplotlib opencv-python seaborn scikit-learn gradio
```

**Usage:**

1. Execute the code cells sequentially for data loading, model training, and evaluation.
2. To use the user interface, ensure Gradio is installed and run the provided UI section of the code.


## Acknowledgments

If you found this project valuable or insightful, please consider giving it a star on GitHub. Your support and feedback are greatly appreciated!

## Author

- Noman Shabbir
- GitHub: [Noman Shabeer](https://github.com/NomanShabeer)

## Contact

If you have any questions, suggestions, or issues related to this project, please do not hesitate to contact me at [engr.nomanshabbir@gmail.com]

We hope this README provides you with a comprehensive understanding of the "Pest Prediction in Agriculture" project and how to use the provided code. Enjoy working on this project and helping to enhance agricultural practices through technology!
