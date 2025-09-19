
````markdown
# Chest X-Ray Pneumonia Classifier 🦠

A deep learning project to classify chest X-ray images as "Normal" or "Pneumonia" using a Convolutional Neural Network (CNN) with transfer learning. This project is deployed as an interactive web application with Streamlit.

!

## ✨ Key Features

* **Transfer Learning**: Uses a pre-trained **VGG16** model, known for its simplicity and effectiveness as a feature extractor.
* **Streamlit Web App**: A user-friendly interface for uploading images and getting instant predictions.
* **Comprehensive Evaluation**: The model's performance is thoroughly evaluated with a focus on **recall**, which is a critical metric for medical diagnoses to minimize false negatives (missed cases of pneumonia).
* **Data Augmentation**: Enhances the model's robustness by generating diverse training data.

## 🚀 Getting Started

Follow these steps to get a local copy of the project up and running.

### Prerequisites

* Python 3.8 or higher
* A local copy of the dataset (e.g., from Kaggle).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RiteshPx/X_ray_prediction_Pneumonia.git
    cd X_ray_prediction_Pneumonia
    ```

2.  **Install dependencies:**
    The required Python libraries are listed in `requirements.txt`. Install them using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

### Running the App Locally

To start the Streamlit web application on your local machine, run the following command from the project's root directory:
```bash
streamlit run app.py
````

This will open the app in your browser at `http://localhost:8501`.

## 📁 Project Structure

  * `app.py`: The main Streamlit web app script.
  * `train_model.py`: Script for training the model, generating plots, and evaluating performance.
  * `pneumonia_classifier_model.h5`: The trained model file (must be present in the same directory as `app.py`).
  * `requirements.txt`: The list of all project dependencies.

## 📈 Model Performance

The model's performance was evaluated on a held-out test set. For a medical application, **Recall** is prioritized to ensure the model catches as many true pneumonia cases as possible, as a false negative has a high cost.

**Classification Report**

```
              precision    recall  f1-score   support

      Normal       0.97      0.73      0.83       234
   Pneumonia       0.86      0.99      0.92       390

    accuracy                           0.89       624
   macro avg       0.91      0.86      0.87       624
weighted avg       0.90      0.89      0.89       624
```

*Note: Values shown are for a hypothetical model. Your actual results may vary.*

**Confusion Matrix**
| | **Predicted Normal** | **Predicted Pneumonia** |
| :--- | :--- | :--- |
| **Actual Normal** | 170 (TN) | 64 (FP) |
| **Actual Pneumonia** | 5 (FN) | 385 (TP) |
             


*Note: Values shown are for a hypothetical model. Your actual results may vary.*

## 🌐 Live Demo

You can try the live version of this app here:
https://xray-prediction-pneumonia-or-normal.streamlit.app/

## 🤝 Contributing

This is an open-source project. Feel free to fork the repository, open an issue, or submit a pull request with new features or bug fixes.

## 📧 Contact

**Name:** Ritesh Parmar
**Email:** parmaritesh17@gmail.com
**GitHub:** https://github.com/RiteshPx/X_ray_prediction_Pneumonia/

```
```


