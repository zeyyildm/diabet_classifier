# 🩺 Diabetes Classification with Deep Learning (TensorFlow)

This project builds a deep-learning model using the **Pima Indians Diabetes Dataset** to predict whether a patient is likely to have diabetes based on several medical measurements.

The workflow includes **data analysis, preprocessing, train/validation/test splitting, scaling, model building with TensorFlow/Keras, training, and evaluation**.

---

## 📌 1. Project Overview

The goal of this project is to create a **neural network classifier** capable of predicting diabetes:

- **0 → No Diabetes**
- **1 → Diabetes**

A **Multi-Layer Perceptron (MLP)** model is trained using **TensorFlow/Keras** and evaluated on unseen test data to measure real performance.

---

## 📊 2. Dataset

The dataset contains **768 samples** and the following **8 input features**:

- Pregnancies  
- Glucose  
- BloodPressure  
- SkinThickness  
- Insulin  
- BMI  
- DiabetesPedigreeFunction  
- Age  

🎯 **Target Variable:**
- `Outcome` → 0 or 1

---

## 🧹 3. Data Preprocessing

### 🔸 Handling Invalid Zero Values
Some medical features (e.g., **Glucose, BloodPressure, Insulin, BMI**) contain zeros, which are not realistic physiological values.  
These zeros were treated as **missing values** and replaced with the **median** of each column.

### 🔸 Train / Validation / Test Split
Data was split into:
- **70% Training**
- **15% Validation**
- **15% Test**

`stratify=y` was used to keep class distribution stable across all splits.

### 🔸 Feature Scaling
All input features were standardized with **StandardScaler**:
- Fit only on the **training set**
- Transform applied to **validation and test sets**

This prevents **data leakage** and ensures stable training.

---

## 🧠 4. Model Architecture (TensorFlow / Keras)

A simple but effective **Fully Connected Neural Network (MLP)** was built:

- Input Layer → 8 features  
- Hidden Layer 1 → `Dense(32)`, Activation: **ReLU**  
- Hidden Layer 2 → `Dense(16)`, Activation: **ReLU**  
- Output Layer → `Dense(1)`, Activation: **Sigmoid**

### ⚙️ Loss & Optimizer
- **Loss:** Binary Crossentropy  
- **Optimizer:** Adam  
- **Metric:** Accuracy  

---

## 🚀 5. Model Training

The model was trained for **100 epochs** with **batch size = 32**, using the validation set to monitor generalization.

Training and validation accuracy were tracked across epochs to observe **model behavior and overfitting**.

---

## 📈 6. Results

### ✅ Final Results

| Metric | Score |
|--------|--------|
| Training Accuracy | ≈ 85.66% |
| Validation Accuracy | ≈ 69.56% |
| Test Accuracy | ≈ 76.72% |

### 📊 Interpretation

- The model performs **reasonably well** for this dataset  
  (≈ 76–78% accuracy is standard for Pima Diabetes).
- A gap between training and validation accuracy indicates **mild overfitting**, which is expected with small tabular datasets.
- Test accuracy reflects the model’s **real-world performance**, since test data was never used during training.

---

## 📉 7. Evaluation Visualizations

The project includes the following visualizations:

- ✅ **Confusion Matrix**

---

## 🔧 8. Possible Improvements

The following methods could improve performance:

- Add **Dropout** to reduce overfitting  
- Add **Batch Normalization**  
- Use **EarlyStopping**  
- Try more complex architectures  
- Tune hyperparameters (learning rate, layer sizes, etc.)

---

## 📎 9. Technologies Used

- 🐍 Python  
- 🧮 Pandas / NumPy  
- 📊 Scikit-Learn  
- 🧠 TensorFlow / Keras  
- 📈 Matplotlib / Seaborn  
