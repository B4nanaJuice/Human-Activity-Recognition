# Human Activity Recognition (HAR) Using WiFi CSI

This project focuses on **Human Activity Recognition (HAR)** using **Channel State Information (CSI)** extracted from WiFi signals. The primary goal of the current phase is to detect **whether a fall event has occurred or not**, using deep learning models.

---

## 🚀 Project Overview

Traditional HAR systems rely on wearable sensors or cameras. In contrast, this project explores a **device-free**, **privacy-preserving**, and **non-invasive** approach using WiFi signals. By analyzing variations in CSI, we aim to recognize human activities and, more specifically, detect falls with a certain accuracy.

Our approach uses:

* 📡 **WiFi Channel State Information (CSI)** as input features
* 🧠 **Convolutional Neural Networks (CNNs)** for fall/no-fall classification

---

## 🏗️ Current Features

* **CNN-Based Classifier**: A deep learning model trained to detect falls.
* **Binary Activity Detection**: Output is either `Fall` or `No Fall`.

---

## 🧪 Methodology

### 1. **CSI Collection**

As this project aims on fall recognition, we will use free CSI database online.

### 2. **Model Architecture**

We use CNNs to capture spatial and temporal structures in the CSI matrices. The model outputs a binary classification:

* `1` → Fall detected
* `0` → No fall detected

---

## 📈 Performance Metrics
[Todo]

---

## 🔧 Requirements

* Python 3.9+

Install dependencies:

```bash
pip install -r requirements.txt
```
