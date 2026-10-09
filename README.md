# AI Phishing Website Detection

<p align="center">
  <img src="https://img.shields.io/badge/Python-3-blue?logo=python&logoColor=white" alt="Python 3" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Random-Forest-green" alt="Random Forest" />
  <img src="https://img.shields.io/badge/Tkinter-GUI-4B8BBE" alt="Tkinter GUI" />
</p>

A machine-learning project that classifies URLs as phishing or legitimate using only URL text features. The project trains a Random Forest model on the UCI phishing dataset and provides both a desktop app and a quick command-line prediction tool.

Repository: https://github.com/jyotisah-78/AI-Phishing-Website-Detection-

## Overview

This project focuses on phishing detection by analyzing the structure and content of a URL rather than opening the website itself. It uses a dataset of phishing and legitimate URLs, extracts meaningful URL-based features, and trains a machine-learning model to classify suspicious links.

The system is designed to be simple, educational, and easy to run locally.

## Features

- Detects phishing URLs using URL text patterns and structure
- Uses a Random Forest classifier for prediction
- Extracts URL-based features such as length, special characters, domain patterns, and other lexical indicators
- Splits training and testing data carefully to avoid hostname leakage
- Saves the trained model and feature metadata in the `model/` folder
- Includes a Tkinter-based desktop app for user interaction
- Includes a terminal-based predictor for quick checks

## Tech Stack

- Python 3
- scikit-learn
- pandas
- NumPy
- joblib
- Tkinter
- UCI phishing URL dataset

## Project Structure

```text
.
├── data/
│   ├── dataset.csv
│   ├── dataset_clean.csv
│   ├── train.csv
│   └── test.csv
├── model/
│   ├── feature_names.txt
│   ├── metrics.json
│   └── phishing_model.joblib
├── screenshots/
├── src/
│   ├── app.py
│   ├── clean_data.py
│   ├── download_data.py
│   ├── feature_extraction.py
│   ├── predict.py
│   ├── split_data.py
│   ├── test_project.py
│   └── train_model.py
├── .gitignore
├── make_submission_zip.sh
├── README.md
├── requirements.txt
└── .github/
```

## Installation

```bash
git clone https://github.com/jyotisah-78/AI-Phishing-Website-Detection-.git
cd AI-Phishing-Website-Detection-
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run locally

### Run the desktop app

```bash
python src/app.py
```

### Quick prediction from the terminal

```bash
python src/predict.py "https://www.example.com"
```

## Run tests

```bash
python src/test_project.py
```

## Model Workflow

1. Download the phishing dataset.
2. Clean and prepare the URL data.
3. Extract URL-based features.
4. Split the data into training and test sets.
5. Train the Random Forest model.
6. Evaluate the model and save results.
7. Use the saved model to classify new URLs.

## Screenshot Section

### Desktop app interface

![AI Phishing Website Detection App](https://raw.githubusercontent.com/jyotisah-78/AI-Phishing-Website-Detection-/main/screenshots/app.png)

### Example prediction

![Example URL Check](https://raw.githubusercontent.com/jyotisah-78/AI-Phishing-Website-Detection-/main/screenshots/prediction.png)

> If screenshots are added later, they can be placed in the `screenshots/` folder and linked here for a cleaner GitHub presentation.

## Model Performance

The model is evaluated on a test set and produces accuracy, precision, recall, and F1-score metrics. The prediction is intended as a helpful estimate rather than a guarantee.

| Metric | Value |
|---|---:|
| Accuracy | 0.9678 |
| Precision (phishing) | 0.9782 |
| Recall (phishing) | 0.9465 |
| F1-score (phishing) | 0.9621 |
| Training-set accuracy | 0.9676 |
| Majority-class baseline | 0.5687 |

## Architecture Overview

```text
User URL Input
      ↓
URL Feature Extraction
      ↓
Data Cleaning / Grouped Split
      ↓
Random Forest Training
      ↓
Model Evaluation
      ↓
Prediction Output (Phishing / Legitimate)
```

This architecture keeps the solution simple and transparent: URL strings are transformed into measurable features, the model learns patterns from labeled data, and predictions are generated for new or suspicious links.

## Conclusion

This project demonstrates a practical and effective approach to phishing detection using machine learning and URL-based features. It offers a lightweight, accessible solution for identifying suspicious links without visiting or scanning the website itself. The model is useful for educational and research purposes, and it provides a strong foundation for future improvements such as expanded feature engineering, real-time detection, or integration with browser-based security tools.

## Notes

- The model predicts phishing based only on URL text.
- It does not open websites or inspect page content.
- It cannot guarantee that a website is safe or malicious.
- This project is intended for educational and research use.

## License

This project is intended for educational and research use.

## Acknowledgements

- UCI Machine Learning Repository
- PhiUSIIL phishing dataset
- scikit-learn community
