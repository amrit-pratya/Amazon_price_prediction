# Amazon_price_prediction

Product Price Prediction with Text Embeddings and Image Features

This repository explores predicting product prices by leveraging multi-modal features derived from product descriptions and images. The core logic is implemented in a Jupyter Notebook (price-prediction.ipynb).

Workflow:

  Data Loading & Preprocessing: Reads train/test CSV data. Cleans catalog_content text and extracts an "Items Per Quantity" (IPQ) feature using regex.

  Text Feature Extraction: Generates dense vector representations (embeddings) of the cleaned text using the sentence-transformers library (all-MiniLM-L6-v2).

  Image Feature Extraction: Downloads images from provided URLs, preprocesses them (resize, normalize), and extracts deep features using a pre-trained ResNet-50 model from torchvision. Handles potential image download errors.

  Feature Engineering: Concatenates text embeddings, image features, and the IPQ feature into a final feature set.

  Model Training & Optimization:

  Splits data into training and validation sets.

  Uses Optuna to perform hyperparameter tuning for a LightGBM (lgb.LGBMRegressor) model.

  Optimizes parameters based on the Symmetric Mean Absolute Percentage Error (SMAPE) metric on the validation set.

  Trains the final LightGBM model with the best parameters on the entire training dataset.

  Prediction: Generates price predictions for the test set and creates a submission.csv file.

Dependencies: pandas, numpy, torch, torchvision, sentence-transformers, lightgbm, optuna, requests, Pillow (PIL), scikit-learn.
