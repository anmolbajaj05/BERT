# BERT Sentiment Analysis on IMDb Dataset

## Objective
Our goal with this project is to fine-tune BERT, one of the most advanced language models, to classify sentiment in movie reviews from the IMDb dataset. This dataset provides a balanced set of positive and negative reviews, making it ideal for sentiment analysis tasks. By leveraging BERT’s sophisticated language understanding, we aim to achieve high accuracy in classifying these reviews, demonstrating BERT’s adaptability to real-world data.

## IMDb Dataset
The IMDb dataset, sourced from Stanford AI’s repository, contains 50,000 highly polarised movie reviews. These reviews are divided equally into training and testing sets, with a balanced distribution of positive and negative sentiments. To streamline the loading and preprocessing of this data, we employed the `load_dataset` function from the Hugging Face `datasets` library. This library offers a seamless way to access and manage large datasets, enabling efficient data handling for machine learning tasks.

### Processing Steps
To prepare the IMDb data for BERT, we followed a structured set of processing steps:
1. **Tokenisation**: Using BERT’s pre-trained tokenizer, each review was converted into token IDs, allowing the model to interpret text as numerical inputs.
2. **Padding and Truncation**: Given the varied lengths of movie reviews, each review was either padded or truncated to a fixed length of 512 tokens. This ensures consistency across samples, allowing the model to process them uniformly.

## Project Setup

### Environment
We used Python 3.7+ for this project, maintaining compatibility with the original BERT repository requirements. The primary framework for training and evaluation is TensorFlow 2.x, chosen for its integration with BERT and robust support for deep learning. Additionally, we incorporated essential libraries like `transformers`, `numpy`, `scikit-learn`, and `pandas` to facilitate data preprocessing, model training, and evaluation.

### Infrastructure Considerations
This project was executed on a CPU-only setup due to hardware constraints. While CPUs are effective for experimentation, training BERT without GPU support can be time-intensive. For optimal performance, running this model on a GPU- or TPU-enabled environment is recommended, as it would considerably reduce training time and improve efficiency.

## Code Modifications
We adapted the original BERT code from Google’s GitHub repository to align with our project’s specific requirements. Adjustments included:
- **File Path Configuration**: Ensuring all file paths matched our local dataset locations for seamless loading.
- **Parameter and Hyperparameter Tuning**: Modifying key parameters to align with TensorFlow 2.x compatibility and our specific dataset requirements.

These adjustments were minimal yet critical to adapting BERT for sentiment analysis on IMDb data, highlighting the flexibility of the original code in supporting different applications and datasets.

## Key Components and Steps
- **Tokenisation and Preprocessing**: Our approach involves transforming raw text into token IDs that BERT can process, a step fundamental to working with transformer models.
- **Model Training and Evaluation**: BERT’s powerful language understanding is fine-tuned on the IMDb data to optimise performance for sentiment classification. Key metrics such as accuracy, precision, and recall are used to evaluate the model’s success in classifying positive and negative reviews.
- **Prediction Output**: After training, the model generates prediction probabilities for each test sample. These results are stored in a structured format, making it easy to analyse the model’s performance across the test dataset.

## Project Reflections
This project demonstrates BERT’s adaptability and effectiveness for sentiment analysis. Despite running on a CPU, the model’s performance showcases its strength in capturing nuanced sentiment patterns in text. Each processing step—from tokenisation to evaluation—adds to a robust framework for text classification, setting a high standard for future NLP projects.

In summary, this project not only highlights BERT’s capabilities in sentiment analysis but also provides a blueprint for implementing transformer-based models on large-scale datasets. By meticulously adapting the original BERT setup to our dataset, we showcase a structured approach to deploying state-of-the-art NLP models for practical applications.

## License
This project builds upon the open-source code from Google’s BERT repository, which is licensed under the Apache License 2.0. This license supports free usage and modification, aligning with the principles of open science and reproducible research.
