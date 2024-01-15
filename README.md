# Book Recommendation System

[img](/brs01.png)
[img](/brs02.png)
[img](/brs03.png)
## Overview
This repository contains a book recommendation system implemented in Python. The system utilizes Natural Language Processing (NLP) techniques and machine learning to recommend books based on user input.

## Architecture

### 1. Data Preprocessing
- The program starts by loading book data from a CSV file (`books_data.csv`) using the Pandas library.
- Data cleaning involves handling missing values, dropping unnecessary columns, and extracting relevant information from columns like 'categories', 'authors', and 'publishedDate'.
- Duplicate entries and books with a low 'ratingsCount' are filtered out.

### 2. Exploratory Data Analysis
- Various visualizations, including boxplots and pie charts, are used to analyze top categories, authors, and publishers.
- The publication year of books is extracted and plotted against the ratings count.

### 3. Text Processing
- Book content is created by combining 'Title', 'authors', 'publisher', and 'categories'.
- Custom stopwords are defined, and text data is preprocessed using tokenization, lemmatization, and removal of stopwords.
- Bag-of-Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF) representations are generated.

### 4. Recommendation System
- Two recommendation functions (`recommend` and `recommendBIG`) use cosine similarity to suggest books based on user input.
- A user can receive recommendations for a specific book or a more general recommendation if the input does not match any book in the dataset.

### 5. Model Persistence
- The generated BoW, TF-IDF matrices, and the filtered dataset are saved using Pickle for later use.

## Usage
- The user can interact with the recommendation system by calling the `recommend` or `recommendBIG` functions with a specific book or user input.

## Files
- `books_data.csv`: Original dataset.
- `books_bow.pkl`: Pickled BoW matrix.
- `books_small.pkl`: Pickled filtered dataset.
- `books_tfidf.pkl`: Pickled TF-IDF matrix.
- `recommendation_system.py`: Python script containing the recommendation system implementation.

## Dependencies
- Pandas
- Numpy
- Seaborn
- Matplotlib
- NLTK
- Scikit-learn

## Acknowledgements
- The book data used in this project is sourced from [insert source link].

## License
This project is licensed under the [License Name] - see the [LICENSE.md](LICENSE.md) file for details.
