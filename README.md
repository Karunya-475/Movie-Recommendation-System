# Movie Recommendation System

An NLP-based Movie Recommendation System that generates personalized movie suggestions from a user's recently watched movies. The project compares two content-based recommendation approaches:

-  Word2Vec-based movie embeddings
-  Sentence Transformer-based semantic embeddings

The notebook demonstrates that the Sentence Transformer approach provides substantially better recommendation performance than the Word2Vec approach and the previous recommendation approach used in the project scenario.

### Project Overview

The project is designed around a fictional streaming platform called Streamora, where the existing movie recommendation approach is not sufficiently matching user preferences.

The objective is to analyze historical viewing behavior and build an AI-powered recommendation engine that can provide more relevant and personalized movie suggestions.

The system uses movie metadata such as:

-   Movie title
-   Genres
-   Movie overview/plot description
-   A user's recently watched movies

The recommendation engine converts movie text into numerical embeddings, represents the user's recent viewing history using an average embedding, and then uses cosine similarity to identify the most similar unwatched movies.

### Problem Statement

The existing recommendation approach used by Streamora is not resonating effectively with users. This creates a need to evaluate alternative recommendation strategies that better represent movie content and user preferences.

The project analyzes historical viewer behavior over an 11-month period and develops an AI-powered recommendation engine intended to improve personalization, engagement, loyalty, and content consumption.

### Objectives

The major objectives of this project are:

-   Build an NLP-based movie recommendation system.
-   Use movie textual information to understand movie similarity.
-   Develop a Word2Vec-based recommendation baseline.
-   Develop a Sentence Transformer-based recommendation model.
-   Generate recommendations from a user's recently watched movies.
-   Compare the recommendation success of both approaches.
-   Evaluate recommendations against actual watched movies.
-   Demonstrate the final model through an interactive Gradio interface.
-   Test the final model on a set of unseen movie inputs.

The notebook uses two datasets.

#### 1. Movie Dataset

The movie dataset contains 4,803 movie records and 3 main columns:

  Column       Description
  
  `title`      Name of the movie
  `genres`     Genres associated with the movie
  `overview`   Short description of the movie plot/story

During preprocessing:

-   `overview` contains 3 missing records.
-   `genres` contains 28 missing records.
-   Rows containing missing values are removed.
-   The index is reset after cleaning.

#### 2. Evaluation Dataset

The evaluation dataset contains 2,000 rows and 10 columns.

  
  Column                              Description
   
  `movie_1` -- `movie_7`              Seven recently watched movies by a
                                      user

  `movie_watch`                       Movie actually watched after the
                                      recommendation

  `past_success`                      Indicates whether the previous
                                      recommendation approach was
                                      successful

  `date`                              Date associated with the
                                      recommendation/evaluation

The evaluation dataset contains no missing values according to the
notebook.

### Overall Workflow

``` text
Movie Dataset
      │
      ▼
Data Loading
      │
      ▼
Data Cleaning
      │
      ├───────────────────────────┐
      │                           │
      ▼                           ▼
Word2Vec Approach        Sentence Transformer
      │                           │
      ▼                           ▼
Text Preprocessing        Semantic Embeddings
      │                           │
      ▼                           ▼
Word Embeddings           Movie Embeddings
      │                           │
      ▼                           ▼
Average User Vector       Average User Vector
      │                           │
      └─────────────┬─────────────┘
                    ▼
             Cosine Similarity
                    │
                    ▼
          Top 10 Movie Recommendations
                    │
                    ▼
              Model Evaluation
                    │
                    ▼
              Model Comparison
                    │
                    ▼
       Sentence Transformer Selected
                    │
                    ▼
             Gradio Demonstration
```



### Technologies Used

  Technology              Purpose

  Python                  Core programming language
  Pandas                  Data loading and manipulation
  NumPy                   Numerical and vector operations
  Gensim                  Word2Vec model
  Sentence Transformers   Semantic text embeddings
  Scikit-learn            Cosine similarity and evaluation
  Matplotlib              Visualization
  Seaborn                 Confusion matrix visualization
  Gradio                  Interactive recommendation interface
  Google Colab            Notebook execution environment


### Project Structure

A recommended GitHub repository structure is:

``` text
Movie-Recommendation-System/
│
├── Movie Recommendation System.ipynb
├── movie_dataset_title.csv
├── Evaluation_Data.csv
├── README.md
└── requirements.txt
```

### Data Preprocessing

The project combines movie information before generating embeddings.

The relevant fields are:

``` text
title + genres + overview
```

For the Word2Vec approach, the notebook:

1.  Creates a copy of the movie DataFrame.
2.  Combines title, genres, and overview.
3.  Replaces the `|` separator in genres with spaces.
4.  Tokenizes the combined text using Gensim's `simple_preprocess`.
5.  Creates token lists for each movie.
6.  Uses these tokens to train Word2Vec.

Example conceptual representation:

``` text
Title: The Avengers
Genres: Action|Adventure|Science Fiction
Overview: Earth's mightiest heroes...
                 │
                 ▼
      Combined Movie Text
                 │
                 ▼
             Tokens
                 │
                 ▼
         Word Embeddings
```

### Recommendation Approach 1: Word2Vec

#### Word2Vec Pipeline

``` text
Movie Text
   ↓
Tokenization
   ↓
Word2Vec Training
   ↓
Word Vectors
   ↓
Average Word Vector
   ↓
Movie Embedding
```

### Recommendation Approach 2: Sentence Transformers

#### Sentence Transformer Pipeline

``` text
Title + Genres + Overview
          ↓
     Combined Text
          ↓
Sentence Transformer
          ↓
   Movie Embedding
          ↓
Average Watched-Movie Embedding
          ↓
    Cosine Similarity
          ↓
     Top 10 Movies
```

#### Embedding Generation

The notebook combines:

``` python
combined_text = f"{title} {genres} {overview}"
```

and generates an embedding using:

``` python
model.encode(combined_text)
```

The resulting embedding is stored in:

``` text
data_sentf['embedded_vector']
```

### Cosine Similarity

Cosine similarity is used to measure how similar two movie embeddings
are.

Conceptually:

``` text
Similarity
    ↑
    │
Movie A ─────────────── Movie B
    │
    └── Similarity Score
```

A higher cosine similarity means the two vectors are more directionally similar.
In this project, the system compares:

``` text
Average embedding of recently watched movies
                         VS
              Embeddings of all movies
```

The movies with the highest similarity are selected as recommendations.


### User Preference Representation

Instead of using only one watched movie, the project can use multiple recently watched movies.

For example:

``` text
User History
│
├── Movie 1
├── Movie 2
├── Movie 3
├── Movie 4
├── Movie 5
├── Movie 6
└── Movie 7
        │
        ▼
Average of movie embeddings
        │
        ▼
User preference representation
```

This average embedding represents the user's general recent viewing preference.
The recommendation function then excludes movies that are already present in the user's watched list.

### Evaluation Strategy

The project evaluates whether the movie that the user actually watched was included in the model's top 10 recommendations.

For every evaluation record:

``` text
Recommended movie list
          +
Actual watched movie
          │
          ▼
Is actual movie present?
          │
     ┌────┴────┐
     │         │
    Yes        No
     │         │
   Match      No Match
```

A match is represented by:

``` python
movie_watch in recommendation_list
```

The overall success percentage is calculated as:

``` text
Successful Recommendations
-------------------------- × 100
Total Evaluations
```

### Model Performance

The notebook reports the following results.

  Model                             Reported Recommendation Success
   
  Previous recommendation model                             \~13.5%
  Word2Vec                                                **22.7%**
  Sentence Transformer                                   **64.75%**

#### Word2Vec

The notebook reports an overall success rate of:

**22.7%**

The monthly success rate is approximately **17%--27%**.

-   Highest: February 2024, approximately 27%
-   Lowest: September and November 2024, approximately 16%--17%

#### Sentence Transformer

The notebook reports an overall success rate of:

**64.75%**

The monthly success rate is approximately **62%--68%**.

-   Highest: March and August 2024, approximately 67%--68%
-   Lowest: October 2024, approximately 63%
-   Most months remain above 64%


### Model Comparison

The Sentence Transformer approach substantially outperforms the Word2Vec approach in the notebook's evaluation.

#### Why Sentence Transformers perform better

Word2Vec provides word-level embeddings. These embeddings are useful for representing individual words but have limitations when understanding the meaning of complete movie descriptions. Sentence Transformers create semantic representations of complete text.
This helps the system capture relationships involving:

-   Movie context
-   Genre
-   Storyline
-   Semantic similarity
-   Overall movie meaning

Therefore, the Sentence Transformer approach provides a stronger representation for this content-based recommendation task.

### Future Enhancements

The notebook suggests several directions for improving the system. These can include:

-   Incorporating additional user behavior signals.
-   Using richer movie metadata.
-   Fine-tuning Transformer architectures.
-   Incorporating ratings and watch duration.
-   Considering explicit likes/dislikes.
-   Combining content-based and collaborative filtering.
-   Using user-level embeddings.
-   Introducing hybrid recommendation techniques.
-   Adding diversity and novelty to recommendations.
-   Deploying the recommendation engine as a production API.
-   Building a scalable vector database for large movie catalogs.
-   Monitoring recommendation quality continuously after deployment.

These enhancements would help move the demonstrated content-based system toward a more complete production recommendation platform.

### Conclusion

This project presents an end-to-end NLP-based Movie Recommendation System using Word2Vec and Sentence Transformers.

The system starts with movie metadata, preprocesses the textual information, converts movie content into embeddings, represents recent user viewing behavior through an average embedding, and ranks candidate movies using cosine similarity. The evaluation in the notebook shows that the Sentence Transformer model achieves a reported 64.75% recommendation success rate, compared with 22.7% for Word2Vec and approximately 13.5% for the previous recommendation approach.

The results highlight the value of contextual semantic representations for personalized content recommendation. The project also demonstrates a practical path from experimentation to an interactive application using Gradio, making it a useful foundation for further development into a real-world recommendation platform.
in the accompanying **Movie Recommendation
System** notebook.
