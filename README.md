# Book Insights: Analysis and Trends in Popular Literature

This project dives into a book dataset to explore key insights such as genre popularity, rating distribution, publication trends, and pricing strategies. It features detailed visualizations, statistical analysis, and a basic book recommendation system based on genre and rating similarity.

## 📂 Dataset
The dataset includes information on over 50,000 books (sampled down to ~500 entries with full metadata), covering fields like:
- Title, Author, Description
- Genre, Characters, Format
- Rating, Number of Ratings, Price
- Awards, Publisher, Publish Dates
- Settings, Series Info

## 📊 Features
- **Data Cleaning**: Handling missing values, data type conversions, and parsing list fields.
- **Visualizations**:
  - Top genres and their distributions (bar and pie charts)
  - Rating distributions and trends over time
  - Author popularity and Price vs Rating
  - WordCloud from book descriptions
- **Exploratory Data Analysis**: Genre counts, publishers, Pages vs Rating
- **Statistical Testing**:
  - T-test to compare ratings between awarded and non-awarded books
  - ANOVA to compare ratings across top genres
  - Correlation heatmap and linear regression to predict price
- **Recommendation System**: Suggests books based on shared genres and similar ratings

## 🛠️ Libraries Used
```python
pandas, numpy, seaborn, matplotlib, sklearn, wordcloud, scipy
