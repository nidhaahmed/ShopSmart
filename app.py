
"""
# **Tech Assignment: To Develop an ML Model for Product Recommendation**
"""
"""
# **1-Data collection**
"""
import pandas as pd

# load CSV file from Google Drive
df = pd.read_csv("C:\\Users\\\hp\\Desktop\\ProductRecommender\\product.csv")

df.head()


"""
# **2-Data preprocessing**
"""


"""
- **Inspect**: Check for missing data, duplicates, and errors.  
- **Clean**: Fix issues and ensure accuracy.  
- **Transform**: Scale, normalize, or engineer features.  
- **Save**: Save cleaned data separately.  
"""


df.shape


# Missing values
def check_missing_values(df):
  return df.isnull().sum()

print(check_missing_values(df))
df[df.rating_count.isnull()]


# Remove rows with missing values in the rating_count column
df.dropna(subset=['rating_count'], inplace=True)
print(check_missing_values(df))


# Duplicates
def check_duplicates(dataframe):
    return dataframe.duplicated().sum()
print(check_duplicates(df))


# Data types
df.dtypes


df['discounted_price'] = df['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
df['actual_price'] = df['actual_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%','').astype(float)/100



# The rating column has a value with an incorrect character, so we will exclude
# the row to obtain a clean dataset.
count = df['rating'].str.contains('\|').sum()
print(f"Total de linhas com '|' na coluna 'rating': {count}")
df = df[df['rating'].apply(lambda x: '|' not in str(x))]
count = df['rating'].str.contains('\|').sum()
print(f"Total de linhas com '|' na coluna 'rating': {count}")


df['rating'] = df['rating'].astype(str).str.replace(',', '').astype(float)
df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').astype(float)



df.dtypes


# Creating the column "rating_weighted"
df['rating_weighted'] = df['rating'] * df['rating_count']


df['sub_category'] = df['category'].astype(str).str.split('|').str[-1]
df['main_category'] = df['category'].astype(str).str.split('|').str[0]


df.columns

len(df)


df.head()


"""
# **3-Exploratory Data Analysis (EDA) & Data visualization**
"""


import matplotlib.pyplot as plt

# Analyzing distribution of products by main category
main_category_counts = df['main_category'].value_counts()[:30] # Select only the top 30 main categories.
plt.bar(range(len(main_category_counts)), main_category_counts.values)
plt.ylabel('Number of Products')
plt.title('Distribution of Products by Main Category (Top 30)')
plt.xticks(range(len(main_category_counts)), '') # hide X-axis labels
plt.show()

# Top 30 main categories
top_main_categories = pd.DataFrame({'Main Category': main_category_counts.index, 'Number of Products': main_category_counts.values})
print('Top 30 main categories:')
print(top_main_categories.to_string(index=False))


"""
Insights:

- The top 3 main categories are Electronics, Computers & Accessories, and Home & Kitchen.

- Office Products, Musical Instruments, Home Improvement, Toys & Games, Car & Motorbike, and Health & Personal Care have a very small number of products i.e., these categories have less demand.
.
"""


# Analyzing distribution of products by last category
sub_category_counts = df['sub_category'].value_counts()[:30] # only the top 30 last categories.
plt.bar(range(len(sub_category_counts)), sub_category_counts.values)
plt.ylabel('Number of Products')
plt.title('Distribution of Products by Sub Category (Top 30)')
plt.xticks(range(len(sub_category_counts)), '')
plt.show()

# Top 30 sub categories
top_sub_categories = pd.DataFrame({'Sub Category': sub_category_counts.index, 'Number of Products': sub_category_counts.values})
print('Top 30 sub categories:')
print(top_sub_categories.to_string(index=False))



"""
Insights:

- The top six subcategories are USB cables, smartwatches, smartphones, smart televisions, in-ear headphones, and remote controls.

- Other popular subcategories include mixer grinders, HDMI cables, dry irons, mice, and instant water heaters. These subcategories may be less popular than the top six, but they still have a significant number of products, indicating that there is demand for them.

"""


# Calculate the top main categories
top = df.groupby(['main_category'])['rating'].mean().sort_values(ascending=False).head(10).reset_index()

# Create a bar plot
plt.bar(top['main_category'], top['rating'])

# Add labels and title
plt.xlabel('main_category')
plt.ylabel('Rating')
plt.title('Top main_category by Rating')

# Rotate x-axis labels
plt.xticks(rotation=90)

# Show the plot
plt.show()
ranking = df.groupby('main_category')['rating'].mean().sort_values(ascending=False).reset_index()
print(ranking)


"""
The insights tell much about the main categories ranked by their average rating.

- The main categories with the highest ratings are Office Products, Toys & Games, and Home Improvement, with ratings above 4.0.

- On the other hand, the main categories with lower ratings are Car & Motorbike, Musical Instruments, and Health & Personal Care, with ratings below 4.0.

- Additionally, we can see that Computers & Accessories and Electronics have ratings above 4.0, which indicates that these categories are popular and well-liked by customers who purchase them.

"""


# Calculate the top sub categories
top = df.groupby(['sub_category'])['rating'].mean().sort_values(ascending=False).head(10).reset_index()

# Create a bar plot
plt.bar(top['sub_category'], top['rating'])

# Add labels and title
plt.xlabel('sub_category')
plt.ylabel('Rating')
plt.title('Top sub_category by Rating')

# Rotate x-axis labels
plt.xticks(rotation=90)

# Show the plot
plt.show()
ranking = df.groupby('sub_category')['rating'].mean().sort_values(ascending=False).reset_index()
print(ranking)


"""
The top and bottom sub-categories in terms of customer ratings.

- The "Tablets" sub-category is at the top with a rating of 4.6, which indicates that customers are satisfied with their purchase.

- However, there are some sub-categories at the bottom, such as "DustCovers" and "ElectricGrinders", which have lower ratings, implying that customers are not very happy with these products.
"""


# sort the means in descending order
mean_discount_by_category = df.groupby('main_category')['discount_percentage'].mean()
mean_discount_by_category = mean_discount_by_category.sort_values(ascending=True)

# create the horizontal bar chart
plt.barh(mean_discount_by_category.index, mean_discount_by_category.values)
plt.title('Discount Percentage by Main Category')
plt.xlabel('Discount Percentage')
plt.ylabel('Main Category')
plt.show()

table = pd.DataFrame({'Main Category': mean_discount_by_category.index, 'Mean Discount Percentage': mean_discount_by_category.values})

print(table)


"""
The mean discount percentage by main category, in descending order.

- The category with the lowest mean discount percentage is Toys&Games, with a value of 0.0. This may indicate that the demand for toys and games is high enough that retailers do not need to offer significant discounts to sell products in this category.

- Home&Kitchen and Car&Motorbike have similar mean discount percentages, with values of 0.401745 and 0.42, respectively. This suggests that there may be a similar level of competition and price sensitivity in these two categories.

- The categories with the highest mean discount percentages are HomeImprovement, Computers&Accessories, and Electronics, with values of 0.575, 0.539202, and 0.508289, respectively. This may indicate that these categories are more price-sensitive, and retailers need to offer attractive discounts to compete effectively.

- It's also interesting to note that OfficeProducts and Health&PersonalCare have mean discount percentages of 0.123548 and 0.53, respectively, which are in between the categories with the lowest and highest mean discount percentages. This suggests that these categories may have some level of price sensitivity, but not to the same extent as HomeImprovement, Computers&Accessories, and Electronics.
"""


# sort the means in descending order
mean_discount_by_sub_category = df.groupby('sub_category')['discount_percentage'].mean().head(15)
mean_discount_by_sub_category = mean_discount_by_sub_category.sort_values(ascending=True)

# create the horizontal bar chart
plt.barh(mean_discount_by_sub_category.index, mean_discount_by_sub_category.values)
plt.title('Discount Percentage by Sub Category')
plt.xlabel('Discount Percentage')
plt.ylabel('Sub Category')
plt.show()

table = pd.DataFrame({'Sub Category': mean_discount_by_sub_category.index, 'Mean Discount Percentage': mean_discount_by_sub_category.values})

print(table)


from wordcloud import WordCloud

# Analyze the reviews by creating word clouds or frequency tables of the most common words used in the reviews.
reviews_text = ' '.join(df['review_content'].dropna().values)
wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(reviews_text)
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


"""
The code generates a word cloud from reviews of products rated above 4, highlighting the most frequently mentioned words. Larger words indicate higher frequency, offering insights into customer sentiment, key product features, or issues to help businesses improve.
"""


# Filter the dataframe to include only products with a rating lower than 2
low_rating_df = df[df['rating'] > 4.0]

# Create a string of all the reviews for these products
reviews_text = ' '.join(low_rating_df['review_content'].dropna().values)

# Generate the wordcloud
wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(reviews_text)

# Plot the wordcloud
plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()


# Perform statistical analysis to identify any correlations between different features, such as the relationship between product price and customer rating.
# Drop non-numeric columns
numeric_cols = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_cols.corr()

# Print the correlation matrix
print(correlation_matrix)



import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


"""
The correlation table shows:

- Weak Positive Correlation: Between overall rating, rating count, and weighted rating, suggesting higher-rated products tend to have more reviews and higher weighted ratings.
- Moderate Positive Correlation (0.121): Between rating and discounted price, indicating discounts may influence higher ratings.
- Note: Correlation ≠ causation. General rules for interpreting r :
- - 0.1 - 0.3: Weak
- - 0.3 - 0.5: Moderate
- - 0.5: Strong
- Other factors like sample size and outliers should be considered for accurate interpretation.
"""


"""
# 4. Recommendation system
"""


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['user_id_encoded'] = le.fit_transform(df['user_id'])

# Create a new dataframe with the user_id frequency table
freq_table = pd.DataFrame({'User ID': df['user_id_encoded'].value_counts().index, 'Frequency': df['user_id_encoded'].value_counts().values})

# Display the dataframe
print(freq_table)
id_example = freq_table.iloc[0,0]
print(id_example)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def recommend_products(df, user_id_encoded):
    # Use TfidfVectorizer to transform the product descriptions into numerical feature vectors
    tfidf = TfidfVectorizer(stop_words='english')
    df['about_product'] = df['about_product'].fillna('')  # fill NaN values with empty string
    tfidf_matrix = tfidf.fit_transform(df['about_product'])

    # Get the purchase history for the user
    user_history = df[df['user_id_encoded'] == user_id_encoded]

    # Use cosine_similarity to calculate the similarity between each pair of product descriptions
    # only for the products that the user has already purchased
    indices = user_history.index.tolist()

    if indices:
        # Create a new similarity matrix with only the rows and columns for the purchased products
        cosine_sim_user = cosine_similarity(tfidf_matrix[indices], tfidf_matrix)

        # Create a pandas Series with product indices as the index and product names as the values
        products = df.iloc[indices]['product_name']
        indices = pd.Series(products.index, index=products)

        # Get the indices and similarity scores of products similar to the ones the user has already purchased
        similarity_scores = list(enumerate(cosine_sim_user[-1]))
        similarity_scores = [(i, score) for (i, score) in similarity_scores if i not in indices]

        # Sort the similarity scores in descending order
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

        # Get the indices of the top 5 most similar products
        top_products = [i[0] for i in similarity_scores[1:6]]

        # Get the names of the top 5 most similar products
        recommended_products = df.iloc[top_products]['product_name'].tolist()

        # Get the reasons for the recommendation
        score = [similarity_scores[i][1] for i in range(5)]

        # Create a DataFrame with the results
        results_df = pd.DataFrame({'Id Encoded': [user_id_encoded] * 5,
                                   'recommended product': recommended_products,
                                   'score recommendation': score})

        return results_df

    else:
        print("No purchase history found.")
        return None


recommend_products(df, 893)



# Streamlit App for Product Recommendations
import streamlit as st

# Title and description
st.title("Amazon Product Recommendation System")
st.write("Provide your User ID to get personalized product recommendations!")

# User input for User ID
user_id = st.text_input("Enter your User ID:")

# Recommendation logic using your existing function
if st.button("Get Recommendations"):
    try:
        # Convert the user ID to encoded format if required
        user_id_encoded = le.transform([user_id])[0]

        # Call the recommendation function from your notebook
        recommendations = recommend_products(df, user_id_encoded)

        if recommendations is not None:
            st.subheader("Recommended Products for You:")
            st.table(recommendations)
        else:
            st.error("No recommendations found. Try another User ID.")
    except Exception as e:
        st.error(f"An error occurred: {e}")