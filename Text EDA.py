#!/usr/bin/env python
# coding: utf-8

# **Context**<br>
# Welcome. This is a Women’s Clothing E-Commerce dataset revolving around the reviews written by customers. Its nine supportive features offer a great environment to parse out the text through its multiple dimensions. Because this is real commercial data, it has been anonymized, and references to the company in the review text and body have been replaced with “retailer”.
# 
# **Content**<br>
# 
# Clothing ID: Integer Categorical variable that refers to the specific piece being reviewed.
# 
# Age: Positive Integer variable of the reviewers age.
# 
# Title: String variable for the title of the review.
# 
# Review Text: String variable for the review body.
# 
# Rating: Positive Ordinal Integer variable for the product score granted by the customer from 1 Worst, to 5 Best.
# 
# Recommended IND: Binary variable stating where the customer recommends the product where 1 is recommended, 0 is not 
# recommended.
# 
# Positive Feedback Count: Positive Integer documenting the number of other customers who found this review positive.
# 
# Division Name: Categorical name of the product high level division.
# 
# Department Name: Categorical name of the product department name.
# 
# Class Name: Categorical name of the product class name.

# In[1]:


import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('Womens Clothing E-Commerce Reviews.csv')


# In[3]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)


# In[4]:


df1 = df.copy()


# In[5]:


df1.head()


# In[6]:


df1.info()


# In[7]:


df1.columns


# In[8]:


df1.drop('Unnamed: 0', axis = 1, inplace = True)


# In[9]:


df1['Title'].value_counts()


# In[10]:


df1['Title'].mode()


# In[11]:


df1['Title'].isnull().sum()


# In[12]:


df1['Title']


# In[13]:


df1.columns = map(lambda x : x.lower(), df1.columns)


# In[14]:


df1.columns = map(lambda x : x.replace(" ", "_"), df1.columns)


# In[15]:


df1.columns


# In[16]:


df1.head(5)


# In[17]:


df1.drop('clothing_id', axis = 1, inplace = True)


# In[18]:


df1.head()


# In[19]:


df1.isnull().sum()


# In[20]:


df1.dropna(subset = ['review_text', 'division_name'], inplace = True)


# In[21]:


df1.isnull().sum()


# In[22]:


df1.drop(labels = ['title'], axis = 1, inplace = True)


# In[23]:


for i in df1.columns:
    print(f"{i} mode is: {df1[i].mode()}")
    print()


# In[24]:


for i in df1.columns:
    if df1[i].dtype == 'int64':
        print(f"{i}, mean value is: {df1[i].mean()}")
        print()
    else:
        print(f"{i}, mode value is: {df1[i].mode()[0]}")
        print()


# In[25]:


df1.head(2)


# In[26]:


df1[df1['division_name'] == 'General']


# In[27]:


df1['division_name'].value_counts()


# In[28]:


df1[df1['division_name'].isna()]


# In[29]:


def expand_contractions(text):
    if isinstance(text, str):
        contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "could've": "could have",
        "couldn't": "could not",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he had",
        "he'll": "he will",
        "he's": "he is",
        "I'd": "I had",
        "I'll": "I will",
        "I'm": "I am",
        "I've": "I have",
        "isn't": "is not",
        "it's": "it is",
        "let's": "let us",
        "might've": "might have",
        "mustn't": "must not",
        "shan't": "shall not",
        "she'd": "she had",
        "she'll": "she will",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "that's": "that is",
        "there's": "there is",
        "they'd": "they had",
        "they'll": "they will",
        "they're": "they are",
        "wasn't": "was not",
        "we'd": "we had",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "where's": "where is",
        "who'd": "who had",
        "who'll": "who will",
        "who're": "who are",
        "who's": "who is",
        "who've": "who have",
        "won't": "will not",
        "would've": "would have",
        "wouldn't": "would not",
        "you'd": "you had",
        "you'll": "you will",
        "you're": "you are",
        "you've": "you have"
    }

        
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)

    return text


# In[30]:


df1['review_text'].shape


# In[31]:


df1['cleaned_text'] = df1['review_text'].apply(expand_contractions)


# In[32]:


import re

def remove_special_characters(text):
    pattern = r'[^a-zA-Z0-9\s]'
    
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text


# In[33]:


df1['cleaned_text'].apply(remove_special_characters)


# In[34]:


df1.head()


# In[35]:


from textblob import TextBlob

# Sample text for sentiment analysis
text = "I love this product. It's amazing!"

# Create a TextBlob object
blob = TextBlob(text)

# Get the sentiment polarity (a value between -1 and 1, where negative values indicate negative sentiment)
sentiment_polarity = blob.sentiment.polarity

# Interpret the sentiment
if sentiment_polarity > 0:
    sentiment = "Positive"
elif sentiment_polarity < 0:
    sentiment = "Negative"
else:
    sentiment = "Neutral"

# Print the sentiment and polarity
print("Sentiment: ", sentiment)
print("Sentiment Polarity: ", sentiment_polarity)


# In[36]:


df1['cleaned_text'] = df1['cleaned_text'].astype(str)


# In[37]:


df1['sentiment_score'] = df1['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)


# In[38]:


plt.plot(df1['sentiment_score'])
plt.axhline(y = 0, color = 'black', label = 'above it (positive_sentiment_score), below it (negative_sentiment_score)')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()


# It seems that my major reviews are positive

# In[39]:


df1['sentiment_score'].mode()


# In[40]:


df1['sentiment_score'].mean()


# In[41]:


df1['sentiment_score'].max()


# In[42]:


df1['sentiment_score'].min()


# In[43]:


sns.histplot(df1['sentiment_score'], bins = 50, kde = True)
plt.tight_layout()
plt.show()


# In[44]:


import pandas as pd
import scipy.stats as stats

# Assuming 'sentiment_score' is a column in your DataFrame df1
sentiment_data = df1['sentiment_score']

# Calculate kurtosis and skewness
kurtosis = sentiment_data.kurtosis()
skewness = sentiment_data.skew()

print("Kurtosis:", kurtosis)
print("Skewness:", skewness)


# Distribution Shape:<br>
# 
# The histogram visually appears to have a bell-shaped curve, resembling a normal distribution. 
# 
# Descriptive Statistics:<br>
# 
# Minimum (Min): -0.975<br>
# Maximum (Max): 1.0<br>
# Mean: 0.24<br>
# Mode: 0.5<br>
# Mean and Mode:<br>
# 
# The mean sentiment score of 0.24 suggests that, on average, the sentiment in the dataset tends to be positive, although it is not extremely high.
# The mode of 0.5 indicates that a sentiment score of 0.5 is the most frequently occurring score, which may suggest that a substantial number of reviews are moderately positive.
# Range:
# 
# The sentiment scores range from -0.975 (indicating a very negative sentiment) to 1 (indicating a very positive sentiment). <br>
# 
# 
# 
# 
# 
# 
# Kurtosis: 1.5870665497442111<br>
# Skewness: 0.2865174843821539
# 
# With a kurtosis of approximately 1.587 and a skewness of approximately 0.287, we can interpret the characteristics of the distribution of your 'sentiment_score' data as follows:
# 
# Kurtosis (1.587):
# 
# The positive kurtosis indicates that the distribution has slightly heavier tails compared to a normal distribution (which has a kurtosis of 0).
# A positive kurtosis suggests that there may be more extreme values or outliers in the data, resulting in a distribution with slightly fatter tails than a normal distribution.
# Skewness (0.287):
# 
# The positive skewness (0.287) indicates that the distribution is slightly right-skewed.
# A right-skewed distribution means that the tail on the right side of the distribution is longer or more stretched out, while the left side is shorter.
# 
# 
# This suggests that there may be a concentration of sentiment scores on the positive side (more positive sentiments) with a smaller number of negative sentiments on the left side.
# <br>
# 
# In summary, the 'sentiment_score' distribution is slightly right-skewed, meaning it tends to have more positive sentiments but also has some variability with slightly heavier tails, indicating the presence of potential outliers or extreme values. This information provides additional insights into the nature of the sentiment distribution in your dataset.
# 
# 
# 
# 
# 

# In[45]:


df1['word_count'] = df1['cleaned_text'].apply(lambda x : len(x.split()))


# In[46]:


df1['word_count']


# In[47]:


df1['review_length'] = df1['cleaned_text'].apply(lambda x : len(x))


# In[48]:


def average_word(x):
    if isinstance(x, str):
        words = x.split()
        word_len = 0
        for word in words:
            word_len += len(word)

        if len(words) > 0:
            return word_len / len(words)
    else:
        return 0  # Handle non-string input (e.g., integers)


# In[49]:


df1['avg_review_len'] = df1['cleaned_text'].apply(lambda x : average_word(x))


# In[50]:


def sentimen(x):
    if x > 0:
        sentiment = "Positive"
        return sentiment
    elif x < 0:
        sentiment = "Negative"
        return sentiment
    else:
        sentiment = "Neutral"
        return sentiment
        
    


# In[51]:


df1['rating'].value_counts()


# In[52]:


custom_bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
bins = [1,2,3,4,5,6]
sns.histplot(df1['rating'], bins = custom_bins)
plt.title('Review_rating_distribution')
plt.tight_layout()


# The majority of reviews have a rating of 5, indicating that a significant portion of customers provided very positive feedback for the products.<br>
# Ratings 4 and 3 also have a substantial number of reviews, suggesting that many customers had generally positive or neutral experiences with the products.<br>
# Ratings 2 and 1 have fewer reviews, indicating a smaller proportion of customers had less positive or negative experiences.
# 
# The distribution suggests that a large portion of customers had positive or at least neutral sentiments about the products.<br>
# The presence of lower ratings (2 and 1) indicates that there is also a minority of customers who had negative experiences or were dissatisfied with the products.<br>

# In[53]:


df1['age'].value_counts()


# In[54]:


print("mean",df1['age'].mean())
print("mode",df1['age'].mode())
print("max",df1['age'].max())
print("min", df1['age'].min())


# In[55]:


age_data = df1['age']

# Calculate kurtosis and skewness
kurtosis = age_data.kurtosis()
skewness = age_data.skew()

print("Kurtosis:", kurtosis)
print("Skewness:", skewness)


# In[56]:


sns.histplot(df1['age'], bins = 40, kde = True)
plt.title('Age_rating_distribution_of_reviewers')
plt.tight_layout()


# Age Distribution<br>
# 
# The dataset contains a wide range of ages, from a minimum of 18 to a maximum of 99.<br>
# The most common age is 39, with 1,225 occurrences (the mode), followed by 35, 36, 34, and 38.<br>
# 
# Central Tendency:<br>
# 
# The mean age is approximately 43.28, which provides a measure of central tendency. It suggests that the average age of customers in your dataset is around 43 years.<br>
# 
# Kurtosis (-0.1304):<br>
# 
# The negative kurtosis indicates that the distribution has lighter tails compared to a normal distribution (kurtosis of 0). This means that the distribution of ages is less "tailed" and less prone to extreme values.
# 
# Skewness (0.5156):<br>
# 
# The positive skewness (0.5156) indicates that the age distribution is slightly right-skewed.<br>
# A right-skewed distribution suggests that there is a concentration of ages toward the younger end, with a tail extending towards older ages.<br>
# 
# Age Diversity:<br>
# 
# The dataset includes a diverse range of ages, covering a broad spectrum of customer demographics.<br>
# The presence of customers in various age groups indicates that the retailer's products are likely appealing to a wide audience.<br>
# 
# Marketing Insights:<br>
# 
# Understanding the age distribution can help retailers tailor their marketing strategies and product offerings to cater to different age groups.<br>
# Analyzing age-related patterns in product preferences and reviews can offer valuable insights.<br>
# 
# Potential Target Groups:<br>
# 
# Retailers can identify specific age groups that are most prevalent and tailor their marketing or product development strategies to target these groups.<br>
# 

# In[57]:


sns.histplot(df1['review_length'])
plt.tight_layout()


# In[58]:


df1['word_count'].min()


# In[59]:


np.round(df1['word_count'].median(),0)


# In[60]:


df1['word_count'].value_counts().reset_index()


# In[61]:


df1['word_count'].value_counts()


# In[62]:


sns.histplot(df1['word_count'])
plt.tight_layout()


# **Frequency of Word Counts:** The 'word_count' values range from 2 to 117. The majority of word counts fall in the range of 94 to 105, with the highest frequency occurring at 98 (424 occurrences) and 100 (419 occurrences).<br>
# 
# **Central Tendency:** The distribution seems to be somewhat symmetric with the peak at 98 and 100. This suggests that a significant portion of the data falls within this range.<br>
# 
# **Spread:** The word counts vary from 2 to 117, indicating a broad range of word counts within the dataset.<br>
# 
# **Skewness:** The distribution is slightly right-skewed, as there are more occurrences of word counts on the higher end, but the skew is relatively mild.<br>
# 
# **Outliers:** There are a few outliers on both ends of the distribution, such as extremely low word counts (2 to 7) and very high word counts (110 to 117).<br>
# 
# **Tails:** The distribution has long tails on both ends, which means there are relatively few observations with very low and very high word counts.<br>
# 
# 
# 
# 
# 

# The minimum word count in the dataset is 2 words.<br>
# The maximum word count in the dataset is 117 words.<br>
# The average word count across all the text entries is approximately 61 words.<br>
# The median word count (the middle value when sorted) is approximately 60 words.<br>
# The max word_count range are from 98, 100, 99	(ascending order)

# Word Count Distribution:<br>
# 
# The dataset contains a wide range of word counts in the reviews, spanning from a minimum of 2 words to a maximum of 117 words.
# 
# Frequent Word Counts:<br>
# 
# The most common word counts are in the range of 94 to 105 words, with a significant number of reviews falling into these categories. For example, there are 424 reviews with 98 words, 419 with 100 words, and 411 with 99 words.
# 
# Word Count Diversity:<br>
# 
# The dataset includes a diverse set of review lengths, suggesting that customers provide varying levels of detail and commentary in their reviews.
# 
# Marketing and Product Insights:
# 
# Analyzing the word count distribution can help retailers understand the level of detail and elaboration that customers provide in their reviews. This information can guide product descriptions and marketing efforts.
# Text Analytics:
# 
# Retailers can use natural language processing and text analytics to extract insights from reviews of different lengths. For instance, shorter reviews might be analyzed for sentiment, while longer reviews could provide more detailed feedback about specific products.
# 
# Customer Engagement:
# 
# Longer reviews may indicate a higher level of customer engagement or a more in-depth review process, which can be valuable for retailers in understanding customer preferences and areas for improvement.
# 
# Customer Behavior:
# 
# The word count distribution can offer insights into how customers choose to express their opinions and experiences. Longer reviews might include more specific feedback, while shorter reviews could represent quick ratings or brief comments.

# In[63]:


df1['rating'].corr(df1['word_count'])


# In[64]:


df1['rating'].corr(df1['avg_review_len'])


# In[65]:


df1['avg_review_len'].value_counts().nlargest(100)


# In[66]:


avg_age_data = df1['avg_review_len']

# Calculate kurtosis and skewness
kurtosis = avg_age_data.kurtosis()
skewness = avg_age_data.skew()

print("Kurtosis:", kurtosis)
print("Skewness:", skewness)
print("max",df1['avg_review_len'].max())
print("min",df1['avg_review_len'].min())
print("mean",df1['avg_review_len'].mean())
print("mode",df1['avg_review_len'].mode())


# In[67]:


sns.histplot(df1['avg_review_len'])
plt.tight_layout()


# In[68]:


sns.boxplot(df1['word_count'])


# In[69]:


dep_sent = df1.groupby('department_name')['sentiment_score'].mean().reset_index().sort_values(by = 'sentiment_score')


# In[70]:


dep_sent


# In[71]:


plt.plot(dep_sent['department_name'], dep_sent['sentiment_score'])
plt.tight_layout()
plt.xlabel('department_name')
plt.ylabel('sentiment_score')
plt.title('Department wise sentiment score')
plt.grid(True)
plt.show()


# Though overall my sentiment score is above zero which is positive<br> 
# 
# The "Intimate" department stands out as having the highest sentiment score, indicating that reviews associated with this department tend to be more positive or favorable.<br>
# "Tops" and "Bottoms" have very similar sentiment scores, suggesting that customer sentiment is balanced between these two departments.<br>
# "Dresses" also has a relatively high sentiment score, indicating that customers generally have positive sentiments toward dress-related products.<br>
# "Jackets" and "Trend" have slightly lower sentiment scores, suggesting that reviews for these departments may have a slightly less positive tone on average.<br>

# In[72]:


df1['department_name'].value_counts().plot(kind = 'line', color = 'black', marker = "o")
plt.xlabel('department_name')
plt.ylabel('count')
plt.grid(True)
plt.gca().set_facecolor('lightgrey')
plt.tight_layout()
plt.show()


# In[73]:


df1['department_name'].value_counts().plot(kind = 'bar')
plt.xlabel('department_name')
plt.ylabel('count')
plt.grid(True)
plt.gca().set_facecolor('white')
plt.tight_layout()
plt.show()


# In[74]:


df1['division_name'].value_counts()


# In[75]:


df1['division_name'].value_counts().plot(kind = 'bar')
plt.xlabel('division_name')
plt.ylabel('count')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[76]:


df1['class_name'].value_counts()


# In[77]:


df1['class_name'].value_counts().plot(kind = 'bar')
plt.xlabel('class_name')
plt.ylabel('count')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[78]:


from sklearn.feature_extraction.text import CountVectorizer


# countvectorizer count unqiue words</bn>
# it takes input of list
# 

# In[79]:


def get_top_n_words(x, n):
    vec = CountVectorizer().fit(x)
    bow = vec.transform(x)
    sum_words = bow.sum(axis = 0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x : x[1], reverse = True)
    return words_freq[:n]


# In[80]:


get_top_n_words(df1['cleaned_text'], 20)


# In[81]:


import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download NLTK stopwords if you haven't already (you only need to do this once)
nltk.download('stopwords')

def remove_stopwords(text):
    words = text.split()
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)



# In[82]:


df1['cleaned'] = df1['cleaned_text'].apply(remove_stopwords)


# In[83]:


import string

def remove_special_characters(text):
    # Define a set of allowed characters (letters, digits, and spaces)
    allowed_characters = set(string.ascii_letters + string.digits + ' ')

    # Filter the text to keep only the allowed characters
    cleaned_text = ''.join(char for char in text if char in allowed_characters)

    return cleaned_text

# Example usage:
text_with_special_characters = "This is an example text with @special characters and punctuation!"
cleaned_text = remove_special_characters(text_with_special_characters)
print("Original Text:")
print(text_with_special_characters)
print("\nCleaned Text:")
print(cleaned_text)


# In[84]:


df1['cleaned'] = df1['cleaned'].apply(remove_special_characters)


# In[85]:


df1['cleaned'] = df1['cleaned'].apply(lambda x : x.lower())


# In[86]:


top_20_words = pd.DataFrame(get_top_n_words(df1['cleaned'], 20), columns = ['words', 'freq'])


# In[87]:


plt.plot(top_20_words['words'], top_20_words['freq'], marker = 'o')
plt.grid(True)
plt.xlabel('words')
plt.ylabel('count')
plt.title('Top 20 unigram words')
plt.xticks(rotation = 75)
plt.tight_layout()
plt.show()


# Top 20 Words and Their Frequencies:<br>
# 
# "dress" is the most common word in the cleaned text, appearing 10,460 times.<br>
# "love" and "size" are the second and third most common words, with 8,910 and 8,685 occurrences, respectively.<br>
# "top," "fit," "like," "wear," "great," "im" (short for "I'm"), "would," and "it" are also highly frequent words in the reviews.
# 
# Analysis:<br>
# 
# The most common word, "dress," suggests that many reviews in your dataset are related to dresses, indicating that this clothing item is frequently reviewed or discussed.<br>
# "love" is a positive sentiment word and often indicates that customers have a strong positive sentiment towards the products.<br>
# "size" is crucial for clothing reviews, as it pertains to the fit and sizing of the products.
# <br>
# Words like "top," "fit," "like," "wear," and "great" are expected in clothing reviews, as they reflect aspects of style, comfort, and satisfaction.<br>

# In[88]:


def remove_stopwords(text, custom_stopwords=[]):
    words = text.split()
    stop_words = set(stopwords.words('english'))
    stop_words.update(custom_stopwords)  # Add custom stopwords to the set
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


# In[89]:


df1['cleaned']


# In[90]:


def get_top_n_words_bigram(x, n):
    vec = CountVectorizer(ngram_range=(2,2)).fit(x)
    bow = vec.transform(x)
    sum_words = bow.sum(axis = 0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x : x[1], reverse = True)
    return words_freq[:n]


# In[91]:


top_20_words_bigram = pd.DataFrame(get_top_n_words_bigram(df1['cleaned'], 20), columns = ['words', 'freq'])


# In[92]:


top_20_words_bigram


# In[93]:


plt.plot(top_20_words_bigram['words'], top_20_words_bigram['freq'])
plt.grid(True)
plt.xticks(rotation = 75)
plt.title('bigram_line_graph')
plt.tight_layout()
plt.show()


# Top 20 Bigrams and Their Frequencies:<br>
# 
# "true size" is the most common bigram, appearing 1,298 times.<br>
# "love it" is the second most common bigram, with 845 occurrences, indicating that customers frequently express their love for a product in reviews.<br>
# "love dress," "usually wear," "looks great," and "fit perfectly" are also among the top bigrams with high frequencies.<br>
# "well made," "size small," "fits perfectly," "fit well," and "highly recommend" are meaningful bigrams that convey information about product quality, size, and recommendations.<br>
# 
# Analysis:<br>
# 
# Bigrams provide more context and can capture specific phrases or expressions used by customers in their reviews.<br>
# "True size" suggests that customers frequently mention the accuracy of sizing in their reviews, indicating that sizing is an important consideration for clothing products.<br>
# "Love it" is a common expression indicating strong positive sentiment and satisfaction with the product.<br>
# "Well made" is a positive assessment of product quality, and "highly recommend" is a strong endorsement for the product.<br>
# "Fits perfectly" and "fit well" highlight the importance of the fit of the clothing items.<br>
# "Looks great" and "super cute" reflect customers' opinions on the style and appearance of the products.<br>
# 
# Implications:<br>
# 
# Analyzing the most frequent bigrams can help retailers identify recurring themes in customer reviews, focusing on factors like sizing, quality, style, and overall satisfaction.<br>
# Retailers can use this information to improve product descriptions, address common customer concerns, and tailor their product offerings and marketing strategies to align with customers' preferences and priorities.<br>

# In[94]:


plt.bar(top_20_words_bigram['words'], top_20_words_bigram['freq'])
plt.xticks(rotation = 75)
plt.tight_layout()
plt.show()


# In[95]:


def get_top_n_words_trigram(x, n):
    vec = CountVectorizer(ngram_range=(3,3)).fit(x)
    bow = vec.transform(x)
    sum_words = bow.sum(axis = 0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x : x[1], reverse = True)
    return words_freq[:n]


# In[96]:


top_20_words_trigram_2 = pd.DataFrame(get_top_n_words_trigram(df1['cleaned'], 20), columns = ['words', 'freq'])


# In[97]:


top_20_words_trigram_2


# In[98]:


plt.plot(top_20_words_trigram_2['words'], top_20_words_trigram_2['freq'])
plt.grid(True)
plt.xticks(rotation = 75)
plt.title('trigram_line_graph')
plt.tight_layout()
plt.show()


# Top 20 Trigrams and Their Frequencies:<br>
# 
# "Fits true size" is the most common trigram, appearing 269 times. This trigram is often used to express that the product's sizing is accurate and it fits well.<br>
# "Cannot wait wear" is the second most common trigram, suggesting anticipation and excitement to wear the product.<br>
# "Received many compliments" indicates that customers frequently receive positive feedback from others when wearing the product.<br>
# "Runs true size" is another trigram related to sizing and fit.<br>
# "Love love love" is a repeated expression of strong affection and satisfaction with the product.<br>
# 
# Analysis:<br>
# 
# Trigrams provide even more context and capture specific phrases or expressions that customers use in their reviews.<br>
# "Fits true size" is a critical trigram because it reflects customers' satisfaction with the product's sizing accuracy.<br>
# "Cannot wait wear" indicates a positive and excited sentiment, which can be valuable for understanding customer enthusiasm.<br>
# "Received many compliments" suggests that customers are not only satisfied with the product but also receive positive feedback from others when wearing it.<br>
# "Love love love" is an emphatic expression of love and satisfaction with the product.<br>

# In[99]:


top_20_words_trigram_1 = pd.DataFrame(get_top_n_words_trigram(df1['cleaned_text'], 20), columns = ['words', 'freq'])


# In[100]:


plt.plot(top_20_words_trigram_1['words'], top_20_words_trigram_1['freq'])
plt.grid(True)
plt.xticks(rotation = 75)
plt.title('trigram_line_graph')
plt.tight_layout()
plt.show()


# In[101]:


department_mean_rating = df1.groupby(['department_name'])['rating'].mean().reset_index().sort_values(by = 'rating', ascending = True)


# In[102]:


plt.plot(department_mean_rating['department_name'], department_mean_rating['rating'], marker = 'o')
plt.tight_layout()
plt.grid(True)
plt.show()


# In[103]:


df1.groupby(['department_name'])['sentiment_score'].mean()


# In[104]:


division_mean_rating = df1.groupby(['division_name'])['rating'].mean().reset_index().sort_values(by = 'rating')


# In[105]:


plt.plot(division_mean_rating['division_name'], division_mean_rating['rating'], marker = 'o')
plt.tight_layout()
plt.grid(True)
plt.show()


# In[106]:


df1['division_name'].value_counts().reset_index()


# In[107]:


sns.pairplot(data = df1)


# In[108]:


sns.catplot(x = df1['division_name'], y = df1['sentiment_score'])


# In[109]:


sns.catplot(x = 'division_name', y = 'sentiment_score', data = df1, kind = 'box')
plt.tight_layout()


# In[110]:


sns.catplot(x = 'department_name', y = 'sentiment_score', data = df1)
plt.tight_layout()


# In[111]:


sns.catplot(x = 'department_name', y = 'sentiment_score', data = df1, kind = 'box')
plt.tight_layout()


# In[112]:


df1.columns


# In[113]:


sns.catplot(x = 'division_name', y='review_length', data = df1, kind = 'box')


# In[114]:


sns.catplot(x = 'department_name', y='review_length', data = df1, kind = 'box')


# In[115]:


recommended = df1[df1['recommended_ind'] == 1]['sentiment_score']
not_recommended = df1[df1['recommended_ind'] == 0]['sentiment_score']


# In[117]:


sns.histplot(recommended, label = 'recommended', alpha = 0.8)
sns.histplot(not_recommended, label = 'not_recommended', alpha = 0.6)
plt.legend()
plt.tight_layout()
plt.show()


# Recommended Products (recommended_ind = 1):<br>
# 
# Kurtosis: 1.282148255305755<br>
# Skewness: 0.564339914612073<br>
# The kurtosis value for recommended products is 1.282, which is less than the kurtosis of a normal distribution (which is 3). This suggests that the distribution of sentiment scores for recommended products is less peaked and has lighter tails than a normal distribution. The positive skewness value (0.564) indicates that the distribution is slightly skewed to the right, meaning that sentiment scores for recommended products tend to be slightly higher than the mean.
# <br>
# 
# Not Recommended Products (recommended_ind = 0):<br>
# 
# Kurtosis: 2.578502135887394<br>
# Skewness: -0.25119914789918174<br>
# The kurtosis value for not recommended products is 2.578, which is higher than the kurtosis of a normal distribution. This suggests that the distribution of sentiment scores for not recommended products is more peaked and has heavier tails than a normal distribution. The negative skewness value (-0.251) indicates a slight leftward skew, meaning that sentiment scores for not recommended products tend to be slightly lower than the mean.<br>

# In[118]:


recommended_rating = df1[df1['recommended_ind'] == 1]['rating']
not_recommended_rating = df1[df1['recommended_ind'] == 0]['rating']


# In[119]:


custom_bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
sns.histplot(recommended_rating, label = 'recommended', alpha = 0.8, bins = custom_bins)
sns.histplot(not_recommended_rating, label = 'not_recommended', alpha = 0.4, bins = custom_bins)
plt.legend()
plt.tight_layout()
plt.show()


# In[120]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the documents



# In[121]:


tfidf_matrix = tfidf_vectorizer.fit_transform(df1['cleaned'])



# In[122]:


tfidf_matrix


# In[123]:


# Get feature names (words)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Create a DataFrame to display the TF-IDF scores for each word
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)


# In[125]:


word_scores = [(word, tfidf_matrix[0, idx]) for idx, word in enumerate(feature_names)]

# Sort the words by TF-IDF score in descending order
word_scores.sort(key=lambda x: x[1], reverse=True)

# Display the top N words by TF-IDF score (e.g., top 10 words)
N = 5
top_words = word_scores[:N]

print("Top", N, "words by TF-IDF score:")
for word, score in top_words:
    print(f"{word}: {score:.4f}")


# #### A high TF-IDF score for a term in a document indicates that the term is important within that specific document but not very common across the corpus.
# #### A low TF-IDF score suggests that the term is either not significant in the document or it's common in many documents.

# Silky (TF-IDF Score: 0.5487):
# 
# - The word "silky" has the highest TF-IDF score, indicating that it is a distinctive and important term in the context of the text data.
# - "Silky" is often associated with the texture and feel of clothing, suggesting that customers frequently use this word to describe the texture of the products.
# - Products described as "silky" are likely to be perceived as having a luxurious and smooth texture, which can be a selling point.
# 
# Sexy (TF-IDF Score: 0.5038):
# 
# - "Sexy" is the second-highest term by TF-IDF score, suggesting that it's a prominent word in the text data.
# - This word likely appears in reviews when customers are describing clothing that makes them feel attractive and confident.
# - Clothing described as "sexy" is often associated with style and confidence-boosting attributes.
# 
# Wonderful (TF-IDF Score: 0.4742):
# 
# - "Wonderful" has a high TF-IDF score, indicating that it is an important and positively charged term in the text data.
# - Customers use "wonderful" to express a high level of satisfaction and delight with the product.
# - Reviews that include "wonderful" likely reflect very positive feedback and customer enthusiasm.
# 
# Absolutely (TF-IDF Score: 0.3843):
# 
# - "Absolutely" is another word with a significant TF-IDF score, often used to emphasize a sentiment or opinion.
# - When customers say "absolutely" in reviews, they are likely expressing strong agreement or affirmation, which can be related to positive experiences with the product.
# 
# Comfortable (TF-IDF Score: 0.2693):
# 
# - "Comfortable" has a lower but still noteworthy TF-IDF score, indicating that it's a relevant term in the text data.
# - Customers frequently use "comfortable" to describe clothing that feels pleasant and easy to wear, reflecting a focus on comfort and fit.

# Implications:<br>
# 
# These top TF-IDF words provide insights into customer sentiments and the attributes they prioritize when reviewing clothing products.<br>
# Retailers can use this information to understand and emphasize the features and characteristics that customers value the most.<br>
# Words like "silky" and "sexy" can be used to market products with a focus on texture and style, while emphasizing "comfortable" can be important for products designed for comfort.<br>
# Additionally, terms like "wonderful" and "absolutely" suggest the importance of providing products that lead to high customer satisfaction.<br>
