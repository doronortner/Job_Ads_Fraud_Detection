import nltk
from nltk import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import re
from wordcloud import WordCloud
import country_converter as coco

# --------------------------------------------------- Q1 - EDA ----------------------------------------------------

stop_words = set(stopwords.words('english'))

# Reading the file
df = pd.read_csv("XY_train.csv")

# ------------------ General Check
df_copy = df.copy()
df_copy = df_copy.drop('job_id', axis = 1)
df_copy = df_copy.drop_duplicates()

# Fraud / real DFs
df_fraud = df[df['fraudulent'] == 1]
df_real = df[df['fraudulent'] == 0]

# ----------------- Job ID
# scatter plot to identify whether the feature job id differs by the length of the job id between fraud and real
temp_df = df.copy()
temp_df['job_id_length'] = temp_df['job_id'].apply(lambda x: len(str(x)))

# calculate the frequency of each value in the "job_id_length" column
freq = temp_df['job_id_length'].value_counts(normalize = True)
df_freq = pd.DataFrame({'job_id_length': freq.index, 'frequency': freq.values})
temp_df = pd.merge(temp_df, df_freq, on = 'job_id_length', how = 'left')

sns.set_theme(style = "whitegrid")
sns.barplot(x = 'job_id_length', y = 'frequency', hue = 'fraudulent', data = temp_df)
plt.xlabel('Fraudulent', fontsize = 10)
plt.ylabel('Job ID Length', fontsize = 10)
plt.title("Job ID Length Frequency Bar Plot", fontsize = 12)
plt.show()

# ---------------- Title
# 10 Most frequent words
titles = df['title']
cleaned_titles = []
for title in titles:
    title = re.sub(r'[^a-zA-Z\s]', '', title)
    title = title.lower()
    cleaned_titles.append(title)

fake_titles = []
real_titles = []

for i in range(len(df)):
    if df['fraudulent'][i] == 1:
        fake_titles.append(cleaned_titles[i])
    else:
        real_titles.append(cleaned_titles[i])

fake_tokens = [nltk.word_tokenize(title) for title in fake_titles]
real_tokens = [nltk.word_tokenize(title) for title in real_titles]
fake_fdist = FreqDist([token for sublist in fake_tokens for token in sublist])
real_fdist = FreqDist([token for sublist in real_tokens for token in sublist])

print("Most common words in fake titles:")
print(fake_fdist.most_common(10))
print("Most common words in real titles:")
print(real_fdist.most_common(10))

# Create a word cloud
fake_wordcloud = WordCloud(width = 800, height = 800, background_color ='white').generate_from_frequencies(fake_fdist)
real_wordcloud = WordCloud(width = 800, height = 800, background_color ='white').generate_from_frequencies(real_fdist)

# Plot the word clouds
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
ax1.imshow(fake_wordcloud)
ax1.axis('off')
ax1.set_title('Fake Ads')
ax2.imshow(real_wordcloud)
ax2.axis('off')
ax2.set_title('Real Ads')
plt.show()

# Check the length of real and fake ads' titles
fake_ads = pd.DataFrame({'Title': fake_titles})
real_ads = pd.DataFrame({'Title': real_titles})

fake_ads['Title_length'] = fake_ads['Title'].apply(len)
real_ads['Title_length'] = real_ads['Title'].apply(len)

fake_ads_mean = fake_ads['Title_length'].mean()
real_ads_mean = real_ads['Title_length'].mean()

print('Average length of fake ads:', fake_ads_mean, 'words')
print('Average length of real ads:', real_ads_mean, 'words')
print('Words difference:', abs(real_ads_mean - fake_ads_mean), 'words')

# ---------------- Location

df_with_countries = df_copy.copy()
# create a new column "country" by extracting the first two characters from the "location" column
df_with_countries['country'] = df_with_countries['location'].str[:2]

# Check null and unique values
print('Amount of null values:' ,df_with_countries['country'].isnull().sum())
print('Percentage of null values:', 100 * (df_with_countries[
'country'].isnull(
).sum() / df.shape[0]), '%')

print('Unique countries:', df_with_countries['country'].nunique())

# get the top 10 countries with the most ads
top_countries = df_with_countries['country'].value_counts().nlargest(10).index.tolist()
country_counts = pd.DataFrame(
    columns=['country', 'real_count', 'fake_count', 'real_to_fake_ratio',
             'total_count'])

# loop through each top country and count the number of real and fake ads
for country in top_countries:
    country_data = df_with_countries.loc[df_with_countries['country'] == country]
    real_count = country_data.loc[country_data['fraudulent'] == 0][
        'fraudulent'].count()
    fake_count = country_data.loc[country_data['fraudulent'] == 1][
        'fraudulent'].count()
    total_count = real_count + fake_count

    if fake_count == 0:
        real_to_fake_ratio = float(
            'inf')  # set the ratio to infinity if there are no fake ads
    else:
        real_to_fake_ratio = real_count / fake_count

    country_counts = pd.concat([country_counts, pd.DataFrame(
        {'country': [country], 'real_count': [real_count],
         'fake_count': [fake_count], 'real_to_fake_ratio': [real_to_fake_ratio],
         'total_count': [total_count]})], ignore_index=True)

print(country_counts)

df_with_countries.dropna(subset=['country'], inplace=True)
df_with_countries['continent'] = coco.convert(names=df_with_countries[
    'country'], to='continent')

# get the top 7 continents with the most ads
top_continents = df_with_countries['continent'].value_counts().nlargest(
    7).index.tolist()

continent_counts = pd.DataFrame(
    columns=['continent', 'real_count', 'fake_count', 'real_to_fake_ratio',
             'total_count'])

for continent in top_continents:
    continent_data = df_with_countries.loc[df_with_countries['continent'] ==
                                         continent]
    real_count = continent_data.loc[continent_data['fraudulent'] == 0][
        'fraudulent'].count()
    fake_count = continent_data.loc[continent_data['fraudulent'] == 1][
        'fraudulent'].count()
    total_count = real_count + fake_count

    if fake_count == 0:
        real_to_fake_ratio = float(
            'inf')  # set the ratio to infinity if there are no fake ads
    else:
        real_to_fake_ratio = real_count / fake_count

    continent_counts = continent_counts._append(pd.DataFrame(
        {'continent': [continent], 'real_count': [real_count],
         'fake_count': [fake_count], 'real_to_fake_ratio': [real_to_fake_ratio],
         'total_count': [total_count]}), ignore_index=True)

print(continent_counts)



# ---------------- Department
print('Percentage of null values:', 100 * (df['department'].isnull().sum() / \
                                     df.shape[0]), '%')
print('Unique departments before we merged categories:', df['department'].nunique())

# Attempt to discretize 'department'
df_copy['department'].fillna('', inplace=True)
for i in df_copy.index:
    department = df_copy.loc[i, 'department']
# for i, department in enumerate(df_copy['department']):
    if any(substring in department.lower() for substring in ['sale',
                                                             'customer',
                                                             'marketing', 'cs', 'member', 'commercial', 'retail',
                                                             'support']):
        df_copy.loc[i, 'department'] = 'Sales'
    elif any(substring in department.lower() for substring in ['data','information','qa', 'it', 'tech','engineer', ' \
            ''research','development', 'r&d', 'product']):
        df_copy.loc[i, 'department'] = 'Tech'
    elif any(substring in department.lower() for substring in ['creative','design', 'media']):
        df_copy.loc[i, 'department'] = 'Creative'

    elif any(substring in department.lower() for substring in ['operations','admin', 'management', 'account',
                                                               'finance']):
        df_copy.loc[i, 'department'] = 'Operations'

print('Unique departments after we merged categories:', df_copy['department'].nunique())




# ---------------- Salary
print('Percentage of null values:', 100 * (df['salary_range'].isnull().sum() / \
                                     df.shape[0]), '%')
df_salary = df_copy.copy()

# Split the "salary_range" column into two separate columns
df_salary[['min_salary', 'max_salary']] = df['salary_range'].str.split('-',
                                                                      expand=True)

# Convert the new columns to numeric types
df_salary['min_salary'] = pd.to_numeric(df_salary['min_salary'], errors='coerce')
df_salary['max_salary'] = pd.to_numeric(df_salary['max_salary'], errors='coerce')
df_salary.dropna(subset=['min_salary', 'max_salary'], inplace=True)

fig, ax = plt.subplots()
ax.boxplot([df_salary['min_salary'], df_salary['max_salary']])
ax.set_xticklabels(['Min Salary', 'Max Salary'])
ax.set_ylabel('Salary (USD)')
ax.set_title('Salary Range Box Plot')
plt.show()

# Remove outliers - First time
print('Num of records before:', df_salary.shape[0])
df_salary = df_salary.drop(df_salary[df_salary['min_salary'] > 10000000].index)
df_salary = df_salary.drop(df_salary[df_salary['max_salary'] > 20000000].index)
print('Num of records after:', df_salary.shape[0])

# Remove outliers - Second time
print('Num of records before:', df_salary.shape[0])
df_salary = df_salary.drop(df_salary[df_salary['min_salary'] > 300000].index)
df_salary = df_salary.drop(df_salary[df_salary['max_salary'] > 300000].index)
print('Num of records after:', df_salary.shape[0])

print('Max offered salary:', max(df_salary['max_salary']))
print('Min offered salary:', min(df_salary['min_salary']))

fig, ax = plt.subplots()
ax.boxplot([df_salary['min_salary'], df_salary['max_salary']])
ax.set_xticklabels(['Min Salary', 'Max Salary'])
ax.set_ylabel('Salary (USD)')
ax.set_title('Salary Range Box Plot')
plt.show()

df_salary['mean_salary'] = (df_salary['min_salary'] + df_salary['max_salary']) / 2

# Calculate the average salary for real and fake ads separately
real_mean_salary = df_salary[df_salary['fraudulent'] == 0]['mean_salary'].mean()
fake_mean_salary = df_salary[df_salary['fraudulent'] == 1]['mean_salary'].mean()

print('Average salary for real ads:', real_mean_salary)
print('Average salary for fake ads:', fake_mean_salary)

# Calculate the average salary range for real and fake ads separately
real_mean_range = df_salary[df_salary['fraudulent'] == 0]['max_salary'].mean() - df_salary[df_salary['fraudulent'] == 0]['min_salary'].mean()
fake_mean_range = df_salary[df_salary['fraudulent'] == 1]['max_salary'].mean() - df_salary[df_salary['fraudulent'] == 1]['min_salary'].mean()

print('Average salary range for real ads:', real_mean_range)
print('Average salary range for fake ads:', fake_mean_range)

# Salary categories
bins = [0, 40000, 80000, 120000, np.inf]
labels = ['low', 'medium', 'high', 'very high']
df_salary['salary_category'] = pd.cut(df_salary['mean_salary'], bins=bins, labels=labels)
grouped_df = df_salary.groupby('salary_category').agg(num_records=('mean_salary', 'count'),
                                                       num_fake_ads=('fraudulent', 'sum'))
grouped_df['percentage_fake_ads'] = (grouped_df['num_fake_ads'] / grouped_df['num_records']) * 100

print('Number of records and fake ads for each salary category:')
print(grouped_df)


# ---------------- Company Profile
print('Percentage of null values:', 100 * (df['company_profile'].isnull().sum() / \
                                     df.shape[0]), '%')
# 10 most frequent words
profiles = df['company_profile']
cleaned_profiles = []
for profile in profiles:
    if isinstance(profile, str):
        profile = re.sub(r'[^a-zA-Z\s]', '', profile)
        profile = profile.lower()
        cleaned_profiles.append(profile)

fake_profiles = []
real_profiles = []

for i in range(len(df)):
    if df['fraudulent'][i] == 1:
        if i < len(cleaned_profiles):
            fake_profiles.append(cleaned_profiles[i])
    else:
        if i < len(cleaned_profiles):
            real_profiles.append(cleaned_profiles[i])

fake_profile_tokens = [nltk.word_tokenize(profile) for profile in fake_profiles]
real_profile_tokens = [nltk.word_tokenize(profile) for profile in real_profiles]
fake_profile_tokens_cleaned = []
real_profile_tokens_cleaned = []

for tokens in fake_profile_tokens:
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    fake_profile_tokens_cleaned.extend(tokens)

for tokens in real_profile_tokens:
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    real_profile_tokens_cleaned.extend(tokens)

fake_profile_fdist = FreqDist(fake_profile_tokens_cleaned)
real_profile_fdist = FreqDist(real_profile_tokens_cleaned)

print("Most common words in fake company profiles:")
print([word for word in fake_profile_fdist.most_common() if word[0] not in stop_words and word[0] not in ['and', 'the', 'to', 'of', 'a', 'in', 'we', 'our', 'is', 'for']][:10])
print("Most common words in real company profiles:")
print([word for word in real_profile_fdist.most_common() if word[0] not in stop_words and word[0] not in ['and', 'the', 'to', 'of', 'a', 'in', 'we', 'our', 'is', 'for']][:10])

# Create a word cloud for the most common words
fake_wordcloud = WordCloud(width = 800, height = 800, background_color ='white').generate_from_frequencies(fake_profile_fdist)
real_wordcloud = WordCloud(width = 800, height = 800, background_color ='white').generate_from_frequencies(real_profile_fdist)

# Plot the word clouds
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
ax1.imshow(fake_wordcloud)
ax1.axis('off')
ax1.set_title('Fake Ads')
ax2.imshow(real_wordcloud)
ax2.axis('off')
ax2.set_title('Real Ads')
plt.show()

# Check the length of real and fake ads' profiles
fake_ads = pd.DataFrame({'profile': fake_profiles})
real_ads = pd.DataFrame({'profile': real_profiles})

fake_ads['profile_length'] = fake_ads['profile'].apply(len)
real_ads['profile_length'] = real_ads['profile'].apply(len)

fake_ads_mean = fake_ads['profile_length'].mean()
real_ads_mean = real_ads['profile_length'].mean()

print('Average company profile length of fake ads:', fake_ads_mean, 'words')
print('Average company profile length of real ads:', real_ads_mean, 'words')
print('Words difference:', abs(real_ads_mean - fake_ads_mean), 'words')

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()
df_copy['company_profile'] = df_copy['company_profile'].apply(lambda x: str(x) if isinstance(x, str) else '')
df_copy['sentiment'] = df_copy['company_profile'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Calculate the mean sentiment score for real and fake ads
real_mean = df_copy[df_copy['fraudulent'] == 0]['sentiment'].mean()
fake_mean = df_copy[df_copy['fraudulent'] == 1]['sentiment'].mean()
print(f"Average sentiment score for real ads: {real_mean:.2f}")
print(f"Average sentiment score for fake ads: {fake_mean:.2f}")



# ---------------- Description
print('Percentage of null values:', 100 * (df['description'].isnull().sum() / \
                                     df.shape[0]), '%')

# 10 most frequent words
descriptions = df['description']
cleaned_descriptions = []
for description in descriptions:
    if isinstance(description, str):
        description = re.sub(r'[^a-zA-Z\s]', '', description)
        description = description.lower()
        cleaned_descriptions.append(description)

fake_descriptions = []
real_descriptions = []

for i in range(len(df)):
    if df['fraudulent'][i] == 1:
        if i < len(cleaned_descriptions):
            fake_descriptions.append(cleaned_descriptions[i])
    else:
        if i < len(cleaned_descriptions):
            real_descriptions.append(cleaned_descriptions[i])

fake_description_tokens = [nltk.word_tokenize(description) for description in fake_descriptions]
real_description_tokens = [nltk.word_tokenize(description) for description in real_descriptions]
fake_description_tokens_cleaned = []
real_description_tokens_cleaned = []

for tokens in fake_description_tokens:
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    fake_description_tokens_cleaned.extend(tokens)

for tokens in real_description_tokens:
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    real_description_tokens_cleaned.extend(tokens)

fake_description_fdist = FreqDist(fake_description_tokens_cleaned)
real_description_fdist = FreqDist(real_description_tokens_cleaned)

print("Most common words in fake company description:")
print([word for word in fake_description_fdist.most_common() if word[0] not in stop_words and word[0] not in ['and', 'the', 'to', 'of', 'a', 'in', 'we', 'our', 'is', 'for']][:10])
print("Most common words in real company description:")
print([word for word in real_description_fdist.most_common() if word[0] not in stop_words and word[0] not in ['and', 'the', 'to', 'of', 'a', 'in', 'we', 'our', 'is', 'for']][:10])

# Create a word cloud for the most common words
fake_wordcloud = WordCloud(width = 800, height = 800, background_color ='white').generate_from_frequencies(fake_description_fdist)
real_wordcloud = WordCloud(width = 800, height = 800, background_color ='white').generate_from_frequencies(real_description_fdist)

# Plot the word clouds
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
ax1.imshow(fake_wordcloud)
ax1.axis('off')
ax1.set_title('Fake Ads')
ax2.imshow(real_wordcloud)
ax2.axis('off')
ax2.set_title('Real Ads')
plt.show()

# Check the length of real and fake ads' profiles
fake_ads = pd.DataFrame({'description': fake_descriptions})
real_ads = pd.DataFrame({'description': real_descriptions})

fake_ads['description_length'] = fake_ads['description'].apply(len)
real_ads['description_length'] = real_ads['description'].apply(len)

fake_ads_mean = fake_ads['description_length'].mean()
real_ads_mean = real_ads['description_length'].mean()

print('Average description length of fake ads:', fake_ads_mean, 'words')
print('Average description length of real ads:', real_ads_mean, 'words')
print('Words difference:', abs(real_ads_mean - fake_ads_mean), 'words')

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()
df_copy['description'] = df_copy['description'].apply(lambda x: str(x) if isinstance(x, str) else '')
df_copy['description_sentiment'] = df_copy['description'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Calculate the mean sentiment score for real and fake ads
real_mean = df_copy[df_copy['fraudulent'] == 0]['description_sentiment'].mean()
fake_mean = df_copy[df_copy['fraudulent'] == 1]['description_sentiment'].mean()
print(f"Average sentiment score for real ads: {real_mean:.2f}")
print(f"Average sentiment score for fake ads: {fake_mean:.2f}")

# ---------------- Requirements
print('Percentage of null values:', 100 * (df['requirements'].isnull().sum() / \
                                     df.shape[0]), '%')

# 10 most frequent words
requirements = df['requirements']
cleaned_requirements = []
for requirement in requirements:
    if isinstance(requirement, str):
        requirement = re.sub(r'[^a-zA-Z\s]', '', requirement)
        requirement = requirement.lower()
        cleaned_requirements.append(requirement)

fake_requirements = []
real_requirements = []

for i in range(len(df)):
    if df['fraudulent'][i] == 1:
        if i < len(cleaned_requirements):
            fake_requirements.append(cleaned_requirements[i])
    else:
        if i < len(cleaned_requirements):
            real_requirements.append(cleaned_requirements[i])

fake_requirement_tokens = [nltk.word_tokenize(requirement) for requirement in fake_requirements]
real_requirement_tokens = [nltk.word_tokenize(requirement) for requirement in real_requirements]
fake_requirement_tokens_cleaned = []
real_requirement_tokens_cleaned = []

for tokens in fake_requirement_tokens:
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    fake_requirement_tokens_cleaned.extend(tokens)

for tokens in real_requirement_tokens:
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    real_requirement_tokens_cleaned.extend(tokens)

fake_requirement_fdist = FreqDist(fake_requirement_tokens_cleaned)
real_requirement_fdist = FreqDist(real_requirement_tokens_cleaned)

print("Most common words in the requirements of fake ad:")
print([word for word in fake_requirement_fdist.most_common() if word[0] not in stop_words and word[0] not in ['and', 'the', 'to', 'of', 'a', 'in', 'we', 'our', 'is', 'for']][:10])
print("Most common words in the requirements of real ad:")
print([word for word in real_requirement_fdist.most_common() if word[0] not in stop_words and word[0] not in ['and', 'the', 'to', 'of', 'a', 'in', 'we', 'our', 'is', 'for']][:10])

# Create a word cloud for the most common words
fake_wordcloud = WordCloud(width = 800, height = 800, background_color ='white').generate_from_frequencies(fake_requirement_fdist)
real_wordcloud = WordCloud(width = 800, height = 800, background_color ='white').generate_from_frequencies(real_requirement_fdist)

# Plot the word clouds
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
ax1.imshow(fake_wordcloud)
ax1.axis('off')
ax1.set_title('Fake Ads')
ax2.imshow(real_wordcloud)
ax2.axis('off')
ax2.set_title('Real Ads')
plt.show()

# Check the length of real and fake ads' profiles
fake_ads = pd.DataFrame({'requirements': fake_requirements})
real_ads = pd.DataFrame({'requirements': real_requirements})

fake_ads['requirements_length'] = fake_ads['requirements'].apply(len)
real_ads['requirements_length'] = real_ads['requirements'].apply(len)

fake_ads_mean = fake_ads['requirements_length'].mean()
real_ads_mean = real_ads['requirements_length'].mean()

print('Average requirements length of fake ads:', fake_ads_mean, 'words')
print('Average requirements length of real ads:', real_ads_mean, 'words')
print('Words difference:', abs(real_ads_mean - fake_ads_mean), 'words')

# Relations between 'requirements' and 'required_experience / education'
df_requirements = df_copy.copy()
df_requirements['required_experience'] = df_requirements['required_experience'].astype(str)
df_requirements['requirements'] = df_requirements['requirements'].astype(str)
df_requirements['experience_in_requirements'] = df_requirements.apply(lambda row: int(row['required_experience'] in row['requirements']), axis=1)
percentage = df_requirements['experience_in_requirements'].mean() * 100
print(f"Percentage of records where 'required_experience' appears in the 'requirements': {percentage:.2f}%")

df_requirements['required_education'] = df_requirements['required_education'].astype(str)
df_requirements['requirements'] = df_requirements['requirements'].astype(str)
df_requirements['education_in_requirements'] = df_requirements.apply(lambda row: int(row['required_education'] in row['requirements']), axis=1)
education_percentage = df_requirements['education_in_requirements'].mean() * 100
print(f"Percentage of records where 'required_education' appears in the 'requirements': {education_percentage:.2f}%")



# ---------------- Benefits
print('Percentage of null values:', 100 * (df['benefits'].isnull().sum() / \
                                     df.shape[0]), '%')

# 10 most frequent words
benefits = df['benefits']
cleaned_benefits = []
for benefit in benefits:
    if isinstance(benefit, str):
        benefit = re.sub(r'[^a-zA-Z\s]', '', benefit)
        benefit = benefit.lower()
        cleaned_benefits.append(benefit)

fake_benefits = []
real_benefits = []

for i in range(len(df)):
    if df['fraudulent'][i] == 1:
        if i < len(cleaned_benefits):
            fake_benefits.append(cleaned_benefits[i])
    else:
        if i < len(cleaned_benefits):
            real_benefits.append(cleaned_benefits[i])

fake_benefit_tokens = [nltk.word_tokenize(requirement) for requirement in fake_benefits]
real_benefit_tokens = [nltk.word_tokenize(requirement) for requirement in real_benefits]
fake_benefit_tokens_cleaned = []
real_benefit_tokens_cleaned = []

for tokens in fake_benefit_tokens:
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    fake_benefit_tokens_cleaned.extend(tokens)

for tokens in real_benefit_tokens:
    tokens = [token for token in tokens if token not in stop_words and len(token) > 1]
    real_benefit_tokens_cleaned.extend(tokens)

fake_benefit_fdist = FreqDist(fake_benefit_tokens_cleaned)
real_benefit_fdist = FreqDist(real_benefit_tokens_cleaned)

print("Most common words in the benefits of fake ad:")
print([word for word in fake_benefit_fdist.most_common() if word[0] not in stop_words and word[0] not in ['and', 'the', 'to', 'of', 'a', 'in', 'we', 'our', 'is', 'for']][:10])
print("Most common words in the benefits of real ad:")
print([word for word in real_benefit_fdist.most_common() if word[0] not in stop_words and word[0] not in ['and', 'the', 'to', 'of', 'a', 'in', 'we', 'our', 'is', 'for']][:10])

# Create a word cloud for the most common words
fake_wordcloud = WordCloud(width = 800, height = 800, background_color ='white').generate_from_frequencies(fake_benefit_fdist)
real_wordcloud = WordCloud(width = 800, height = 800, background_color ='white').generate_from_frequencies(real_benefit_fdist)

# Plot the word clouds
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
ax1.imshow(fake_wordcloud)
ax1.axis('off')
ax1.set_title('Fake Ads')
ax2.imshow(real_wordcloud)
ax2.axis('off')
ax2.set_title('Real Ads')
plt.show()

# Check the length of real and fake ads' benefits
fake_ads = pd.DataFrame({'benefits': fake_benefits})
real_ads = pd.DataFrame({'benefits': real_benefits})

fake_ads['benefits_length'] = fake_ads['benefits'].apply(len)
real_ads['benefits_length'] = real_ads['benefits'].apply(len)

fake_ads_mean = fake_ads['benefits_length'].mean()
real_ads_mean = real_ads['benefits_length'].mean()

print('Average benefits length of fake ads:', fake_ads_mean, 'words')
print('Average benefits length of real ads:', real_ads_mean, 'words')
print('Words difference:', abs(real_ads_mean - fake_ads_mean), 'words')

# Sentiment Analysis
sia = SentimentIntensityAnalyzer()
df_copy['benefits'] = df_copy['benefits'].apply(lambda x: str(x) if isinstance(x, str) else '')
df_copy['benefits_sentiment'] = df_copy['benefits'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Calculate the mean sentiment score for real and fake ads
real_mean = df_copy[df_copy['fraudulent'] == 0]['benefits_sentiment'].mean()
fake_mean = df_copy[df_copy['fraudulent'] == 1]['benefits_sentiment'].mean()
print(f"Average sentiment score for real ads: {real_mean:.2f}")
print(f"Average sentiment score for fake ads: {fake_mean:.2f}")




# ---------------- Telecommuting
# plottinhg binaric feature with count plot
sns.set_theme(style = "whitegrid")
sns.countplot(x = 'telecommuting', data = df, hue = "fraudulent")
plt.title("Telecommuting Count Plot", fontsize = 12)
plt.show()

# plottinhg binaric feature data of Fraud Advertisments
df_fraud['telecommuting'].value_counts(normalize = True, dropna = False).plot.bar(rot = 0)
plt.title("Telecommuting Bar Plot of Fraud Advertisments", fontsize = 12)
plt.xlabel('Telecommuting', fontsize = 10)
plt.ylabel('Frequency (%)', fontsize = 10)
plt.show()

# plottinhg binaric feature data of Real Advertisments
df_real['telecommuting'].value_counts(normalize = True, dropna = False).plot.bar(rot = 0)
plt.title("Telecommuting Bar Plot of Real Advertisments", fontsize = 12)
plt.xlabel('Telecommuting', fontsize = 10)
plt.ylabel('Frequency (%)', fontsize = 10)
plt.show()




# ---------------- Has company logo
# plottinhg binaric feature with count plot
sns.set_theme(style = "whitegrid")
sns.countplot(x = 'has_company_logo', data = df, hue = "fraudulent")
plt.title("Has Company Logo Count Plot", fontsize = 12)
plt.show()

# plottinhg binaric feature data of Fraud Advertisments
df_fraud['has_company_logo'].value_counts(normalize = True, dropna = False).plot.bar(rot = 0)
plt.title("Has Company Logo Bar Plot of Fraud Advertisments", fontsize = 12)
plt.xlabel('Has Company Logo', fontsize = 10)
plt.ylabel('Frequency (%)', fontsize = 10)
plt.show()

# plottinhg binaric feature data of Real Advertisments
df_real['has_company_logo'].value_counts(normalize = True, dropna = False).plot.bar(rot = 0)
plt.title("Has Company Logo Bar Plot of Real Advertisments", fontsize = 12)
plt.xlabel('Has Company Logo', fontsize = 10)
plt.ylabel('Frequency (%)', fontsize = 10)
plt.show()




# ---------------- Has questions
# plottinhg binaric feature with count plot
sns.set_theme(style = "whitegrid")
sns.countplot(x = 'has_questions', data = df, hue = "fraudulent")
plt.title("Has Questions Count Plot", fontsize = 12)
plt.show()

# plottinhg binaric feature data of Fraud Advertisments
df_fraud['has_questions'].value_counts(normalize = True, dropna = False).plot.bar(rot = 0)
plt.title("Has Questions Bar Plot of Fraud Advertisments", fontsize = 12)
plt.xlabel('Has Questions', fontsize = 10)
plt.ylabel('Frequency (%)', fontsize = 10)
plt.show()

# plottinhg binaric feature data of Real Advertisments
df_real['has_questions'].value_counts(normalize = True, dropna = False).plot.bar(rot = 0)
plt.title("Has Questions Bar Plot of Real Advertisments", fontsize = 12)
plt.xlabel('Has Questions', fontsize = 10)
plt.ylabel('Frequency (%)', fontsize = 10)
plt.show()




# ---------------- Employment type
# plottinhg categorial data of Fraud Advertisments
df_fraud['employment_type'].value_counts(normalize = True, dropna = False).plot(kind = 'bar')
plt.title("Employment Type Bar Plot of Fraud Advertisments", fontsize = 12)
plt.xlabel('Employment Type', fontsize = 10)
plt.ylabel('Frequency (%)', fontsize = 10)
plt.show()

# plottinhg categorial data of Real Advertisments
df_real['employment_type'].value_counts(normalize = True, dropna = False).plot(kind = 'bar')
plt.title("Employment Type Bar Plot of Real Advertisments", fontsize = 12)
plt.xlabel('Employment Type', fontsize = 10)
plt.ylabel('Frequency (%)', fontsize = 10)
plt.show()

# use value_counts() to get a count of each unique string in the 'employment_type' column
most_common_fraud = df_fraud['employment_type'].value_counts(normalize = True, dropna = False)
print("The Freqyency of each Employment Type in fraud Advertisments is: ", most_common_fraud)
most_common_real = df_real['employment_type'].value_counts(normalize = True, dropna = False)
print("The Freqyency of each Employment Type in real Advertisments is: ", most_common_real)
print("The number of unique values in the feature employment type is:", df['employment_type'].nunique())




# ---------------- Required experience
# plottinhg categorial data of Fraud Advertisments
df_fraud['required_experience'].value_counts(normalize = True, dropna = False).plot(kind = 'bar', fontsize = 6)
plt.title("Required Experience Bar Plot of Fraud Advertisments", fontsize = 12)
plt.xlabel('Required Experience', fontsize = 10)
plt.xticks(rotation = 0)
plt.ylabel('Frequency (%)', fontsize = 10)
plt.show()

# plottinhg categorial data of Real Advertisments
df_real['required_experience'].value_counts(normalize = True, dropna = False).plot(kind = 'bar', fontsize = 6)
plt.xlabel('Required Experience', fontsize = 10)
plt.xticks(rotation = 0) # set the x-axis tick labels horizontal
plt.ylabel('Frequency (%)', fontsize = 10)
plt.title("Required Experience Bar Plot of Real Advertisments", fontsize = 12)
plt.show()

# use value_counts() to get a count of each unique string in the 'required_experience' column
most_common_fraud = df_fraud['required_experience'].value_counts(normalize = True, dropna = False)
print("The Frequency of each Required Experience Type in fraud Advertisments is: ", most_common_fraud)
most_common_real = df_real['required_experience'].value_counts(normalize = True, dropna = False)
print("The Frequency of each Required Experience Type in real Advertisments is: ", most_common_real)
print("The number of unique values in the feature required experience is:", df['required_experience'].nunique()) #how many unique values in this feature





# ---------------- Required education
# plottinhg categorial data of Fraud Advertisments
df_fraud['required_education'].value_counts(normalize = True, dropna = False).plot(kind = 'bar', fontsize = 6)
plt.title("Required Education Bar Plot of Fraud Advertisments", fontsize = 12)
plt.xlabel('Required Education', fontsize = 10)
plt.ylabel('Frequency (%)', fontsize = 10)
plt.show()

# plottinhg categorial data of Real Advertisments
df_real['required_education'].value_counts(normalize = True, dropna = False).plot(kind = 'bar', fontsize = 6)
plt.xlabel('Required Education', fontsize = 10)
plt.ylabel('Frequency (%)', fontsize = 10)
plt.title("Required Education Bar Plot of Real Advertisments", fontsize = 12)
plt.show()

# use value_counts() to get a count of each unique string in the 'function' column, select the top 3
most_common_fraud = df_fraud['required_education'].value_counts(normalize = True, dropna = False).head(3)
print("The three most common Required Education Types in fraud Advertisments are: ", most_common_fraud)
most_common_real = df_real['required_education'].value_counts(normalize = True, dropna = False).head(3)
print("The three most common Required Education Types in real Advertisments are: ", most_common_real)
print("The number of unique values in the feature required education is:", df['required_education'].nunique())




# ---------------- Industry
print('Percentage of null values:', 100 * (df['industry'].isnull().sum() / \
                                     df.shape[0]), '%')
print('Unique departments:', df['industry'].nunique())

# Discretize 'industry'
df_copy['industry'].fillna('', inplace=True)
for i in df_copy.index:
    industry = df_copy.loc[i, 'industry']
    if any(substring in industry.lower() for substring in ['internet', 'data', 'computer', 'information',
                                                           'technology', 'computer software',
                                                           'telecommunications',
               'information technology and services', 'computer networking',
               'computer & network security', 'computer hardware', 'semiconductors',
               'wireless']):
        df_copy.loc[i, 'industry'] = 'technology'

    elif any(substring in industry.lower() for substring in ['medical practice', 'pharmaceuticals', 'hospital & health care',
               'cosmetics', 'health, wellness and fitness', 'medical devices',
               'mental health care', 'health', 'medical']):
        df_copy.loc[i, 'industry'] = 'health'

    elif any(substring in industry.lower() for substring in ['financial services', 'finance', 'insurance', 'venture capital & private equity',
            'investment banking', 'investment management', 'capital markets']):
        df_copy.loc[i, 'industry'] = 'finance'

    elif any(substring in industry.lower() for substring in ['retail', 'consumer services', 'consumer electronics', 'consumer goods',
           'restaurants', 'apparel & fashion', 'sporting goods', 'luxury goods & jewelry', 'cosmetics']):
        df_copy.loc[i, 'industry'] = 'retail'
    elif any(substring in industry.lower() for substring in ['oil & energy', 'building materials', 'materials',
                             'electrical/electronic manufacturing',
                             'mechanical or industrial engineering', 'machinery',
                             'renewables & environment', 'plastics']):
        df_copy.loc[i, 'industry'] = 'manufacturing'
    else:
        df_copy.loc[i, 'industry'] = 'other'

print(df_copy['industry'].nunique())
print(df_copy['industry'].unique())

counts = df_copy['industry'].value_counts()
plt.bar(counts.index, counts.values)
plt.title('Industry Counts')
plt.xlabel('Industry')
plt.ylabel('Count')
plt.show()

industry_fraud = df_copy.groupby('industry')['fraudulent'].mean() * 100
industry_fraud = industry_fraud.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=industry_fraud.index, y=industry_fraud.values, color='skyblue')
plt.xticks(rotation=90)
plt.title('Percentage of fake ads for each industry')
plt.ylabel('Percentage of fake ads')
plt.xlabel('Industry')
plt.tight_layout()
plt.show()



# ---------------- Function
# plottinhg categorial data of Fraud Advertisments
df_fraud['function'].value_counts(normalize = True).plot(kind = 'bar', fontsize = 6)
plt.title("Function Bar Plot of Fraud Advertisments", fontsize = 12)
plt.xlabel('Function', fontsize = 10)
plt.ylabel('Frequency (%)', fontsize = 10)
plt.show()
# plottinhg categorial data of Real Advertisments
df_real['function'].value_counts(normalize = True).plot(kind = 'bar', fontsize = 6)
plt.xlabel('Function', fontsize = 10)
plt.ylabel('Frequency (%)', fontsize = 10)
plt.title("Function Bar Plot of Real Advertisments", fontsize = 12)
plt.show()

# use value_counts() to get a count of each unique string in the 'function' column, select the top 3
most_common_fraud = df_fraud['function'].value_counts(normalize = True).head(3)
print("The three most common functions in fraud Advertisments are: ", most_common_fraud)
most_common_real = df_real['function'].value_counts(normalize = True).head(3)
print("The three most common functions in real Advertisments are: ", most_common_real)
print("The number of unique values in the feature function is:", df['function'].nunique()) #how many unique values in this feature


# ---------------- Fraudulent
# plottinhg binaric feature data
df['fraudulent'].value_counts(normalize = True).plot.bar(rot = 0)
plt.title("Fraudulent Bar Plot", fontsize = 12)
plt.xlabel('Fraudulent', fontsize = 10)
plt.ylabel('Frequency (%)', fontsize = 10)
plt.show()















# -------------------------------------------- Q2 - Pre Processing ----------------------------------------------------
import numpy as np
import pandas as pd
import re
import nltk
from nltk import FreqDist
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Reading the file
df = pd.read_csv("C:/Users/Doron/לימודים/קורסים/למידת מכונה/פרויקט/XY_train.csv")
df_copy = df.copy()
df_fraud = df[df['fraudulent'] == 1]
df_real = df[df['fraudulent'] == 0]


# --------------------------- Dealing with Duplications
df = df.drop('job_id', axis = 1)
real_ads = df[df['fraudulent'] == 0]
real_ads = real_ads.drop_duplicates(subset=real_ads.columns.difference(['fraudulent']))
df = pd.concat([real_ads, df[df['fraudulent'] == 1]])


# --------------------------- Balancing the Data by Up Sampling
fake_ads = df[df['fraudulent'] == 1]
real_ads = df[df['fraudulent'] == 0]

desired_proportion = 0.25
n_desired_fake = int(desired_proportion * len(real_ads))

# Upsample the fake ads to the desired number
fake_ads_upsampled = resample(fake_ads, replace=True, n_samples=n_desired_fake, random_state=123)
df = pd.concat([real_ads, fake_ads_upsampled])


# --------------------------- Missing Values & Data Conversion
stemmer = PorterStemmer()
stop_words = stopwords.words('english')

def preprocess_text(text):
    if isinstance(text, float) and np.isnan(text):
        return ''
    text = text.lower()
    text = re.sub(r'\W+|\d+', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    text = ' '.join(tokens)
    return text

# --------------------------- Title
df['title'] = df['title'].apply(preprocess_text)



# --------------------------- Location
df['country'] = df['location'].str[:2]
df = df.drop('location', axis=1)

# Get the frequency of each country
country_freq = df[df['country'].notna()]['country'].value_counts(normalize=True)
fill_func = lambda x: np.random.choice(country_freq.index, p=country_freq.values) if pd.isna(x) else x
df['country'] = df['country'].apply(fill_func)


# --------------------------- Salary
# Convert 'Salary_range' into 'mean salary'
df[['min_salary', 'max_salary']] = df['salary_range'].str.split('-',expand=True)
df['min_salary'] = pd.to_numeric(df['min_salary'], errors='coerce')
df['max_salary'] = pd.to_numeric(df['max_salary'], errors='coerce')
df['mean_salary'] = (df['min_salary'] + df['max_salary']) / 2
df.drop(['min_salary', 'max_salary', 'salary_range'], axis=1, inplace=True)

# Fill missing values with the mean salary
mean_salary = df.loc[df['mean_salary'] < 1000000, 'mean_salary'].mean()
df['mean_salary'].fillna(mean_salary, inplace=True)



# --------------------------- Company Profile
# Process text
df['company_profile'] = df['company_profile'].apply(preprocess_text)

# Fill missing values according to the 50 most frequent words
counter = Counter(" ".join(df['company_profile']).split())
top_words = [word for word, count in counter.most_common(50)]
df['company_profile'] = df['company_profile'].replace('', ' '.join(top_words))




# --------------------------- Description
# Process text
df['description'] = df['description'].apply(preprocess_text)

# Fill missing values according to the 100 most frequent words
counter = Counter(" ".join(df['description']).split())
top_words = [word for word, count in counter.most_common(100)]
df['description'] = df['description'].replace('', ' '.join(top_words))



# --------------------------- Requirements
# Process text
df['requirements'] = df['requirements'].apply(preprocess_text)

# Fill missing values according to the 50 most frequent words
counter = Counter(" ".join(df['requirements']).split())
top_words = [word for word, count in counter.most_common(50)]
df['requirements'] = df['requirements'].replace('', ' '.join(top_words))



# --------------------------- Benefits
# Process text
df['benefits'] = df['benefits'].apply(preprocess_text)

# Fill missing values according to the 20 most frequent words
counter = Counter(" ".join(df['benefits']).split())
top_words = [word for word, count in counter.most_common(20)]
df['benefits'] = df['benefits'].replace('', ' '.join(top_words))



# --------------------------- Telecommuting & Has Company Logo & Has Questions
# No changes




# --------------------------- Employment Type
for index, e in df.iterrows():
    if pd.isnull(e['employment_type']) or e['employment_type'] == '':
        descriptions = [e['description'], e['benefits'], e['requirements']]
        for d in descriptions:
            if isinstance(d, str):
                if 'full-time' in d.lower() or 'full time' in d.lower():
                    df.at[index, 'employment_type'] = 'Full-time'
                    break
                elif 'part-time' in d.lower() or 'part time' in d.lower():
                    df.at[index, 'employment_type'] = 'Part-time'
                    break
                elif 'temporary' in d.lower():
                    df.at[index, 'employment_type'] = 'Temporary'
                    break
                elif 'contract' in d.lower():
                    df.at[index, 'employment_type'] = 'Contract'
                    break
                elif 'other' in d.lower():
                    df.at[index, 'employment_type'] = 'Other'
                    break
        else: # blanks with no clues will be filled with the most frequent value (null not included)
            if e['fraudulent'] == 0: # real
                # Get the most frequent value of column 'employment_type' as a string
                df.at[index, 'employment_type'] = df_real['employment_type'].dropna().value_counts().idxmax()
            else: # fraud
                # Get the most frequent value of column 'employment_type' as a string
                df.at[index, 'employment_type'] = df_fraud['employment_type'].dropna().value_counts().idxmax()




# --------------------------- Required Experience
for index, e in df.iterrows():
    if pd.isnull(e['required_experience']) or e['required_experience'] == '':
        if 'director' in str(e['description']) or 'director' in str(e['benefits']) or 'director' in str(e['requirements']):
            df.at[index, 'required_experience'] = "Director"
        elif 'mid-senior' in str(e['description']) or 'mid-senior' in str(e['benefits']) or 'mid-senior' in str(e['requirements']) or "mid senior" in str(e['description']) or "mid senior" in str(e['benefits']) or "mid senior" in str(e['requirements']):
            df.at[index, 'required_experience'] = "Mid-Senior level"
        elif "Master's Degree" in str(e['description']) or "Master's Degree" in str(e['benefits']) or "Master's Degree" in str(e['requirements']) or "high school or equivalent" in str(e['description']) or "high school or equivalent" in str(e['benefits']) or "high school or equivalent" in str(e['requirements']):
            df.at[index, 'required_experience'] = "High School or equivalent"
        elif 'high school diploma' in str(e['description']) or 'high school diploma' in str(e['benefits']) or 'high school diploma' in str(e['requirements']) or "high school or equivalent" in str(e['description']) or "high school or equivalent" in str(e['benefits']) or "high school or equivalent" in str(e['requirements']):
            df.at[index, 'required_experience'] = "High School or equivalent"
        else: # blanks with no clues will be filled with the most frequent value (null not included)
            if e['fraudulent'] == 0: # real
                # Get the most frequent value of column 'required_experience' as a string
                df.at[index, 'required_experience'] = df_real['required_experience'].dropna().value_counts().idxmax()
            else: # fraud
                # Get the most frequent value of column 'required_experience' as a string
                df.at[index, 'required_experience'] = df_fraud['required_experience'].dropna().value_counts().idxmax()





# --------------------------- Required Education
for index, row in df.iterrows():
    if pd.isnull(row['required_education']) or row['required_education'] == '':
        if any(word in str(row['description']).lower() or str(row['benefits']).lower() or str(row['requirements']).lower() for word in ['bachelor degree', "bachelor's degree", ' ba ', 'bachelor', 'degree']):
            df.at[index, 'required_education'] = "Bachelor's Degree"
        elif any(word in str(row['description']).lower() or str(row['benefits']).lower() or str(row['requirements']).lower() for word in ['high school diploma', "high school or equivalent"]):
            df.at[index, 'required_education'] = "High School or equivalent"
        elif "associate degree" in str(row['description']).lower() or str(row['benefits']).lower() or str(row['requirements']).lower():
            df.at[index, 'required_education'] = "Associate Degree"
        elif any(word in str(row['description']).lower() or str(row['benefits']).lower() or str(row['requirements']).lower() for word in ["master's degree", ' mba ', 'ma', 'master degree']):
            df.at[index, 'required_education'] = "Master's Degree"
        elif any(word in str(row['description']).lower() or str(row['benefits']).lower() or str(row['requirements']).lower() for word in ['some college coursework completed', 'college coursework', 'college completed']):
            df.at[index, 'required_education'] = "Some College Coursework Completed"
        elif 'unspecified' in str(row['description']).lower() or str(row['benefits']).lower() or str(row['requirements']).lower():
            df.at[index, 'required_education'] = "Unspecified"
        elif 'vocational' in str(row['description']).lower() or str(row['benefits']).lower() or str(row['requirements']).lower():
            if any(word in str(row['description']).lower() or str(row['benefits']).lower() or str(row['requirements']).lower() for word in ['hs diploma', 'high school diploma']):
                df.at[index, 'required_education'] = "Vocational - HS Diploma"
            elif 'degree' in str(row['description']).lower() or str(row['benefits']).lower() or str(row['requirements']).lower():
                df.at[index, 'required_education'] = "Vocational - Degree"
            else:
                df.at[index, 'required_education'] = "Vocational"
        elif 'professional' in str(row['description']).lower() or str(row['benefits']).lower() or str(row['requirements']).lower():
            df.at[index, 'required_education'] = "Professional"
        elif 'doctorate' in str(row['description']).lower() or str(row['benefits']).lower() or str(row['requirements']).lower():
            df.at[index, 'required_education'] = "Doctorate"
        elif 'certification' in str(row['description']).lower() or str(row['benefits']).lower() or str(row['requirements']).lower():
            df.at[index, 'required_education'] = "Certification"
        else:  # blanks with no clues will be filled with the most frequent value (null not included)
            if row['fraudulent'] == 0:  # real
                # Get the most frequent value of column 'required_education' as a string
                df.at[index, 'required_education'] = df_real['required_education'].dropna().value_counts().idxmax()
            else:  # fraud
                # Get the most frequent value of column 'required_education' as a string
                df.at[index, 'required_education'] = df_fraud['required_education'].dropna().value_counts().idxmax()




# --------------------------- Industry
df['industry'] = df['industry'].fillna(df['function'])

# Discretize 'industry'
def categorize_industry(industry):
    if isinstance(industry, str):
        if any(substring in industry.lower() for substring in
               ['internet', 'data', 'computer', 'information', 'technology', 'computer software', 'telecommunications',
                'information technology and services', 'computer networking', 'computer & network security',
                'computer hardware', 'semiconductors', 'wireless']):
            return 'technology'
        elif any(substring in industry.lower() for substring in
                 ['medical practice', 'pharmaceuticals', 'hospital & health care',
                  'cosmetics', 'health, wellness and fitness', 'medical devices',
                  'mental health care', 'health', 'medical']):
            return 'health'
        elif any(substring in industry.lower() for substring in
                 ['financial services', 'finance', 'insurance', 'venture capital & private equity',
                  'investment banking', 'investment management', 'capital markets']):
            return 'finance'
        elif any(substring in industry.lower() for substring in
                 ['retail', 'consumer services', 'consumer electronics', 'consumer goods',
                  'restaurants', 'apparel & fashion', 'sporting goods', 'luxury goods & jewelry', 'cosmetics']):
            return 'retail'
        elif any(substring in industry.lower() for substring in
                 ['oil & energy', 'building materials', 'materials',
                  'electrical/electronic manufacturing',
                  'mechanical or industrial engineering', 'machinery',
                  'renewables & environment', 'plastics']):
            return 'manufacturing'
        else:
            return 'other'
    else:
        return industry

df['industry'] = df['industry'].apply(categorize_industry)

# Fill missing values according to the frequency
industry_freq = df['industry'].value_counts(normalize=True)
def fill_industry(x):
    if pd.isna(x):
        return np.random.choice(industry_freq.index, p=industry_freq.values)
    else:
        return x

df['industry'] = df['industry'].apply(fill_industry)





# --------------------------- Function
def categorize_function(function):
    if isinstance(function, str):
        if function.lower() in ['information technology', 'data analyst', 'engineering', 'product management', 'design']:
            return 'Technology'
        elif function.lower() in ['accounting/auditing', 'business development', 'business analyst', 'consulting', 'finance', 'financial analyst', 'management', 'marketing', 'public relations', 'sales', 'strategy/planning', 'general business']:
            return 'Business'
        elif function.lower() in ['art/creative', 'writing/editing', 'advertising']:
            return 'Creative'
        elif function.lower() == 'customer service':
            return 'Customer Service'
        elif function.lower() in ['administrative', 'human resources', 'distribution', 'education', 'project management', 'quality assurance', 'supply chain', 'production', 'research', 'science', 'training']:
            return 'Operations'
        else:
            return 'Other'
    else:
        return function

df['function'] = df['function'].apply(categorize_function)

# Fill missing values according to the frequency
function_freq = df['function'].value_counts(normalize=True)
def fill_function(x):
    if pd.isna(x):
        return np.random.choice(function_freq.index, p=function_freq.values)
    else:
        return x

df['function'] = df['function'].apply(fill_function)




# --------------------------- Save Pre-Processed DF to a CSV file
df.to_csv('Processed_DF.csv', index=False)











# -------------------------------------------- Q2 - Feature Extraction ------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import country_converter as coco

# Reading the file
df = pd.read_csv("Processed_DF.csv")

# ------------------------------------- Continent
df['continent'] = coco.convert(names=df['country'], to='continent')



# ------------------------------------- Qualification
df['qualification'] = None

for index, row in df.iterrows():
    if ("High School or equivalent" in row['required_education']) or ("Vocational" in row['required_education']) or ("Some High School Coursework" in row['required_education']) or ("Entry level" in row['required_experience']):
        df.loc[index, 'qualification'] = "Entry level"
    elif ("Not Applicable" in row['required_education']) or ("Vocational - HS Diploma" in row['required_education']) or ("Unspecified" in row['required_education']) or ("Unspecified" in row['required_experience']):
        df.loc[index, 'qualification'] = "Unspecified"
    elif ("Associate Degree" in row['required_education']) or ("Vocational - Degree" in row['required_education']) or ("Some College Coursework Completed" in row['required_education']) or ("Associate" in row['required_education']) or ("Internship" in row['required_experience']):
        df.loc[index, 'qualification'] = "Associate"
    elif ("Professional" in row['required_education']) or ("Bachelor's Degree" in row['required_education']) or ("Master's Degree" in row['required_education']) or ("Certification" in row['required_education']) or ("Mid-Senior level" in row['required_experience']):
        df.loc[index, 'qualification'] = "Professional"
    elif ("Doctorate" in row['required_education']) or ("Executive" in row['required_education']) or ("Director" in row['required_education']) or ("Executive" in row['required_experience']):
        df.loc[index, 'qualification'] = "Executive"





# ------------------------------------- Has Salary
df['has_salary'] = df['mean_salary'].apply(lambda x: 0 if abs(x - 67428.0422390109) < 0.0001 else 1)




# ------------------------------------- Has Department
df['has_department'] = df['department'].notnull().astype(int)





# ------------------------------------- Text Feature Extraction: Title
# Feature extraction using TF-IDF
tfidf = TfidfVectorizer()
tfidf.fit(df['title'])
tfidf_features = tfidf.transform(df['title'])

# Compute the TF-IDF scores separately for real and fake ads
real_mask = df['fraudulent'] == 0
fake_mask = df['fraudulent'] == 1
real_tfidf = tfidf_features[real_mask]
fake_tfidf = tfidf_features[fake_mask]

# Compute the mean TF-IDF scores for each word in real and fake ads
real_mean_tfidf = real_tfidf.mean(axis=0).A1
fake_mean_tfidf = fake_tfidf.mean(axis=0).A1

# Compute the difference in mean TF-IDF scores between real and fake ads
tfidf_diff = real_mean_tfidf - fake_mean_tfidf
n = 100
top_features = tfidf_diff.argsort()[-n:]
feature_names = np.array([f"title_{name}" for name in tfidf.get_feature_names_out()])[top_features]

# Add the top n TF-IDF features to the dataset
tfidf_df = pd.DataFrame(tfidf_features[:, top_features].toarray(), columns=feature_names)
df = pd.concat([df, tfidf_df], axis=1)




# ------------------------------------- Text Feature Extraction: Company Profile
# Feature extraction using TF-IDF
tfidf = TfidfVectorizer()
tfidf.fit(df['company_profile'])
tfidf_features = tfidf.transform(df['company_profile'])

# Compute the TF-IDF scores separately for real and fake ads
real_mask = df['fraudulent'] == 0
fake_mask = df['fraudulent'] == 1
real_tfidf = tfidf_features[real_mask]
fake_tfidf = tfidf_features[fake_mask]

# Compute the mean TF-IDF scores for each word in real and fake ads
real_mean_tfidf = real_tfidf.mean(axis=0).A1
fake_mean_tfidf = fake_tfidf.mean(axis=0).A1

# Compute the difference in mean TF-IDF scores between real and fake ads
tfidf_diff = real_mean_tfidf - fake_mean_tfidf

n = 100
top_features = tfidf_diff.argsort()[-n:]
feature_names = np.array([f"company_{name}" for name in tfidf.get_feature_names_out()])[top_features]

# Add the top n TF-IDF features to the dataset
tfidf_df = pd.DataFrame(tfidf_features[:, top_features].toarray(), columns=feature_names)
df = pd.concat([df, tfidf_df], axis=1)




# ------------------------------------- Text Feature Extraction: Description
# Feature extraction using TF-IDF
tfidf = TfidfVectorizer()
tfidf.fit(df['description'])
tfidf_features = tfidf.transform(df['description'])

# Compute the TF-IDF scores separately for real and fake ads
real_mask = df['fraudulent'] == 0
fake_mask = df['fraudulent'] == 1
real_tfidf = tfidf_features[real_mask]
fake_tfidf = tfidf_features[fake_mask]

# Compute the mean TF-IDF scores for each word in real and fake ads
real_mean_tfidf = real_tfidf.mean(axis=0).A1
fake_mean_tfidf = fake_tfidf.mean(axis=0).A1

# Compute the difference in mean TF-IDF scores between real and fake ads
tfidf_diff = real_mean_tfidf - fake_mean_tfidf

n = 100
top_features = tfidf_diff.argsort()[-n:]
feature_names = np.array([f"description_{name}" for name in tfidf.get_feature_names_out()])[top_features]

# Add the top n TF-IDF features to the dataset
tfidf_df = pd.DataFrame(tfidf_features[:, top_features].toarray(), columns=feature_names)
df = pd.concat([df, tfidf_df], axis=1)




# ------------------------------------- Text Feature Extraction: Requirements
# Feature extraction using TF-IDF
tfidf = TfidfVectorizer()
tfidf.fit(df['requirements'])
tfidf_features = tfidf.transform(df['requirements'])

# Compute the TF-IDF scores separately for real and fake ads
real_mask = df['fraudulent'] == 0
fake_mask = df['fraudulent'] == 1
real_tfidf = tfidf_features[real_mask]
fake_tfidf = tfidf_features[fake_mask]

# Compute the mean TF-IDF scores for each word in real and fake ads
real_mean_tfidf = real_tfidf.mean(axis=0).A1
fake_mean_tfidf = fake_tfidf.mean(axis=0).A1

# Compute the difference in mean TF-IDF scores between real and fake ads
tfidf_diff = real_mean_tfidf - fake_mean_tfidf

n = 100
top_features = tfidf_diff.argsort()[-n:]
feature_names = np.array([f"requirements_{name}" for name in tfidf.get_feature_names_out()])[top_features]

# Add the top n TF-IDF features to the dataset
tfidf_df = pd.DataFrame(tfidf_features[:, top_features].toarray(), columns=feature_names)
df = pd.concat([df, tfidf_df], axis=1)




# ------------------------------------- Text Feature Extraction: Benefits
# Feature extraction using TF-IDF
tfidf = TfidfVectorizer()
tfidf.fit(df['benefits'])
tfidf_features = tfidf.transform(df['benefits'])

# Compute the TF-IDF scores separately for real and fake ads
real_mask = df['fraudulent'] == 0
fake_mask = df['fraudulent'] == 1
real_tfidf = tfidf_features[real_mask]
fake_tfidf = tfidf_features[fake_mask]

# Compute the mean TF-IDF scores for each word in real and fake ads
real_mean_tfidf = real_tfidf.mean(axis=0).A1
fake_mean_tfidf = fake_tfidf.mean(axis=0).A1

# Compute the difference in mean TF-IDF scores between real and fake ads
tfidf_diff = real_mean_tfidf - fake_mean_tfidf

n = 100
top_features = tfidf_diff.argsort()[-n:]
feature_names = np.array([f"benefits_{name}" for name in tfidf.get_feature_names_out()])[top_features]

# Add the top n TF-IDF features to the dataset
tfidf_df = pd.DataFrame(tfidf_features[:, top_features].toarray(), columns=feature_names)
df = pd.concat([df, tfidf_df], axis=1)




# --------------------------- Save Feature-Extracted DF to a CSV file
df.to_csv('Feature_Extracted_DF.csv', index=False)









# -------------------------------------------- Q2 - Feature Representation --------------------------------------------
import pandas as pd


# Reading the file
df = pd.read_csv('Feature_Extracted_DF.csv')


# Dropping text features that already been converted to numerical vectors earlier
df = df.drop(columns=['title', 'department', 'company_profile', 'description', 'requirements', 'benefits'])



# Converting categorial features to numeric ones using One-Hot Encode
df['employment_type'] = df['employment_type'].astype('category')
df = pd.get_dummies(df, columns=['employment_type'], dtype=int)

df['required_experience'] = df['required_experience'].astype('category')
df = pd.get_dummies(df, columns=['required_experience'], dtype=int)

df['required_education'] = df['required_education'].astype('category')
df = pd.get_dummies(df, columns=['required_education'], dtype=int)

df['industry'] = df['industry'].astype('category')
df = pd.get_dummies(df, columns=['industry'], dtype=int)

df['function'] = df['function'].astype('category')
df = pd.get_dummies(df, columns=['function'], dtype=int)

df['country'] = df['country'].astype('category')
df = pd.get_dummies(df, columns=['country'], dtype=int)

df['continent'] = df['continent'].astype('category')
df = pd.get_dummies(df, columns=['continent'], dtype=int)

df['qualification'] = df['qualification'].astype('category')
df = pd.get_dummies(df, columns=['qualification'], dtype=int)


# Min-Max Scaling for all the features - Normalize
from sklearn.preprocessing import MinMaxScaler
num_cols = df.select_dtypes(include=['float', 'int']).columns
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])



# --------------------------- Save Feature-Representation DF to a CSV file
df.to_csv('Feature_Representation_DF.csv', index=False)








# ---------------------------- Q2 - Feature Selection & Dimensionality Reduction----------------------------------------
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, RFE, mutual_info_classif, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA

df = pd.read_csv('Feature_Representation_DF.csv')

# Split the data into input features (X) and the target variable (y)
X = df.drop("fraudulent", axis = 1)
y = df["fraudulent"]

# Feature selection using SelectKBest with ANOVA F-value scoring
skb = SelectKBest(f_classif, k = 100)
X_skb = skb.fit_transform(X, y)
skb_features = np.array(X.columns[skb.get_support()])

# Feature selection using Recursive Feature Elimination with cross-validation
clf = LogisticRegression(max_iter = 5000)
rfe = RFE(estimator = clf, n_features_to_select = 50)
X_rfe = rfe.fit_transform(X, y)
rfe_features = np.array(X.columns[rfe.get_support()])

# Feature selection using Gain Ratio
gr_features = SelectKBest(mutual_info_classif, k = 100).fit(X, y).get_support(indices = True)
gr_features = np.array(X.columns[gr_features])

# Merge all the selected features from different methods
selected_features = np.unique(np.concatenate((skb_features, rfe_features, gr_features)))

# New DF with selected features
selected_df = df.loc[:, selected_features]
selected_df["fraudulent"] = y
selected_df.to_csv('Feature_Selection_DF.csv', index=False)




# ------------------------------------- Dimensionality Reduction --------------------------------------------------------
# PCA on selected features
pca = PCA(n_components=80)
X_pca = pca.fit_transform(selected_df.iloc[:, :-1])
X_reconstructed = pca.inverse_transform(X_pca)

# Create a new DF
columns = list(X_reconstructed.columns)[:-1]
dataset = pd.DataFrame(data=X_reconstructed, columns=columns)
dataset['fraudulent'] = selected_df['fraudulent']

# Save the new dataset
dataset.to_csv('Dataset.csv', index=False)
