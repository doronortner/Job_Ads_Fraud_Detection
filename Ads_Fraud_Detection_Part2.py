import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score, confusion_matrix, silhouette_score, davies_bouldin_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from sklearn.ensemble import VotingClassifier


# Reading the file
df = pd.read_csv("Dataset.csv")

# ----------------------------------- Validation -----------------------------------------
# Split the data into input features (X) and the target variable (y)
X = df.drop("fraudulent", axis=1)
y = df["fraudulent"]

# Splitting the data to 80% train and 20% validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=123)


# ----------------------------------- Decision Trees -----------------------------------------
DecisionTreeClassifier()

# Hyperparameters that we will tune: max_depth, criterion, min_samples_split.
decision_trees_param_grid = {
    'max_depth': np.arange(1, 51, 1),
    'criterion': ['entropy', 'gini'],
    'min_samples_split': np.arange(2, 101, 2)
}

# Random Search
trees_random_search = RandomizedSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_distributions=decision_trees_param_grid,
    cv=10,
    random_state=123,
    n_iter=500,
    refit=True
)

# Results
trees_random_search.fit(X_train, y_train)
trees_best_model = trees_random_search.best_estimator_
trees_best_model

trees_train_preds = trees_random_search.predict(X_train)
print("Training accuracy: ", round(roc_auc_score(y_train, trees_train_preds), 3))

trees_validation_preds = trees_random_search.predict(X_val)
print("Validation accuracy: ", round(roc_auc_score(y_val, trees_validation_preds), 3))

# Printing the tree
plt.figure(figsize=(12, 10))
plot_tree(trees_best_model, filled=True, class_names=True)
plt.show()

plt.figure(figsize=(12, 10))
plot_tree(trees_best_model, filled=True, class_names=True, max_depth=3)
plt.show()

# Feature importance function
feature_names = X_train.columns
feature_importance = trees_best_model.feature_importances_
feature_importance_dict = {feature_name: importance_score for feature_name, importance_score in zip(X_train.columns, feature_importance)}
sorted_feature_importances = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
print(sorted_feature_importances)



# ------------------------------------------ ANN -----------------------------------------------
# Scale the data (we did MinMax scale on part 1)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.fit_transform(X_val)

# Default ANN model
default_ANN_model = MLPClassifier()
default_ANN_model.fit(X_train_s, y_train)

print("Training accuracy: ", round(roc_auc_score(y_train, default_ANN_model.predict(X_train_s)), 3))
print("Validation accuracy: ", round(roc_auc_score(y_val, default_ANN_model.predict(X_val_s)), 3))

# Random search - ANN
ANN_param_dist = {
    'hidden_layer_sizes': [(50,), (100,), (150,), (200,), (250,), (50, 50), (100, 100), (150, 150), (200, 200),
                    (250, 250), (50, 50, 50), (100, 100, 100), (150, 150, 150), (200, 200, 200), (250, 250, 250),],
    'activation': ['relu', 'logistic'],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'max_iter': [100]
}

# Define the hyperparameter configurations
hyperparameters = []
train_scores = []
val_scores = []
num_neurons = [50, 100, 150, 200]

for num_layers in range(1, 4):
    for size_ in num_neurons:
        for activation in ['relu', 'logistic']:
            for learning_rate_init in [0.01, 0.1]:
                print(f"Num Layers: {num_layers}, Size: {size_}, Activation: {activation}, Learning Rate: {learning_rate_init}")

                # Create the model with the current hyperparameters
                hidden_layer_sizes = tuple([size_] * num_layers)
                model = MLPClassifier(
                    random_state=1,
                    hidden_layer_sizes=hidden_layer_sizes,
                    max_iter=100,
                    activation=activation,
                    verbose=False,
                    learning_rate_init=learning_rate_init,
                )

                # Fit the model and compute train and validation scores
                model.fit(X_train_s, y_train)
                train_auc = roc_auc_score(y_train, model.predict_proba(X_train_s)[:, 1])
                val_auc = roc_auc_score(y_val, model.predict_proba(X_val_s)[:, 1])

                # Store the hyperparameters and scores
                hyperparameters.append((num_layers, hidden_layer_sizes, activation, learning_rate_init))
                train_scores.append(train_auc)
                val_scores.append(val_auc)

# Create a table with the hyperparameters and scores
ANN_configurations_data = {
    'num_layers': [num for num, _, _, _ in hyperparameters],
    'hidden_layer_sizes': [sizes for _, sizes, _, _ in hyperparameters],
    'activation': [act for _, _, act, _ in hyperparameters],
    'learning_rate_init': [lr for _, _, _, lr in hyperparameters],
    'train_score': train_scores,
    'val_score': val_scores
}
ANN_configurations_df = pd.DataFrame(ANN_configurations_data)
ANN_configurations_df = ANN_configurations_df.sort_values(by='val_score', ascending=False)
print(ANN_configurations_df)

# Confusion matrix
ANN_best_model = MLPClassifier(random_state=1,
                    hidden_layer_sizes=(150, 150),
                    max_iter=100,
                    activation='relu',
                    verbose=False,
                    learning_rate_init=0.01)

ANN_best_model.fit(X_train_s, y_train)
print(confusion_matrix(y_true=y_train, y_pred=ANN_best_model.predict(X_train_s)))


# ------------------------------------------ SVM -----------------------------------------------
# hyperparameters for tuning
SVM_param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l2'],
    'loss': ['hinge', 'squared_hinge']
}

svm = LinearSVC()

# Grid Search
SVM_grid_search = GridSearchCV(svm, SVM_param_grid, cv=10, scoring='roc_auc', n_jobs=-1)
SVM_grid_search.fit(X_train_s, y_train)


print("The best configuration is: ", SVM_grid_search.best_params_)
SVM_best_model = SVM_grid_search.best_estimator_

# Predictions
SVM_train_preds = SVM_best_model.predict(X_train_s)
train_accuracy = roc_auc_score(y_train, SVM_train_preds)
print("Training accuracy:", round(train_accuracy, 3))

SVM_val_preds = SVM_best_model.predict(X_val_s)
val_accuracy = roc_auc_score(y_val, SVM_val_preds)
print("Validation accuracy:", round(val_accuracy, 3))

# Plane equation process
feature_names = df.columns.tolist()
coefficients = SVM_best_model.coef_[0]
intercept = SVM_best_model.intercept_[0]

# Print the dividing equation - 100 features dataset
contribution = [f'{coefficients[i]:.2f} * {feature_names[i]}' for i in range(len(coefficients))]
equation = ' + '.join(contribution) + f' + {intercept:.2f}'
print(f"Equation of the dividing line: y = {equation}")




# --------------------------------------- Clustering -------------------------------------------------------
# Selecting some of the features before we apply K - Medoids
threshold = 0.02
selected_features = [feature[0] for feature in sorted_feature_importances if feature[1] >= threshold and feature[0] != 'fraudulent']
selected_features

new_df = df[selected_features]
new_df

# Perform PCA to reduce the number of features to 2
pca = PCA(n_components=2)
X_pca = pca.fit_transform(new_df)

iner_list = []
dbi_list = []
sil_list = []

# Iterate over different values of k
for n_clusters in range(2, 10):
    kmedoids = KMedoids(n_clusters=n_clusters, metric='manhattan', random_state=42)
    kmedoids.fit(X_pca)
    assignment = kmedoids.labels_

    # Calculate evaluation metrics
    iner = kmedoids.inertia_
    sil = silhouette_score(X_pca, assignment)
    dbi = davies_bouldin_score(X_pca, assignment)

    dbi_list.append(dbi)
    sil_list.append(sil)
    iner_list.append(iner)


# Plot Inertia
plt.plot(range(2, 10), iner_list, marker='o')
plt.title("Inertia")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.show()

# Plot Silhouette
plt.plot(range(2, 10), sil_list, marker='o')
plt.title("Silhouette")
plt.xlabel("Number of clusters")
plt.ylabel("Silhouette score")
plt.show()

# Plot Davies-Bouldin index
plt.plot(range(2, 10), dbi_list, marker='o')
plt.title("Davies-Bouldin Index")
plt.xlabel("Number of clusters")
plt.ylabel("Davies-Bouldin index")
plt.show()

# The optimal number of clusters according to the plots is 3
kmedoids = KMedoids(n_clusters=3, random_state=42)
kmedoids.fit(X_pca)

df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['cluster'] = kmedoids.labels_

# Plot the clusters
sns.scatterplot(x='PC1', y='PC2', hue='cluster', data=df_pca, palette='Accent')
plt.scatter(pca.transform(kmedoids.cluster_centers_)[:, 0], pca.transform(kmedoids.cluster_centers_)[:, 1],
            marker='+', s=100, color='red')
plt.title("K-medoids Clustering")
plt.show()




# ------------------------------------------ Improvements -----------------------------------------------
estimator = LogisticRegression()

# Perform RFE to select top 25 features
rfe = RFE(estimator, n_features_to_select=25)
X_rfe = rfe.fit_transform(df.iloc[:, :-1], df['fraudulent'])

improved_dataset = df.iloc[:, :-1].loc[:, rfe.support_]
improved_dataset['fraudulent'] = df['fraudulent']

# Split the data into input features (X) and the target variable (y)
Improved_X = improved_dataset.drop("fraudulent", axis=1)
Improved_y = improved_dataset["fraudulent"]

# Split the data into train and validation sets
Improved_X_train, Improved_X_val, Improved_y_train, Improved_y_val = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Scale the data
scaler = StandardScaler()
Improved_X_train_s = scaler.fit_transform(Improved_X_train)
Improved_X_val_s = scaler.transform(Improved_X_val)

# Define the  classifiers
svm_classifier = LinearSVC(C = 1, loss = 'squared_hinge', penalty= 'l2', random_state = 42)
dt_classifier = DecisionTreeClassifier(criterion = 'entropy', max_depth = 42, min_samples_split = 3, random_state = 42)
mlp_classifier = MLPClassifier(hidden_layer_sizes = (200), max_iter = 100, activation = 'logistic', verbose = False,
                               learning_rate_init = 0.01, random_state = 42)

# Define the ensemble classifier
ensemble_classifier = VotingClassifier(
    estimators=[
        ('svm', svm_classifier),
        ('dt', dt_classifier),
        ('mlp', mlp_classifier)
    ],
    voting='hard'
)


ensemble_classifier.fit(Improved_X_train_s, Improved_y_train)

# Predictions - 25 Features after RFE
Improved_y_pred = ensemble_classifier.predict(Improved_X_val_s)
accuracy = roc_auc_score(Improved_y_val, Improved_y_pred)
print("Ensemble Classifier ROC Accuracy Score: {:.3f}".format(accuracy))


ensemble_classifier.fit(X_train_s, y_train)
y_pred = ensemble_classifier.predict(X_val_s)

# Predictions - 100 Features without RFE
accuracy = roc_auc_score(y_val, y_pred)
print("Ensemble Classifier ROC Accuracy Score: {:.3f}".format(accuracy))





# ------------------------------------------ Test Set -----------------------------------------------
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
from nltk.stem.porter import PorterStemmer
from collections import Counter
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import country_converter as coco


stop_words = set(stopwords.words('english'))

# -------------------------------------------- Q2 - Pre Processing ----------------------------------------------------

# Reading the file
df = pd.read_csv("X_test.csv")
df_copy = df.copy()



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
            # Get the most frequent value of column 'employment_type' as a string
            df.at[index, 'employment_type'] = df['employment_type'].dropna().value_counts().idxmax()




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
            # Get the most frequent value of column 'required_experience' as a string
            df.at[index, 'required_experience'] = df['required_experience'].dropna().value_counts().idxmax()





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
            # Get the most frequent value of column 'required_education' as a string
            df.at[index, 'required_education'] = df['required_education'].dropna().value_counts().idxmax()




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
df.to_csv('Processed_test_DF.csv', index=False)


#%%
# -------------------------------------------- Q2 - Feature Extraction ------------------------------------------------


# Reading the file
df = pd.read_csv("Processed_test_DF.csv")

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

# Compute the mean TF-IDF scores for each word in the dataset
mean_tfidf = tfidf_features.mean(axis=0).A1

# Select the top n TF-IDF features
n = 100
top_features = mean_tfidf.argsort()[-n:]
feature_names = np.array([f"title_{name}" for name in tfidf.get_feature_names_out()])[top_features]

# Add the top n TF-IDF features to the dataset
tfidf_df = pd.DataFrame(tfidf_features[:, top_features].toarray(), columns=feature_names)
df = pd.concat([df, tfidf_df], axis=1)




# ------------------------------------- Text Feature Extraction: Company Profile
# Feature extraction using TF-IDF
tfidf = TfidfVectorizer()
tfidf.fit(df['company_profile'])
tfidf_features = tfidf.transform(df['company_profile'])

# Compute the mean TF-IDF scores for each word in the dataset
mean_tfidf = tfidf_features.mean(axis=0).A1

# Select the top n TF-IDF features
n = 100
top_features = mean_tfidf.argsort()[-n:]
feature_names = np.array([f"company_{name}" for name in tfidf.get_feature_names_out()])[top_features]

# Add the top n TF-IDF features to the dataset
tfidf_df = pd.DataFrame(tfidf_features[:, top_features].toarray(), columns=feature_names)
df = pd.concat([df, tfidf_df], axis=1)




# ------------------------------------- Text Feature Extraction: Description
# Feature extraction using TF-IDF
tfidf = TfidfVectorizer()
tfidf.fit(df['description'])
tfidf_features = tfidf.transform(df['description'])

# Compute the mean TF-IDF scores for each word in the dataset
mean_tfidf = tfidf_features.mean(axis=0).A1

# Select the top n TF-IDF features
n = 100
top_features = mean_tfidf.argsort()[-n:]
feature_names = np.array([f"description_{name}" for name in tfidf.get_feature_names_out()])[top_features]

# Add the top n TF-IDF features to the dataset
tfidf_df = pd.DataFrame(tfidf_features[:, top_features].toarray(), columns=feature_names)
df = pd.concat([df, tfidf_df], axis=1)




# ------------------------------------- Text Feature Extraction: Requirements
# Feature extraction using TF-IDF
tfidf = TfidfVectorizer()
tfidf.fit(df['requirements'])
tfidf_features = tfidf.transform(df['requirements'])

# Compute the mean TF-IDF scores for each word in the dataset
mean_tfidf = tfidf_features.mean(axis=0).A1

# Select the top n TF-IDF features
n = 100
top_features = mean_tfidf.argsort()[-n:]
feature_names = np.array([f"requirements_{name}" for name in tfidf.get_feature_names_out()])[top_features]

# Add the top n TF-IDF features to the dataset
tfidf_df = pd.DataFrame(tfidf_features[:, top_features].toarray(), columns=feature_names)
df = pd.concat([df, tfidf_df], axis=1)




# ------------------------------------- Text Feature Extraction: Benefits
# Feature extraction using TF-IDF
tfidf = TfidfVectorizer()
tfidf.fit(df['benefits'])
tfidf_features = tfidf.transform(df['benefits'])

# Compute the mean TF-IDF scores for each word in the dataset
mean_tfidf = tfidf_features.mean(axis=0).A1

# Select the top n TF-IDF features
n = 100
top_features = mean_tfidf.argsort()[-n:]
feature_names = np.array([f"benefits_{name}" for name in tfidf.get_feature_names_out()])[top_features]

# Add the top n TF-IDF features to the dataset
tfidf_df = pd.DataFrame(tfidf_features[:, top_features].toarray(), columns=feature_names)
df = pd.concat([df, tfidf_df], axis=1)




# --------------------------- Save Feature-Extracted DF to a CSV file
df.to_csv('Feature_Extracted_test_DF.csv', index=False)
#%%
# -------------------------------------------- Q2 - Feature Representation --------------------------------------------



# Reading the file
df = pd.read_csv('Feature_Extracted_test_DF.csv')


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
num_cols = num_cols.drop('job_id')  # Exclude 'job_id' from scaling
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])



# --------------------------- Save Feature-Representation DF to a CSV file
df.to_csv('Feature_Representation_test_DF.csv', index=False)
#%%
# -------------- Choosing the same features from the training dataset -----------------------------------
df = pd.read_csv("Feature_Representation_test_DF.csv")
dataset_file = "Dataset.csv"
feature_df = pd.read_csv(dataset_file)

# Get the list of feature columns from the Dataset.csv file
feature_columns = feature_df.columns.tolist()
feature_columns.append('job_id')  # Add the 'job_id' feature to the list

# Filter the DataFrame to include only the desired features that exist in both DataFrames
common_columns = set(feature_columns).intersection(df.columns)
df = df[list(common_columns)]

# Save the modified DataFrame to a new file
df.to_csv('Dataset_test.csv', index=False)
#%%
# Load the training dataset
df_train = pd.read_csv("Dataset.csv")

# Load the test dataset
df_test = pd.read_csv('Dataset_test.csv')

# Extract the shared features between the two datasets
shared_features = list(set(df_train.columns).intersection(df_test.columns))

# Select the shared features in the training dataset
df_train_shared = df_train[shared_features]

# Split the training dataset into input features (X_train) and the target variable (y_train)
X_train = df_train_shared
y_train = df_train["fraudulent"]

# Scale the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train the model
ANN_best_model = MLPClassifier(random_state=1,
                               hidden_layer_sizes=(200,),
                               max_iter=100,
                               activation='logistic',
                               verbose=False,
                               learning_rate_init=0.01)

ANN_best_model.fit(X_train_scaled, y_train)

job_id = df_test['job_id']
df_test.drop('job_id', axis=1, inplace=True)

# Scale the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_test)

# Make predictions on the test data
predictions = ANN_best_model.predict(scaled_data)

# Create a DataFrame with the job IDs and predictions
result_df = pd.DataFrame({'job_id': job_id, 'fraudulent': predictions})

# Save the predictions to a CSV file
result_df.to_csv('G25_ytest.csv', index=False)