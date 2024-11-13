# Markdown Cell 1
# import libraries

# Code Cell 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Code Cell 3
df = pd.read_csv("insurance.csv")

# Code Cell 4
df.head(10)

# Code Cell 5
df['sex'].value_counts()

# Code Cell 6
df['region'].value_counts()

# Code Cell 7
df['smoker'].value_counts()

# Code Cell 8
df['children'].value_counts()

# Code Cell 9
df.dtypes

# Code Cell 10
df.shape

# Code Cell 11
df.info()

# Code Cell 12
df.describe().T

# Code Cell 13
df.isnull().sum()

# Code Cell 14
from sklearn import preprocessing 
label_encoder = preprocessing.LabelEncoder() 

# Code Cell 15
# Apply LabelEncoder to categorical columns
df['sex'] = label_encoder.fit_transform(df['sex'])  # female -> 0, male -> 1
df['smoker'] = label_encoder.fit_transform(df['smoker'])  # smoker yes -> 1, no -> 0

# Code Cell 16
df.head(10)

#female -> 0, #male -> 1
#smoker yes-> 1 , no->0

# Markdown Cell 17
# EDA and Visualizations

# Code Cell 18
# Now, for visualization, convert the numeric values back to strings:
df_viz = df.copy()

df_viz['sex'] = df_viz['sex'].replace({0: 'female', 1: 'male'})
df_viz['smoker'] = df_viz['smoker'].replace({0: 'non-smoker', 1: 'smoker'})
df_viz['children'] = df_viz['children'].astype(str)  # Convert 'children' to string for visualization

# Code Cell 19
sns.boxplot(df['charges'])

# Code Cell 20
df['charges'].median()

# Code Cell 21
sns.catplot(x="smoker", kind="count", hue='sex', palette="rainbow", data=df_viz[df_viz['age'] == 18])
plt.title("The number of smokers and non-smokers (18 years old)")
plt.show()

# Markdown Cell 22
### Oh. I was hoping the result would be different. 18 years old - a very young age. Does smoking affect the cost of treatment at this age?

# Code Cell 23
plt.figure(figsize=(12,5))
plt.title("Box plot for charges for 18 years old smokers")
sns.boxplot(y="smoker", x="charges", data=df_viz[df_viz['age'] == 18], orient="h", palette="pink")
plt.show()

# Markdown Cell 24
### Oh. As we can see, even at the age of 18 smokers spend much more on treatment than non-smokers. Among non-smokers we are seeing some " tails." I can assume that this is due to serious diseases or accidents. Now let's see how the cost of treatment depends on the age of smokers and non-smokers patients.

# Code Cell 25
sns.lmplot(x="age", y="charges", hue="smoker", data=df_viz, palette="inferno_r")
plt.show()

# Markdown Cell 26
### In non-smokers, the cost of treatment increases with age. That makes sense. So take care of your health, friends! In smoking people, we do not see such dependence. I think that it is not only in smoking but also in the peculiarities of the dataset. Such a strong effect of Smoking on the cost of treatment would be more logical to judge having a set of data with a large number of records and signs. But we work with what we have! Let's pay attention to bmi. I am surprised that this figure but affects the cost of treatment in patients. Or are we on a diet for nothing?

# Code Cell 27
sns.histplot(data=df_viz, x='charges', kde=True)

# Code Cell 28
sns.set(style='whitegrid')
ax = sns.distplot(df['charges'], kde = True, color = 'c')
plt.title('Distribution of Charges')

# Markdown Cell 29


# Markdown Cell 30
##### This distribution is right-skewed. To make it closer to normal we can apply natural log


# Code Cell 31
ax = sns.distplot(np.log10(df['charges']), kde = True, color = 'r' )

# Markdown Cell 32


# Markdown Cell 33
##### Now let's look at the charges by region

0 -> NE
1 -> NW
2 -> SE
3 -> SW

# Code Cell 34
charges = df['charges'].groupby(df['region']).sum().sort_values(ascending = True)
charges = charges.head()

# Code Cell 35
sns.barplot(x=charges.index, y=charges, palette='Blues')

# Markdown Cell 36
##### So overall the highest medical charges are in the Southeast and the lowest are in the Southwest. Taking into account certain factors (sex, smoking, having children) let's see how it changes by region


# Code Cell 37
ax = sns.barplot(x='region', y='charges', hue='sex', data=df_viz, palette='cool')

# Code Cell 38
ax = sns.barplot(x='region', y='charges', hue='smoker', data=df_viz, palette='cool')

# Code Cell 39
# Ensure 'children' is treated as a categorical variable by converting to string
df['children'] = df['children'].astype(str)

# Now the plotting should work fine
ax = sns.barplot(x='region', y='charges', hue='children', data=df_viz, palette='cool')
plt.show()

# Code Cell 40
df['region']= label_encoder.fit_transform(df['region']) 

# Code Cell 41
df

# Code Cell 42
# Select only the numeric columns from the DataFrame
numeric_df = df.select_dtypes(include=[np.number])

# Generate the heatmap for the numeric columns
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.show()

# Markdown Cell 43
# Compare Between Models

# Code Cell 44
from sklearn.model_selection import train_test_split
x = df.drop(['charges'], axis = 1)
y = df['charges']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Code Cell 45
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
# from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


models = {     
    "LR": LinearRegression(),
    "RF": RandomForestRegressor(n_estimators=100, max_depth=7),
    "DT": DecisionTreeRegressor(),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, max_depth=7),
    "KNN": KNeighborsRegressor(),
    "SVR": SVR()
}

for name, model in models.items():
    print(f'Training Model {name} \n-----------------------------------------------')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print(f'Score is {model.score(x_test, y_test)}')


    # Use appropriate regression metrics
    print(f'Training R-squared: {r2_score(y_train, model.predict(x_train))}')
    print(f'Testing R-squared: {r2_score(y_test, y_pred)}')

    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
    print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')

    


# Markdown Cell 46
### The best is RandomForestRegressor

# Markdown Cell 47
# Now We Go To Choose Max Depth

# Code Cell 48
max_depth_values = [1,2,3,4,5,6,7,8]
train_accuracy_values =[]
for max_depth_val in max_depth_values:
    model = RandomForestRegressor(max_depth=max_depth_val,random_state = 2)
    model.fit(x_train, y_train)
    y_pred =model.predict(x_train)
    acc_train=model.score(x_test,y_test) 
    train_accuracy_values.append(acc_train)

# Code Cell 49
train_accuracy_values

# Code Cell 50
final_model = RandomForestRegressor(max_depth=6,random_state = 0)
final_model.fit(x_train, y_train)

# Markdown Cell 51
# Model Evalulation

# Code Cell 52
forest_train_pred = final_model.predict(x_train)
forest_test_pred = final_model.predict(x_test)

print('MSE train data: %.3f, MSE test data: %.3f' % (
mean_squared_error(y_train,forest_train_pred),
mean_squared_error(y_test,forest_test_pred)))
print('R2 train data: %.3f, R2 test data: %.3f' % (
r2_score(y_train,forest_train_pred),
r2_score(y_test,forest_test_pred)))

# Code Cell 53
plt.figure(figsize=(10,6))

plt.scatter(forest_train_pred,forest_train_pred - y_train,
          c = 'black', marker = 'o', s = 35, alpha = 0.5,
          label = 'Train data')
plt.scatter(forest_test_pred,forest_test_pred - y_test,
          c = 'c', marker = 'o', s = 35, alpha = 0.7,
          label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Tailings')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = 0, xmax = 60000, lw = 2, color = 'red')
plt.show()

# Markdown Cell 54
# Try LinearRegression

# Code Cell 55
lr = LinearRegression().fit(x_train,y_train)

y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

print(lr.score(x_test,y_test))

# Markdown Cell 56
### Linear Regression (LR) – This is a simple baseline model that assumes a linear relationship between the features and the target.
### Random Forest Regressor (RF) – This is an ensemble model that fits multiple decision trees and averages their predictions.
### Decision Tree Regressor (DT) – A single decision tree model, which may be prone to overfitting but useful for understanding splits in data.
### Gradient Boosting Regressor (GBR) – Another ensemble method that builds trees sequentially, correcting errors from the previous trees.
### K-Neighbors Regressor (KNN) – A non-parametric method that uses the K nearest data points to make predictions.
### Support Vector Regressor (SVR) – Uses support vectors and a kernel trick to fit complex, non-linear relationships.

# Code Cell 57
import nbformat

def extract_code_and_markdown_from_ipynb(ipynb_file, output_file):
    with open(ipynb_file, 'r', encoding='utf-8') as file:
        notebook = nbformat.read(file, as_version=4)
    
    with open(output_file, 'w', encoding='utf-8') as file:
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                file.write(f"# Code Cell {notebook['cells'].index(cell) + 1}\n")
                file.write(cell['source'])
                file.write('\n\n')
            elif cell['cell_type'] == 'markdown':
                file.write(f"# Markdown Cell {notebook['cells'].index(cell) + 1}\n")
                file.write(cell['source'])
                file.write('\n\n')

# replace 'notebook.ipynb' and 'output.py' with your file names
extract_code_and_markdown_from_ipynb('micp.ipynb', 'output.py')

# Code Cell 58


