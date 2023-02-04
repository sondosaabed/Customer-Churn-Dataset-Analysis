# Sondos Aabed, 1190652
# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, roc_auc_score, confusion_matrix, precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# Load the csv file into a data Structure
dataframe = pd.read_csv('Customer Churn.csv')
# check the size of obsevations and the attributes size
print(len(dataframe)) 
print(dataframe.columns.size)

# Part one: 7 General Tasks
# 1- Show the summary statics for all attributes
# For the quantitative attributes used describe method
print(dataframe.describe())
# Plot the histograms (univariate)
figure, axes = plt.subplots(figsize=(20, 20))
dataframe.hist(ax=axes,color='green')
# for the qualititative attributes Plot pie charts to describe them
# Pie chart for Age group
figure1, axes1 = plt.subplots()
axes1.pie(dataframe['Age Group'].value_counts(), labels=dataframe['Age Group'].unique(), autopct='%1.1f%%')
axes1.set_title('Age Group Distribution')
# Pie chart for Status
figure4, axes4 = plt.subplots()
axes4.pie(dataframe['Status'].value_counts(), labels=dataframe['Status'].unique(), autopct='%1.1f%%')
axes4.set_title('Status Distribution')
# Pie chart for Plan
figure5, axes5 = plt.subplots()
axes5.pie(dataframe['Plan'].value_counts(), labels=dataframe['Plan'].unique(), autopct='%1.1f%%')
axes5.set_title('Plan Distribution')
# Pie chart for Complains
figure6, axes6 = plt.subplots()
axes6.pie(dataframe['Complains'].value_counts(), labels=dataframe['Complains'].unique(), autopct='%1.1f%%')
axes6.set_title('Complains Distribution')

# Task 2
# Pie chart for Churn
figure3, axes3 = plt.subplots()
axes3.pie(dataframe['Churn'].value_counts(), labels=dataframe['Churn'].unique(), autopct='%1.1f%%')
axes3.set_title('Churn Distribution')

# Task 3
# For each age group, draw a histogram detailing the amount of churn in each sub-group
# Group the data by age group and count the number of churn and non-churn customers
age_group_churn = dataframe.groupby(['Age Group', 'Churn']).size().reset_index(name='count')
# Plot the histogram
figure13, axes13 = plt.subplots()
axes13.hist(age_group_churn['Age Group'], weights=age_group_churn['count'], stacked=True, color='orange')
axes13.set_xlabel('Age group')
axes13.set_ylabel('Number of Customers')
axes13.set_title('Churn distribution by Age group')

# Task 4
# For each charge amount, draw a histogram detailing the amount of churn in each sub-group.
age_group_charge_amount = dataframe.groupby(['Charge Amount', 'Age Group']).size().reset_index(name='count1')
# Plot the histogram
figure13b, axes13b = plt.subplots()
axes13b.hist(age_group_charge_amount['Charge Amount'], weights=age_group_charge_amount['count1'], stacked=True, color='green')
axes13b.set_xlabel('Charge Amount')
axes13b.set_ylabel('Number of Customers')
axes13b.set_title('Age Group distribution by Charge Amount')

# Task 5
# Show the details of the charge amount of customers.
print(dataframe['Charge Amount'].describe())
column3 ='Charge Amount'
# Count the values in the column
counts = dataframe[column3].value_counts()
x_labels = counts.index
y_values = counts.values
figure2, axes2 = plt.subplots()
axes2.bar(x_labels, y_values,color='green')
axes2.set_xlabel(column3)
axes2.set_ylabel('Count')
axes2.set_title('Charge Amount Distribution')
print(dataframe.corr())

# Task 6
# Visualize the Correlaion between all the features
# Print the corralation
print(dataframe.corr())
# Convert Qualititative attributes to Quantititaive to use heat Map and to Standarize dataFrame
dataframe['Churn'] =dataframe['Churn'].astype('category').cat.codes
dataframe['Status'] =dataframe['Status'].astype('category').cat.codes
dataframe['Plan'] =dataframe['Plan'].astype('category').cat.codes
dataframe['Complains'] =dataframe['Complains'].astype('category').cat.codes
dataframe['Charge Amount'] =dataframe['Charge Amount'].astype('category').cat.codes
dataframe['Age Group'] =dataframe['Age Group'].astype('category').cat.codes
figure12, axes12= plt.subplots()
sns.set(rc = {'figure.figsize':(16,8)})
sns.heatmap(dataframe.corr(), ax=axes12,annot = True, fmt='.2g',cmap= 'vlag')
# I chose to drop these after interpreting the Correlation as specified in the document
dataframe.drop(['ID','Age Group'], axis=1)
# After dropping these comes the Data Cleansing part
# For data Cleansing I chose to fill in the missing data with the mean of their coulmn, and i chose to remove duplicates
dataframe.fillna(dataframe.mean())
dataframe.drop_duplicates()

# Task 7
# Split the dataset into training (80%) and test (20%), asuming the target is Churn
# will be used in classification tasks
x_train, x_test, y_train, y_test = train_test_split(dataframe.drop('Churn', axis=1), dataframe['Churn'], test_size=0.2)

# Part two: Linear Regression
# Task 1: linear regression to learn the attribute “Customer Value” using all independent attributes  
# since we are learning the Customer Value drop it
dataframe1= dataframe
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(dataframe1.drop('Customer Value', axis=1), dataframe1['Customer Value'], test_size=0.2)
LRM1 = LinearRegression()
LRM1.fit(X_train_1,y_train_1)
y_pred_1=LRM1.predict(X_test_1)
print(LRM1.score(X_test_1,y_test_1))
# Calculate teh mean squred error
mse = mean_squared_error(y_test_1, y_pred_1)
print(f'Mean Squared Error For LRM1: {mse:.2f}')
# Plot the Relation
# residuals = y_test_1 - y_pred_1
figure14, axes14 = plt.subplots()
axes14.scatter(y_test_1, y_pred_1, color='green')
axes14.plot([y_test_1.min(), y_test_1.max()], [y_test_1.min(), y_test_1.max()])
plt.title('LRM1 predicting Customer Value')
axes14.set_xlabel('Actual Customer values')
axes14.set_ylabel('Predicted Customer values')
axes14.grid(visible=False)

# Task 2: Linear regression using the most three attributes
# 3 important features: Frequency of SMS, Subscription  Length and Freq. of use
dataframe2= dataframe
x = dataframe2[['Freq. of SMS', 'Subscription  Length','Freq. of use']]
y = dataframe2['Customer Value']
# split the data into train and test
x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x, y, test_size=0.2)
LRM2 = LinearRegression()
LRM2.fit(x_train_2,y_train_2)
y_pred_2=LRM2.predict(x_test_2)
print(LRM2.score(x_test_2,y_test_2))
# Calcutae the mean square error
mse2 = mean_squared_error(y_test_2, y_pred_2)
print(f'Mean Squared Error For LRM2: {mse2:.2f}')
# plot the relation between the actual values and the predicted values
figure15, axes15 = plt.subplots()
axes15.scatter(y_test_2, y_pred_2, color='green')
# add a diagonal line to the plot
axes15.plot([y_test_2.min(), y_test_2.max()], [y_test_2.min(), y_test_2.max()])
axes15.set_xlabel('Actual Customer values')
axes15.set_ylabel('Predicted Customer values')
axes15.grid(visible=False)
plt.title('LRM2 predicting Customer Value')

# Task 3: Linear regression using the most set of most important attributes
# use dataframeC 
dataframe3= dataframe
x1 = dataframe3[['Freq. of SMS', 'Seconds of Use','Freq. of use']]
y1 = dataframe3['Customer Value']
# split into train and test
X_train_3, X_test_3, y_train_3, y_test_3 = train_test_split(x1, y1, test_size=0.2)
LRM3 = LinearRegression()
LRM3.fit(X_train_3,y_train_3)
y_pred_3=LRM3.predict(X_test_3)
print(LRM3.score(X_test_3,y_test_3))
# calculate the mean squared error
mse3 = mean_squared_error(y_test_3, y_pred_3)
print(f'Mean Squared Error For LRM3: {mse3:.2f}')
# Plot the Relation
# residuals = y_test_3 - y_pred_3
figure16, axes16 = plt.subplots()
axes16.scatter(y_test_3, y_pred_3, color='green')
axes16.plot([y_test_3.min(), y_test_3.max()], [y_test_3.min(), y_test_3.max()])
axes16.set_title('LRM3 predicting Customer Value')
axes16.set_xlabel('Actual Customer values')
axes16.set_ylabel('Predicted Customer values')
axes16.grid(visible=False)

# For each axses show the plot
plt.show()

# Part threee: Classsification
dataframe4 = dataframe
dataframe4.drop(['ID', 'Plan', 'Status','Age Group'], axis=1)
dataframe4.fillna(dataframe4.mean())
dataframe4.drop_duplicates()
# split into train and test for each model
x_train_4, x_test_4, y_train_4, y_test_4 = train_test_split(dataframe4.drop('Churn', axis=1), dataframe4['Churn'], test_size=0.2)
x_train_5 = x_train_6 = x_train_4
x_test_5 = x_test_6 = x_test_4
y_train_5 = y_train_6 = y_train_4
y_test_6 = y_test_5 = y_test_4

# Task 1: k-Nearest Neighbors classifier to predict Churn
# Create the k-NN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_4, y_train_4)
y_pred_knn=knn.predict(x_test_4)
#get the confusion matrix for KNN
confusion_matrix_knn = confusion_matrix(y_test_4, y_pred_knn)
print(confusion_matrix_knn)
# Obtain the predicted probabilities for the positive class
y_pred_proba = knn.predict_proba(x_test_4)[:, 1]
# Calculate the ROC/AUC score
roc_auc_knn = roc_auc_score(y_test_4, y_pred_proba)
print('ROC/AUC KNN: ',roc_auc_knn)
# Calculate presion
precision_knn = precision_score(y_test_4, y_pred_knn)
print('KNN precsion: ', precision_knn)

# Task 2: Naive Bayes classifier to predict Churn 
# Create the NB classifier
nv = GaussianNB()
nv.fit(x_train_5,y_train_5)
y_pred_nv=nv.predict(x_test_5)
#get the confusion matrix for NB
confusion_matrix_nv = confusion_matrix(y_test_5, y_pred_nv)
print(confusion_matrix_nv)
# Obtain the predicted probabilities for the positive class
y_pred_proba = nv.predict_proba(x_test_5)[:, 1]
# Calculate the ROC/AUC score
roc_auc_nv = roc_auc_score(y_test_5, y_pred_proba)
print('ROC/AUC NV: ',roc_auc_nv)
# Calculate presion
precision_nv= precision_score(y_test_5, y_pred_nv)
print('NV precsion: ', precision_nv)

# Task 3: Logistic Regression classifier to predict Churn
# create model
lr = LogisticRegression()
lr.fit(x_train_6, y_train_6)
y_pred_lr = lr.predict(x_test_6)
#get the confusion matrix for NB
confusion_matrix_lr = confusion_matrix(y_test_6, y_pred_lr)
print(confusion_matrix_lr)
# Obtain the predicted probabilities for the positive class
y_pred_proba = lr.predict_proba(x_test_6)[:, 1]
# Calculate the ROC/AUC score
roc_auc_lr = roc_auc_score(y_test_6, y_pred_proba)
print('ROC/AUC LR: ',roc_auc_lr)
# Calculate presion
precision_lr = precision_score(y_test_6, y_pred_lr)
print('LR precsion: ', precision_lr)