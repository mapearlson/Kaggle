import csv as csv
import numpy as np

csv_file_object = csv.reader(open('train.csv', 'rb')) 
header = csv_file_object.next() 
data=[] 

for row in csv_file_object:
    data.append(row)
data = np.array(data) 
print data
print data[0]
print data[-1]
print data[0,3]
print data[0::,3]

number_passengers = np.size(data[0::,1].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers

data[0:15,5]
type(data[0::,5])       #So, any slice we take from the data is still a Numpy array.
ages_onboard = data[0::,5].astype(np.float) #produced an error when numpy got to the missing value ' ' in the 6th row. So let's try again with Pandas.

import pandas as pd
import numpy as np

# For .read_csv, always use header=0 when you know row 0 is the header row
df = pd.read_csv('train.csv', header=0)
df.head()
df.tail()
df.head(3)
type(df)
# pandas.core.frame.DataFrame.  Recall that using the csv package before, every value was interpreted as a string. But how does Pandas interpret them using its own csv reader?
df.dtypes   #Pandas is able to infer numerical types whenever it can detect them. So we have values already stored as integers. When it detected the existing decimal points somewhere in Age and Fare, it converted those columns to float. 
df.info()   #There's a lot of useful info there! You can see immediately we have 891 entries (rows), and for most of the variables we have complete values (891 are non-null). But not for Age, or Cabin, or Embarked -- those have nulls somewhere.
df.describe() #This is also very useful: pandas has taken all of the numerical columns and quickly calculated the mean, std, minimum and maximum value. Convenient! But also a word of caution: we know there are a lot of missing values in Age, 
              #for example. How did pandas deal with that? It must have left out any nulls from the calculation. So if we start quoting the "average age on the Titanic" we need to caveat how we derived that number.

#Data Munging
#Let's acquire the first 10 rows of the Age column. In pandas this is
df['Age'][0:10]
df.Age[0:10]
df.Cabin[0:10]
type(df['Age'])     #A single column is neither an numpy array, nor a pandas dataframe -- but rather a pandas-specific object called a data Series.
df['Age'].mean()
df['Age'].median()
df[ ['Sex', 'Pclass', 'Age'] ].head()   #The next thing we'd like to do is look at more specific subsets of the dataframe. Again pandas makes this very convenient to write. Pass it a [ list ] of the columns desired
df[df['Age'] > 60]  #The .describe() command had indicated that the maximum age was 80. What do the older passengers look like in this data set? This is written by passing the criteria of df as a where clause into df
df[df['Age'] > 60][['Sex', 'Pclass', 'Age', 'Survived']] #If you were most interested in the mix of the gender and Passenger class of these older people, you would want to combine the two skills you just learned and get only a few columns for the same where filter
for i in range(1,4):
    print i, len(df[ (df['Sex'] == 'male') & (df['Pclass'] == i) ])     #It will also be useful to combine multiple criteria (joined by the syntax &). To practice even more functionality in the same line of code, let's take a count of the males in each class.

#Before we finish the initial investigation by hand, let's use one other convenience 
#function of pandas to derive a histogram of any numerical column. The histogram function 
#is really a shortcut to the more powerful features of the matplotlib/pylab packages, so let's be sure that's imported. Type the following:
import pylab as P
df['Age'].hist()
P.show()

#Inside the parentheses of .hist(), you can also be more explicit about options of this 
#function. Before you invoke it, you can also be explicit that you are dropping the missing values of Age:
df['Age'].dropna().hist(bins=16, range=(0,80), alpha = .5)
P.show()

#Cleaning the data
#First of all, it's hard to run analysis on the string values of "male" and "female". Let's practice transforming it in three ways -- twice for fun and once to make it useful. We'll store our transformation in a new column, so the original Sex isn't changed.
df['Gender'] = 4
df['Gender'] = df['Sex'].map( lambda x: x[0].upper() )  #Iambda x is an built-in function of python for generating an anonymous function in the moment, at runtime. Remember that x[0] of any string returns its first character.

#But of course what we really need is a binary integer for female and male, similar to the way Survived is 
#stored. As a matter of consistency, let's also make Gender into values of 0 and 1's. We have a 
#precedent of analyzing the women first in all of our previous arrays, so let's decide female = 0 and male = 1.  So, for real this time:
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
df['Origin'] = df['Embarked'].dropna().map( {'S': 3, 'C': 1, 'Q':2} ).astype(int)

#Now it's time to deal with the missing values of Age, because most machine learning will need a complete set of values 
#in that column to use it. By filling it in with guesses, we'll be introducing some noise into a model, but if we can 
#keep our guesses reasonable, some of them should be close to the historical truth (whatever it was...), and the 
#overall predictive power of Age might still make a better model than before.  We know the average [known] age of all 
#passengers is 29.6991176 -- we could fill in the null values with that. But maybe the median would be better? 
#(to reduce the influence of a few rare 70- and 80-year olds?) The Age histogram did seem positively skewed. These are the kind of decisions you make as you create your models in a Kaggle competition.
#For now let's decide to be more sophisticated, that we want to use the age that was typical in each passenger class. 
#And decide that the median might be better. Let's build another reference table to calculate what each of these medians are:

median_ages = np.zeros((2,3))
#array([[ 0.,  0.,  0.],
       #[ 0.,  0.,  0.]])
for i in range(0, 2):
    for j in range(0, 3):
        median_ages[i,j] = df[(df['Gender'] == i) & (df['Pclass'] == j+1)]['Age'].dropna().median()
median_ages
#array([[ 35. ,  28. ,  21.5],
       #[ 40. ,  30. ,  25. ]])

#We could fill in the missing ages directly into the Age column. But to be extra cautious and not lose the state of the original data, a more formal way would be to create a new column, 
#AgeFill, and even record which ones were originally null (and thus artificially guessed).
df['AgeFill'] = df['Age']
df.head()
df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)

#Use some code to fill in AgeFill based on our median_ages table. Here we happen to use 
#the alternate syntax for referring to an existing column, like df.Age rather than df['Age'].  
#There's a where clause on df and referencing its column AgeFill, then assigning it an appropriate value out of median_ages.
for i in range(0, 2):
    for j in range(0, 3):
        df.loc[ (df.Age.isnull()) & (df.Gender == i) & (df.Pclass == j+1),'AgeFill'] = median_ages[i,j]
df[ df['Age'].isnull() ][['Gender','Pclass','Age','AgeFill']].head(10)

#Let's also create a feature that records whether the Age was originally missing. This is relatively simple by allowing pandas to use the integer conversion of the True/False evaluation of its built-in function, pandas.isnull()
df['AgeIsNull'] = pd.isnull(df.Age).astype(int)
df.describe()

#Feature Engineering
#Let's create a couple of other features, this time using simple math on existing columns. Since we know that Parch is the number of parents or children onboard, and SibSp is the number of siblings or spouses, we could collect those together as a FamilySize:
df['FamilySize'] = df['SibSp'] + df['Parch']
df['Age*Class'] = df.AgeFill * df.Pclass
df['FamilySize'].hist()
df['Age*Class'].hist()

#We have our data almost ready for machine learning. But most basic ML techniques will not work 
#on strings, and in python they almost always require the data to be an array-- the implementations 
#we will see in the sklearn package are not written to use a pandas dataframe. So the last two things 
#we need to do are (1) determine what columns we have left which are not numeric, and (2) send our pandas.DataFrame back to a numpy.array.
df.dtypes
df.dtypes[df.dtypes.map(lambda x: x=='object')]         #With a little manipulation, we can require .dtypes to show only the columns which are 'object', which for pandas means it has strings:
df = df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Age'], axis=1) 
df = df.dropna()

#The final step is to convert it into a Numpy array. Pandas can always send back an array using the .values method. Assign to a new variable, train_data:
train_data = df.values
train_data
df[ df['Origin'].isnull() ][['Origin', 'Gender','Pclass','AgeFill']].head(10)
df['Origin'].fillna(3)

# Import the random forest package
from sklearn.ensemble import RandomForestClassifier

# Create the random forest object which will include all the parameters
# for the fit
forest = RandomForestClassifier(n_estimators = 100)

# Fit the training data to the Survived labels and create the decision trees
forest = forest.fit(train_data[0::,1::],train_data[0::,0])

# Take the same decision trees and run it on the test data
output = forest.predict(test_data)