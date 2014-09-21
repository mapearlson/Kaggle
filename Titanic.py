# -*- coding: utf-8 -*-

import csv as csv 
import numpy as np
csv_file_object = csv.reader(open('/Users/michaelpearlson/Documents/Coursera/train.csv', 'rb'))
data=[]                          # Create a variable called 'data'.
for row in csv_file_object:      # Run through each row in the csv file,
    data.append(row)                 # adding each row to the data variable
data = np.array(data) 	         # Then convert from a list to an array
			         # Be aware that each item is currently
                                 # a string in this format
print data
data[0::,2].astype(np.float)     #Using this, we can calculate the proportion of
                                 #survivors on the Titanic
number_passengers = np.size(data[0::,1].astype(np.float))     #Convert from string to float to get survival rate                                                            
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_survivors = number_survived / number_passengers    
women_only_stats = data[0::,4] == "female" # This finds where all 
                                           # the elements in the gender
                                           # column that equals “female”
men_only_stats = data[0::,4] != "female"   # This finds where all the 
                                           # elements do not equal 
                                           # female (i.e. male)                             
                                           
# Using the index from above we select the females and males separately
women_onboard = data[women_only_stats,1].astype(np.float)     
men_onboard = data[men_only_stats,1].astype(np.float)

# Then we finds the proportions of them that survived
proportion_women_survived = \
                       np.sum(women_onboard) / np.size(women_onboard)  
proportion_men_survived = \
                       np.sum(men_onboard) / np.size(men_onboard) 

# and then print it out
print 'Proportion of women who survived is %s' % proportion_women_survived
print 'Proportion of men who survived is %s' % proportion_men_survived

#Reading the test data and writing the gender model as a csv
test_file = open('test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

#Open a pointer to a new file so we can write to i
prediction_file = open("gendermodel.csv", "wb")
prediction_file_object = csv.writer(prediction_file)

#want to read in the test file row by row, see if it is female or male, and write our survival prediction to a new file.
prediction_file_object.writerow(["PassengerId", "Survived"])
for row in test_file_object:                           # For each row in test.csv
    if row[3] == 'female':                             # is it a female, if yes then                                       
        prediction_file_object.writerow([row[0],'1'])  # predict 1
    else:                                              # or else if male,       
        prediction_file_object.writerow([row[0],'0'])  # predict 0
test_file.close()
prediction_file.close()