import csv as csv
import numpy as np

# open up the csv file in to a python object
csv_file_object = csv.reader(open('/home/hkh/Downloads/train.csv', 'rb'))
header = csv_file_object.next();

rawdata = []
for row in csv_file_object:
	rawdata.append(row)

data = np.array(rawdata)

number_passengers = np.size(data[0::,1].astype(np.float))
number_survived = np.sum(data[0::,1].astype(np.float))
proportion_surviviors = number_survived / number_passengers

women_only_stats = data[0::,4] == "female"
men_only_stats = data[0::,4] != "female"

women_onboard = data[women_only_stats, 1].astype(np.float)
men_onboard = data[men_only_stats, 1].astype(np.float)

proportion_women_survived = np.sum(women_onboard) / np.size(women_onboard)
proportion_men_survived = np.sum(men_onboard) / np.size(men_onboard)

# and then print it out
print 'Proportion of women who survived is %s' % proportion_women_survived
print 'Proportion of men who survived is %s' % proportion_men_survived
