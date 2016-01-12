# Import the random forest package
from sklearn.ensemble import RandomForestClassifier

# Create the random forset object which will include all the parameters for the fit
forest = RandomForestClassifier(n_estimators = 50)

# Fit the training data to the Supervised labels and create the decision trees
forest = forest.fit(train_data[0::, 1::], train_data[0::, 0])

# Take the same decision trees and run it on the test data
output = forest.predict(test_data).astype(int)

predictions_file = open("/home/hkh/sources/kagglepy/titanic/output/myfirstforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'