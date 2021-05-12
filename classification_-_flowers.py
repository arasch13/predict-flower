from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import tensorflow as tf

# ignore this:
# ignore

#
CSV_COLUMN_NAMES = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]
SPECIES = ["Setosa", "Versicolor", "Virginica"]

# load csv data as pandas dataframe
train = pd.read_csv("data/flower/iris_training.csv", names=CSV_COLUMN_NAMES, header=0)
test = pd.read_csv("data/flower/iris_test.csv", names=CSV_COLUMN_NAMES, header=0)

# pop species data and save separately
train_y = train.pop("Species")
test_y = test.pop("Species")

# create input functions
def input_function(features, labels, training=True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    if training:
        dataset = dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)
train_input_fn = input_function(train, train_y)
test_input_fn = input_function(test, test_y)

# create feature columns
feature_columns = []
for feature_name in train.keys():
    feature_columns.append(tf.feature_column.numeric_column(feature_name))

# create the classification model (deep neural network)
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[30, 10], # 2 hidden neural network layers with 30 and 10 nodes
    n_classes=3 # nr of classes the model needs to classify to
)

# train the model
classifier.train(
    input_fn=lambda: input_function(train, train_y, training=True),
    steps=5000
)
eval_result = classifier.evaluate(input_fn=lambda: input_function(test, test_y, training=False))
print(eval_result)

# predict a species based on different input features/arguments
def input_function(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)
# ask user for input features
features = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]
predict = {}
print("Please type numeric values as prompted.")
for feature in features:
    valid = True
    while valid:
        val = input(f"{feature}: ")
        if not val.isdigit():
            valid = False
    predict[feature] = [float(val)] # save each number inside a list (the tensorflow predict method needs a list as it
                                    # can handle multiple input arguments for each feature)
                                    # and save all features in a dict
# predict species
predictions = classifier.predict(input_fn=lambda: input_function(predict))
for predicted_dict in predictions:
    class_id = predicted_dict["class_ids"][0]
    probability = predicted_dict["probabilities"][class_id]

    print(f"Prediction is {SPECIES[class_id]} ({round(100*probability, 2)}%)")

