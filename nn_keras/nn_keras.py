# Ivan Chu
# 1001593303

import tensorflow as tf
import numpy as np
import os
import sys

def read_uci_file(pathname, labels_to_ints, ints_to_labels):
    if not(os.path.isfile(pathname)):
        print("read_data: %s not found", pathname)
        return None

    in_file = open(pathname)
    file_lines = in_file.readlines()
    in_file.close()

    rows = len(file_lines)
    if (rows == 0):
        print("read_data: zero rows in %s", pathname)
        return None
        
    
    cols = len(file_lines[0].split())
    data = np.zeros((rows, cols-1))
    labels = np.zeros((rows,1))
    for row in range(0, rows):
        line = file_lines[row].strip()
        items = line.split()
        if (len(items) != cols):
            print("read_data: Line %d, %d columns expected, %d columns found" %(row, cols, len(items)))
            return None
        for col in range(0, cols-1):
            data[row][col] = float(items[col])
        
        # the last column is a string representing the class label
        label = items[cols-1]
        if (label in labels_to_ints):
            ilabel = labels_to_ints[label]
        else:
            ilabel = len(labels_to_ints)
            labels_to_ints[label] = ilabel
            ints_to_labels[ilabel] = label
        
        labels[row] = ilabel

    labels = labels.astype(int)
    return (data, labels)
    
    
def read_uci2(directory, dataset_name):

    labels_to_ints = {}
    ints_to_labels = {}

    (train_data, train_labels) = read_uci_file(directory, labels_to_ints, ints_to_labels)
    (test_data, test_labels) = read_uci_file(dataset_name, labels_to_ints, ints_to_labels)
    return ((train_data, train_labels), (test_data, test_labels))    
    
directory = sys.argv[1]
dataset_name = sys.argv[2]
layers = int(sys.argv[3])
unit_layer = int(sys.argv[4])
rounds = int(sys.argv[5])

print ('1')


(training_set, test_set) = read_uci2(directory, dataset_name)
(training_inputs, training_labels) = training_set
(test_inputs, test_labels) = test_set
max_value = np.max(np.abs(training_inputs))
training_inputs  = training_inputs / max_value
test_inputs = test_inputs/ max_value

input_shape = training_inputs[0].shape
number_of_classes = np.max([np.max(training_labels), np.max(test_labels)]) + 1

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape = input_shape))

for x in range(layers - 2):
   model.add(tf.keras.layers.Dense(unit_layer, activation='sigmoid'))
model.add(tf.keras.layers.Dense(number_of_classes, activation='sigmoid'))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
              
model.fit(training_inputs, training_labels, epochs=rounds)

# Testing the model
test_loss, test_acc = model.evaluate(test_inputs,  test_labels, verbose=0)

classification_accuracy = 0
for x in range(len(test_inputs)):
   input_vector = test_inputs[x,:]
   input_vector = np.reshape(input_vector, (1, len(test_inputs[x])))
   nn_output = model.predict(input_vector, verbose = 0)
   nn_output = nn_output.flatten()
   predicted_class = np.argmax(nn_output)
   actual_class = test_labels[x]

   (indices,) = np.nonzero(nn_output == nn_output[predicted_class])
   number_of_ties = np.prod(indices.shape)

   if (nn_output[actual_class] == nn_output[predicted_class]):
      accuracy = 1.0 / number_of_ties
   else:
      accuracy = 0

   print('ID=%5d, predicted=%10s, true=%10s, accuracy=%4.2f\n' % (x+1, predicted_class, actual_class, accuracy))
   classification_accuracy = classification_accuracy + accuracy


classification_accuracy = classification_accuracy / len(test_labels)*100
print('classification_accuracy = %6.4f%%\n' % (classification_accuracy))
