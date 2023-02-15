from keras.layers import Dense, Activation, Conv2D
from keras.layers import MaxPooling2D, Dropout, Flatten,GlobalMaxPooling2D
from keras.models import Sequential
import random
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
# from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
# from keras.layers import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory


train_dir="/home/shubhransu_etc/project_sys22/data/thermogram/train"
validation_dir=train_dir
batch_size=64
test_train_split=0.2
train_data = image_dataset_from_directory(\
      train_dir,color_mode="grayscale",image_size=(256,256) ,\
      subset='training',seed=12, validation_split=test_train_split,\
      batch_size=batch_size)
validation_data = image_dataset_from_directory(validation_dir,
      color_mode="grayscale",image_size=(256,256), subset='validation',seed=12,\
      validation_split=test_train_split,batch_size=batch_size)

def build_model(layer_dims, input_shape=(256,256,3,),len_classes=3, dropout_rate=0.2,activation='relu'):
    #print(1)
    """Function to build a model with specified layer dimensions and activation function."""
    model = Sequential()
    for i, dim in enumerate(layer_dims):
        if i == 0:
            model.add(Conv2D(dim[0], dim[1], input_shape=input_shape, activation=activation))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        else:
            model.add(Conv2D(dim[0], dim[1], activation=activation))
            model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Conv2D(filters=, kernel_size=1, strides=1))
        #model.add(Dropout(dropout_rate))
        # model.add(BatchNormalization())
        #model.add(GlobalMaxPooling2D())
        #model.add(Activation('sigmoid'))
    # model.add(Dropout(dropout_rate))
    # Output Layer
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(dropout_rate))

    model.add(Dense(len_classes-1))
    model.add(Activation('sigmoid'))
    return model

def sample_architecture(min_layers=1, max_layers=5, min_filters=32, max_filters=512, min_kernel=3, max_kernel=5):
    """Function to sample a random architecture from the search space."""
    num_layers = random.randint(min_layers, max_layers)
    layer_dims = [(random.randint(min_filters, max_filters), random.randint(min_kernel, max_kernel)) for _ in range(num_layers)]
    return layer_dims

def mutate_architecture(layer_dims, mutation_rate=0.1, min_layers=1, max_layers=5, min_filters=32, max_filters=512, min_kernel=3, max_kernel=5):
    """Function to mutate an architecture by randomly modifying some of its layer dimensions."""
    num_layers = len(layer_dims)
    for i in range(num_layers):
        if np.random.rand() < mutation_rate:
            layer_dims[i] = (random.randint(min_filters, max_filters), random.randint(min_kernel, max_kernel))
    return layer_dims

def breed_architectures(parent1, parent2, mutation_rate, min_layers, max_layers, min_filters, max_filters, min_kernel, max_kernel):
    """Function to breed two parent architectures to produce a child architecture."""
    child = []
    for i in range(len(parent1)):
        if np.random.rand() < 0.5:
            child.append(parent1[i])
        else:
            if i < len(parent2):
                child.append(parent2[i])
    return mutate_architecture(child, mutation_rate, min_layers, max_layers, min_filters, max_filters, min_kernel, max_kernel)

#Train the model on the dataset and evaluate its performance.
def train_and_evaluate_model(model,epochs=10, train_dir=None,X_train=None, y_train=None,\
                             X_test=None,y_test=None):
    """Function to train and evaluate a model."""
    if train_dir is not None:
        #train_data,validation_data=get_data(train_dir,train_dir)
        model.compile(loss="binary_crossentropy", optimizer='Adam', metrics=["BinaryAccuracy"])
        print(model.summary())
        history = model.fit(train_data, epochs=epochs, verbose=1,validation_data=validation_data)
        # return history
        print(history.history.keys())
        # Extract the accuracy from the history object
        acc = history.history['val_binary_accuracy'][len(history.history['val_binary_accuracy'])-1]
        return acc
    else:
        model.compile(loss="binary_crossentropy", optimizer='Adam', metrics=["BinaryAccuracy"])
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
        scores = model.evaluate(X_test, y_test, verbose=1)
        return scores[1]  # Return accuracy


# 4) Define the genetic algorithm to evolve the architecture.
def genetic_algorithm(train_dir,epochs,population_size=20,len_classes=3, num_generations=10, mutation_rate=0.1,\
                      min_layers=1, max_layers=5, min_filters=32, max_filters=512,\
                      min_kernel=3, max_kernel=5):
    """Function to implement the genetic algorithm to evolve the architecture."""
    # Initialize population
    population = [sample_architecture(min_layers, max_layers, min_filters,\
                                      max_filters,min_kernel, max_kernel)\
                  for _ in range(population_size)]
    scores = []
    for i in range(num_generations):
        print("Generation {}/{}".format(i + 1, num_generations))
        new_population = []
        for j in range(population_size):
            # Select two random parents
            the_choice=np.random.choice(population_size, 2, replace=False)
            parent1=population[the_choice[0]]
            parent2=population[the_choice[1]]
            # Breed the parents to produce a child
            child = breed_architectures(parent1, parent2, mutation_rate,\
                                        min_layers,max_layers, min_filters,\
                                        max_filters, min_kernel, max_kernel)
            new_population.append(child)
        # Evaluate each model in the population
        scores = []
        for j, layer_dims in enumerate(new_population):
            model = build_model(layer_dims,len_classes=len_classes,input_shape=(256,256,1,))
            # (X_train, y_train), (X_test, y_test)=tf.keras.datasets.mnist.load_data()
            score = train_and_evaluate_model(model,epochs=epochs,train_dir=train_dir)
#  X_train, y_train, X_test, y_test)
            scores.append((layer_dims, score))
        # Sort the models by accuracy and select the best ones for the next generation
        scores.sort(key=lambda x: x[1], reverse=True)
        population = [x[0] for x in scores[:population_size]]
    # Return the best architecture
    return population[0]

best_architecture2 = genetic_algorithm(train_dir=train_dir,len_classes=2,epochs=3, num_generations=10, mutation_rate=0.15,\
                      min_layers=1, max_layers=6, min_filters=32, max_filters=512,\
                      min_kernel=3, max_kernel=5)

best_model2=build_model(best_architecture2)

train_data,validation_data=get_data(train_dir,train_dir)
best_model2.compile(loss="binary_crossentropy", optimizer='Adam', metrics=["BinaryAccuracy"])

history = best_model2.fit(train_data, epochs=epochs, verbose=1,validation_data=validation_data)      

np.save('hnas_1_history.npy',history.history)
best_model2.save('hnas_stable_1_best')
