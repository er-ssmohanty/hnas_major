from keras.layers import Dense, Activation, Conv2D, BatchNormalization 
from keras.layers import MaxPooling2D, Dropout, Flatten,GlobalMaxPooling2D
from keras.models import Sequential
import random
import os
import pickle
import time
import pandas as pd
from PIL import Image
from keras.callbacks import EarlyStopping
import numpy as np
import math
import matplotlib.pyplot as plt
from keras.layers import BatchNormalization
# from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory


train_dir="/notebooks/ultrasound/train"
validation_dir=train_dir
batch_size=32
test_train_split=0.2
train_data = image_dataset_from_directory(\
      train_dir,color_mode="grayscale",image_size=(227,227) ,\
      subset='training',seed=12, validation_split=test_train_split,\
      batch_size=batch_size)
validation_data = image_dataset_from_directory(validation_dir,
      color_mode="grayscale",image_size=(227,227), subset='validation',seed=12,\
      validation_split=test_train_split,batch_size=batch_size)


def build_model(layer_dims, input_shape=(227,227,3,),len_classes=3, dropout_rate=0.2,activation='relu'):
    """Function to build a model with specified layer dimensions and activation function."""
    model = Sequential()
    print(layer_dims)
    if type(layer_dims)==type(tuple([12])):
        layer_dims=layer_dims[0]
    for i, dim in enumerate(layer_dims):
        if i == 0:
            model.add(BatchNormalization(input_shape=input_shape))
            if type(dim)==type(np.int64(1)):
                dim0=dim
                dim1=3
            else:
                dim0=dim[0]
                dim1=dim[1]
            if (type(dim0)!=type(np.int64(1))) and (type(dim0)!=type(int(1))):
                continue            
            if dim0>450:
                dim0=256
                dim1=4
                
            model.add(Conv2D(dim0, dim1,  activation=activation))
            model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
        else:
            if (type(dim0)!=type(np.int64(1))) and (type(dim0)!=type(int(1))):
                continue
            if type(dim)==type(np.int64(1)):
                dim0=dim
                dim1=3
            else:
                dim0=dim[0]
                dim1=dim[1]
            if (dim0>450) and (i==1):
                dim0=256
                dim1=4
                
            model.add(Conv2D(dim0, dim1, activation=activation))
            if i%2!=0:
                model.add(MaxPooling2D(pool_size=(2, 2),strides=2))
        #model.add(Conv2D(filters=, kernel_size=1, strides=1))
        #model.add(Dropout(dropout_rate))
        # model.add(BatchNormalization())
        #model.add(GlobalMaxPooling2D())
        #model.add(Activation('sigmoid'))
    # model.add(Dropout(dropout_rate))
    model.add(Dropout(dropout_rate))
    
    model.add(Conv2D(filters=math.ceil(math.log(len_classes,2)), kernel_size=1, strides=1))
    model.add(Dropout(dropout_rate))
    model.add(BatchNormalization())
    model.add(GlobalMaxPooling2D())
    model.add(Activation('sigmoid'))
    
    return model
                

def sample_architecture(min_layers=1, max_layers=5, min_filters=32, max_filters=512, min_kernel=3, max_kernel=5):
    """Function to sample a random architecture from the search space."""
    num_layers = np.random.randint(min_layers, max_layers+1)
    layer_dims = [(np.random.randint(min_filters, max_filters+1), np.random.randint(min_kernel, max_kernel+1)) for _ in range(num_layers)]
    return layer_dims


# def mutate_architecture(layer_dims, mutation_rate=0.1, min_layers=1, max_layers=5, min_filters=32, max_filters=512, min_kernel=3, max_kernel=5):
#     """Function to mutate an architecture by randomly modifying some of its layer dimensions."""
#     num_layers = len(layer_dims)
#     for i in range(num_layers):
#         if np.random.rand() < mutation_rate:
#             layer_dims[i] = (random.randint(min_filters, max_filters), random.randint(min_kernel, max_kernel))
#     return layer_dims

def mutate_architecture(layer_dims, mutation_rate=0.1, min_layers=1, max_layers=5, min_filters=32, max_filters=512, min_kernel=3, max_kernel=5):
    """Function to mutate an architecture by randomly modifying some of its layer dimensions."""
    num_layers = len(layer_dims)
    
    # Randomly add or remove layers
    if np.random.rand() < mutation_rate:
        if num_layers == max_layers:
            # Remove a layer if maximum number of layers is already reached
            layer_dims.pop(random.randint(0, num_layers-1))
        elif num_layers == min_layers:
            # Add a layer if minimum number of layers is already reached
            layer_dims.insert(random.randint(0, num_layers), (random.randint(min_filters, max_filters), random.randint(min_kernel, max_kernel)))
        else:
            # Add or remove a layer randomly
            if np.random.rand() < 0.5:
                layer_dims.pop(random.randint(0, num_layers-1))
            else:
                layer_dims.insert(random.randint(0, num_layers), (random.randint(min_filters, max_filters), random.randint(min_kernel, max_kernel)))
    num_layers = len(layer_dims)
    # Randomly modify existing layer dimensions
    for i in range(num_layers):
        if np.random.rand() < mutation_rate:
            layer_dims[i] = (random.randint(min_filters, max_filters), random.randint(min_kernel, max_kernel))
    
    return layer_dims



def breed_architectures(parent1, parent2, mutation_rate, min_layers, max_layers, min_filters, max_filters, min_kernel, max_kernel):
    """Function to breed two parent architectures to produce a child architecture."""
    child = []
    q=np.int64(12.32)
    if type(parent1)!=type(q):
        len_par1=len(parent1)
    else:
        len_par1=1
    for i in range(len_par1):
        if np.random.rand() < 0.5:
            try:
                parent1_i=parent1[i]
            except IndexError:
                parent1_i=parent1
            child.append(parent1_i)
        else:
            if type(parent2)!=type(q):
                len_parent2=len(parent2)
            else:
                len_parent2=1
            if i < len_parent2:
                try:
                    parent2_i=parent2[i]
                except IndexError:
                    parent2_i=parent2
                child.append(parent2_i)
    return mutate_architecture(child, mutation_rate, min_layers, max_layers, min_filters, max_filters, min_kernel, max_kernel)


#Train the model on the dataset and evaluate its performance.
def train_and_evaluate_model(model,epochs=10, train_dir=None,X_train=None, y_train=None,\
                             X_test=None,y_test=None):
    """Function to train and evaluate a model."""
    if train_dir is not None:
        #train_data,validation_data=get_data(train_dir,train_dir)
        model.compile(loss="binary_crossentropy", optimizer='Adam', metrics=["BinaryAccuracy"])
        # print(model.summary())
        callback = EarlyStopping(monitor='val_loss', patience=2)

        history = model.fit(train_data, epochs=epochs, callbacks=[callback],verbose=1,validation_data=validation_data)
        # return history
        # print(history.history.keys())
        # Extract the accuracy from the history object
        acc = history.history['val_binary_accuracy'][len(history.history['val_binary_accuracy'])-1]
        return acc
    else:
        model.compile(loss="binary_crossentropy", optimizer='Adam', metrics=["BinaryAccuracy"])
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
        scores = model.evaluate(X_test, y_test, verbose=1)
        return scores[1]  # Return accuracy
    

def genetic_algorithm(train_dir, epochs, population_size=20, len_classes=3, num_generations=10, mutation_rate=0.1,\
                      min_layers=1, max_layers=5, min_filters=32, max_filters=512,\
                      min_kernel=3, max_kernel=5, save_after=10, checkpoint_file=None):
    """Function to implement the genetic algorithm to evolve the architecture."""
    if checkpoint_file and os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
            population = data['population']
            scores = data['scores']
            start_generation = data['generation']
            start_population = data['population_id']
    else:
        # Initialize population
        population = [sample_architecture(min_layers, max_layers, min_filters, max_filters, min_kernel, max_kernel) for _ in range(population_size)]
        scores = []
        start_generation = 0
        start_population = 0

    for generation in range(start_generation, num_generations):
        print('Generation', generation+1)

        # Evaluate population
        print('Evaluating population...')
        population = list(map(list, set(map(frozenset, population))))
        score_population = []
        for i, arch in enumerate(population):
            print(type(population))
            print(f'Evaluating architecture {i+1}/{len(population)}')
            if i>12:
                time.sleep(58)
            model = build_model(arch, input_shape=(227,227,1,), len_classes=len_classes)
            score = train_and_evaluate_model(model, epochs, train_dir=train_dir)
            score_population.append(score)

        # Select top 20% parents
        num_parents = int(len(population) * 0.2)
        parent_indices = np.argsort(score_population)[-num_parents:]
        parents = [population[i] for i in parent_indices]

        # Breed new generation
        print('Breeding new generation...')
        new_population = parents.copy()
        while len(new_population) < population_size:
            parent1, parent2 = np.random.choice(np.ravel(parents), 2)
            child = breed_architectures(parent1, parent2, mutation_rate, min_layers, max_layers, min_filters, max_filters, min_kernel, max_kernel)
            new_population.append(child)

        # Replace population with new generation
        population = new_population
        scores = [score_population[i] for i in parent_indices] + [None]*(population_size-num_parents)

        # Save checkpoint
        if checkpoint_file:
            data = {'population': population, 'scores': scores, 'generation': generation+1, 'population_id': start_population}
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(data, f)

    # Evaluate final population
    print('Evaluating final population...')
    score_population = []
    population = list(map(list, set(map(frozenset, population))))
    for i, arch in enumerate(population):
        print(f'Evaluating architecture {i+1}/{len(population)}')
        model = build_model(arch, input_shape=(227,227,1,), len_classes=len_classes)
        score = train_and_evaluate_model(model, epochs, train_dir=train_dir)
        score_population.append(score)

    # Select best architecture
    best_arch_index = np.argmax(score_population)
    best_arch = population[best_arch_index]
    best_score = score_population[best_arch_index]

    return best_arch, best_score


best_architecture2 = genetic_algorithm(train_dir=train_dir,len_classes=2,epochs=10,population_size=20, num_generations=40, mutation_rate=0.5,\
                      min_layers=3, max_layers=7, min_filters=32, max_filters=512,\
                      min_kernel=3, max_kernel=5, checkpoint_file='/notebooks/hnas_major/models/checkpoint_file_ultrasound.pkl')

epochs=100
callback = EarlyStopping(monitor='val_loss', patience=3)

print(best_architecture2)
best_model2=build_model(best_architecture2,input_shape=(227,227,1,), len_classes=2)
best_model2.compile(loss="binary_crossentropy", optimizer='Adam', metrics=["BinaryAccuracy"])


historyX=best_model2.fit(train_data, epochs=epochs, verbose=1,validation_data=validation_data, callbacks=[callback])

with open('history_best_thermogram_base_hnas', 'wb') as file_pi:
    pickle.dump(historyX.history, file_pi)

best_model2.save("/notebooks/hnas_major/models/hnas_ultrasound_fcn_0")
