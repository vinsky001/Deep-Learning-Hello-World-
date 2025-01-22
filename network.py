from keras.callbacks import ModelCheckpoint
from keras.layers import (
    Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
)
from keras.models import Sequential, load_model
from load_data import x_train, y_train, x_test, y_test

#createtheNetwork

def create_model():
    """
    create the model
    """
    model = Sequential()
    model.add(Conv2D(filters=10, kernel_size=(4, 4), input_shape=(28,28, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=40, kernel_size=(4, 4), padding='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(100), Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10), Activation('softmax'))
    return model

initial_model = create_model()

#Compile the model
#converts the created model into a sequence of matrix operations enabling quicker computation by TensorFlow

def compile_model(model):
    """
    Compile the model
    """
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['categorical_accuracy']
    )
    return model

compile_model = compile_model(initial_model)

#fit the model
#define a checkpoint to store the best model at the current time. It is stored in the /mnist/models/ directory

def fit_model(model):
    """
    train the model while storing the best weights
    """
    checkpoint = ModelCheckpoint(
        filepath = 'models/mnist.h5',
        monitor = 'categorical_accuracy',
        save_best_only = True,
        mode = 'max')
    model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=100,
        validation_splits=(0.25),
        callbacks=[checkpoint]
    )
    return model

print('\nTraining the model\n')
trained_model = fit_model(compile_model)
        
#evaluate the model
#we check how well model we’ve trained performed.

def evaluate_model():
    """
    Evaluate the model
    """
    best_model = load_model('models/mnist.h5')
    test_score = best_model.evaluate(x_test, y_test)
    train_score = best_model.evaluate(x_train, y_train)
    return test_score, train_score


print('\nEvaluating the model\n')
test_score, train_score = evaluate_model() 

print('\nPercentage predicted correctly:\n')   
print('\nTraining:', train_score[1] * 100, '%')
print('\nTesting:', test_score[1] * 100, '%')
    
    
    
    