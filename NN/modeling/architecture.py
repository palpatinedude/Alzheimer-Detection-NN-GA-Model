# this file contains functions to create different models: an  neural network  or logistic regression and based on different parameters
# the function 'create_model_wrapper' allows you to choose between different models by providing the model type as input

from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD
from keras.metrics import MeanSquaredError, BinaryCrossentropy
from sklearn.linear_model import LogisticRegression
from keras.regularizers import l2
from keras.layers import Dropout

# this function wraps the model creation and allows the user to choose model type ('ann' or 'logistic')
def create_model_wrapper(model_type, input_dim, hidden_units, learning_rate, momentum,regularization, simple_metrics):
    return create_model(
        input_dim=input_dim ,
        hidden_units=hidden_units,
        learning_rate=learning_rate,
        momentum=momentum,
        model_type=model_type,  # determines which model to create
        regularization=regularization,  # regularization parameter 
        simple_metrics=simple_metrics  # whether to use simple metrics or not
    )

# this function creates the specified model based on the type (ann or logistic regression)
def create_model(input_dim=None, hidden_units=None, learning_rate=0.001, momentum=None, model_type='ann', regularization=None, simple_metrics=None):
    if model_type == 'logistic':
        model = LogisticRegression(solver='liblinear')
    else:
        input_dropout_rate = 0.3  #  Fixed dropout rate here

        layers = [
            Input(shape=(input_dim,), dtype='float32'),
            Dropout(input_dropout_rate),  # 💡 input dropout to simulate GA masking
            Dense(hidden_units, activation='relu', 
                  kernel_regularizer=l2(regularization) if regularization else None,
                  dtype='float32'),
            Dense(1, activation='sigmoid', 
                  kernel_regularizer=l2(regularization) if regularization else None,
                  dtype='float32')
        ]

        model = Sequential(layers)

        if momentum is None:
            momentum = 0.0
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)

        if simple_metrics:
            metrics = ['accuracy', MeanSquaredError(name='mse')]
        else:
            metrics = ['accuracy', MeanSquaredError(name='mse'), BinaryCrossentropy(name='ce_loss')]

        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=metrics
        )
    return model

'''
def create_model(input_dim=None, hidden_units=None, learning_rate=0.001, momentum=None, model_type='ann',regularization=None):
    if model_type == 'logistic':  # logistic regression model
        model = LogisticRegression(solver='liblinear')  # solver for logistic regression
    else:
        # artificial neural network (ANN) model
        model = Sequential([
            Input(shape=(input_dim,)),  # input layer with given dimensions
            Dense(hidden_units, activation='relu', kernel_regularizer=l2(regularization) if regularization else None),  # hidden layer with specified units and ReLU activation
            Dense(1, activation='sigmoid', kernel_regularizer=l2(regularization) if regularization else None)  # output layer with sigmoid activation for binary classification
        ])
        # If momentum is None, set it to 0 
        if momentum is None:
            momentum = 0.0

        # optimizer with learning rate and momentum
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
        model.compile(
            loss='binary_crossentropy',  # loss function for binary classification
            optimizer=optimizer,  # chosen optimizer
            metrics=['accuracy', MeanSquaredError(name='mse'), BinaryCrossentropy(name='ce_loss')]  # evaluation metrics
        )
    return model  # returns the created model

'''