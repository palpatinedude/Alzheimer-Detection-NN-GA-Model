from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.metrics import MeanSquaredError, BinaryCrossentropy

def create_model_bonus(input_dim, hidden_units, learning_rate, momentum, regularization): 
    model = Sequential()
    model.add(Input(shape=(input_dim,)))  # Input layer
    
    # Add hidden layers
    for units in hidden_units:
        model.add(Dense(units, activation='relu', kernel_regularizer=l2(regularization)))
    
    # Output layer (sigmoid for binary classification)
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(regularization)))
    
    # Compile the model
    optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', MeanSquaredError(name='mse'), BinaryCrossentropy(name='ce_loss')])
    
    return model