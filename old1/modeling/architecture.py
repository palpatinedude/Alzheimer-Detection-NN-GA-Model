from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import SGD
from keras.metrics import MeanSquaredError, BinaryCrossentropy
from sklearn.linear_model import LogisticRegression

def create_model(input_dim=None, hidden_units=None, learning_rate=0.001, momentum=0.2,model_type='ann'):
    if model_type == 'logistic':
        model = LogisticRegression(solver='liblinear')
    else:
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(hidden_units, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
        model.compile(
            loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy', MeanSquaredError(name='mse'), BinaryCrossentropy(name='ce_loss')]
        )
    return model
