import argparse
import sys
import os
import json

import pandas as pd
import tensorflow as tf

# import model
from model import SimpleNet


def model_fn(model_dir):
    """
    This is the function that sagemaker used to load the model from a specific directory
    """
    model_path = os.path.join(model_dir, "model.hdf5")
    assert os.path.exists(model_path), f"{model_path} does not exists, please recheck the model path"
    print(f"Loading model from path {model_path}")
    model = tf.kears.models.load_model(model_path)
    print("Successfully Load Model")
    return model


# Provided train function
def train(model, train_data_feature, train_data_label, validation_data, epochs=10, model_dir=None, verbose=1, batch_size=128):
    """
    This is the training method that is called by the tensorflow training script
    """
    model.summary()
    
    model_path = os.path.join(model_dir, "model.hdf5")
    print(f"Model will be saved to {model_path} ...")
    
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        model_path, monitor='val_loss', verbose=1, save_best_only=True, 
        mode='auto', save_freq='epoch', options=None
    )
    
    model.fit(
        x=train_data_feature, y=train_data_label, 
        validation_data=validation_data,
        epochs=epochs, verbose=verbose,
        shuffle=True, batch_size=batch_size,
        callbacks = [ckpt]
    )
    
def acquire_inputs(input_data_dict):
    """
    acquire the training set feature, label and validation set feature, label
    wrap them into the proper manner that fit into the model training procedure
    """
    
    train_data_path = os.path.join(input_data_dict['train'], "encoded_train_set.csv")
    validation_data_path = os.path.join(input_data_dict['validation'], "encoded_dev_set.csv")
    print("... Training Data Will be Acquired From: {}".format(train_data_path))
    print("... Validation Data Will be Acquired From: {}".format(validation_data_path))
    encoded_trainset = pd.read_csv(train_data_path, header=None)
    encoded_devset = pd.read_csv(validation_data_path, header=None)

    encoded_trainset_feature = encoded_trainset.iloc[:, 1:].values
    encoded_trainset_label = encoded_trainset.iloc[:, 0].values
    encoded_trainset = None
    print("Successfully Load Data, Shape of the features:", encoded_trainset_feature.shape)

    encoded_devset_feature = encoded_devset.iloc[:, 1:].values
    encoded_devset_label = encoded_devset.iloc[:, 0].values
    encoded_devset = None
    print("Successfully Load Data, Shape of the features:", encoded_devset_feature.shape)
    return encoded_trainset_feature, encoded_trainset_label, (encoded_devset_feature, encoded_devset_label)


## TODO: Complete the main code
if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--train_env', type=json.loads, default=os.environ['SM_TRAINING_ENV'])
    
    # Training Parameters, given
    parser.add_argument('--epochs', type=int, default=3,
                        help='number of epochs to train (default: 3)')
    parser.add_argument('--total_feature_dim', type=int, default=403,
                        help='dimension of features from input data (default: 403)')
    parser.add_argument('--embedding_dim', type=int, default=32,
                        help='output dimension of the embedding layer (default: 32)')
    parser.add_argument('--embedding_vocab_size', type=int, default=100000, metavar='H',
                        help='vocab size for the embedding layer(default: 100000)')
    parser.add_argument('--sequence_size', type=int, default=200,
                        help='the input text sequence size')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for training')
    
    args = parser.parse_args()
    
    print(f"\n============== Parsed Args ===============: \n{args}")
    
    ## TODO:  Build the model by passing in the input params
    # To get params from the parser, call args.argument_name, ex. args.epochs or ards.hidden_dim
    # Don't forget to move your model .to(device) to move to GPU , if appropriate
    model = SimpleNet(input_dim = args.total_feature_dim, 
                      embedding_vocab_size = args.embedding_vocab_size,
                      embedding_dim = args.embedding_dim,
                      sequence_size = args.sequence_size)
    
    
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])
    model.summary()
    
    # load dateset
    x_train, y_train, validation_data = acquire_inputs(args.train_env["channel_input_dirs"])

    
    # Trains the model (given line of code, which calls the above training function)
    # This function *also* saves the model state dictionary
    train(model, x_train, y_train, validation_data, args.epochs, args.model_dir, 2, args.batch_size)
    
    

