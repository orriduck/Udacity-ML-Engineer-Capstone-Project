import argparse
import os
import sys
import pandas as pd
import numpy as np

import tensorflow as tf

from io import StringIO
from six import BytesIO

# import model
from model import SimpleNet

# accepts and returns numpy data
CONTENT_TYPE = 'application/x-npy'


def model_fn(model_dir):
    """
    This is the function that sagemaker used to load the model from a specific directory
    """
    model_path = os.path.join(model_dir, "tensorflow_mdodel/1")
    assert os.path.exists(model_path), f"{model_path} does not exists, please recheck the model path"
    print(f"Loading model from path {model_path}")
    model = tf.keras.models.load_model(os.path.join(model_dir, "tensorflow_model/1"))
    print("Successfully Load Model")
    return model

def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == CONTENT_TYPE:
        stream = BytesIO(serialized_input_data)
        return np.load(stream)
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    if accept == CONTENT_TYPE:
        buffer = BytesIO()
        np.save(buffer, prediction_output)
        return buffer.getvalue(), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)

def predict_fn(input_data, model):
    print('Predicting class labels for the input data...')
    # Process input_data so that it is ready to be sent to our model
    # convert data to numpy array then to Tensor
    # Put model into evaluation mode
    prediction = model.predict(input_data)
    # Compute the result of applying the model to the input data.
    return prediction