from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Lambda, GlobalMaxPooling1D, Concatenate
from tensorflow.keras.metrics import Precision, Recall, Accuracy

def SimpleNet(input_dim=403, embedding_vocab_size=100000, embedding_dim=32, sequence_size=200):
    input_layer = Input(shape=(input_dim,))

    text_x = Lambda(lambda x: x[:, : sequence_size * 2])(input_layer)
    rule_x = Lambda(lambda x: x[:, sequence_size * 2 - input_dim :])(input_layer)

    text_emb = Embedding(embedding_vocab_size, embedding_dim)(text_x)

    text_pool = GlobalMaxPooling1D()(text_emb)

    concat_x = Concatenate()([text_pool, rule_x])

    x = Dense(32, activation='relu')(concat_x)
    output_layer = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
        
    return model