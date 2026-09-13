import tensorflow as tf
from tensorflow.keras import layers, Model

def create_cf_model(num_users, num_items, embedding_dim=32):
    """
    Create a Matrix Factorization (Neural Collaborative Filtering) model.
    We use standard Keras layers so that we can easily integrate with TFF.
    """
    user_input = layers.Input(shape=(1,), name='user_id', dtype=tf.int32)
    item_input = layers.Input(shape=(1,), name='movie_id', dtype=tf.int32)

    # User embedding
    user_embedding = layers.Embedding(
        input_dim=num_users, output_dim=embedding_dim, name='user_embedding'
    )(user_input)
    user_vec = layers.Flatten(name='flatten_users')(user_embedding)

    # Item embedding
    item_embedding = layers.Embedding(
        input_dim=num_items, output_dim=embedding_dim, name='item_embedding'
    )(item_input)
    item_vec = layers.Flatten(name='flatten_items')(item_embedding)

    # Dot product
    dot = layers.Dot(axes=1, name='dot_product')([user_vec, item_vec])
    
    # We can also add LLM profile embeddings later as a separate feature vector
    
    model = Model(inputs=[user_input, item_input], outputs=dot)
    
    return model

def get_compiled_model(num_users, num_items, embedding_dim=32, learning_rate=0.01):
    model = create_cf_model(num_users, num_items, embedding_dim)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )
    return model

if __name__ == "__main__":
    # Test model compilation
    model = get_compiled_model(6041, 3953)
    model.summary()
