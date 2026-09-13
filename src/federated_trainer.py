import os
import tensorflow as tf
import tensorflow_federated as tff
from base_model import get_compiled_model
from data_loader import load_data, create_federated_partitions

# Allow TFF to execute on GPUs if available, otherwise it falls back to CPU
# This requires python 3.9+ and TF 2.x
# Note: TensorFlow Federated setup can be tricky; this is a simplified simulation loop

def preprocess_for_tff(client_df):
    """
    Convert pandas dataframe of a client to tf.data.Dataset
    """
    # Create dataset from tensor slices
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            "user_id": client_df["user_id"].values,
            "movie_id": client_df["movie_id"].values,
        },
        client_df["rating"].values
    ))
    # Batch the dataset
    return dataset.batch(32)

def model_fn():
    # Number of users (6041) + 1 for OOV, movies (3953) + 1 for OOV in ml-1m
    # We use a compiled Keras model and convert it to TFF model format
    keras_model = get_compiled_model(num_users=6041, num_items=3953)
    
    # We must define the input spec based on our preprocessed data
    input_spec = (
        {
            'user_id': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'movie_id': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        },
        tf.TensorSpec(shape=(None,), dtype=tf.int64)
    )
    
    return tff.learning.models.from_keras_model(
        keras_model,
        input_spec=input_spec,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

def main():
    print("Loading data...")
    users, movies, ratings = load_data()
    
    print("Partitioning data for federated learning...")
    clients_data = create_federated_partitions(ratings, num_clients=10)
    
    print("Preprocessing for TFF...")
    # Convert client datasets
    tff_train_data = []
    for client_id in clients_data.keys():
        tff_dataset = preprocess_for_tff(clients_data[client_id])
        tff_train_data.append(tff_dataset)
        
    print("Building federated averaging process...")
    # Build federated averaging algorithm
    fed_avg = tff.learning.algorithms.build_unweighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
    )
    
    print("Initializing federated state...")
    state = fed_avg.initialize()
    
    # Simulate a few rounds of training
    num_rounds = 3
    for round_num in range(num_rounds):
        print(f"--- Round {round_num + 1} ---")
        result = fed_avg.next(state, tff_train_data)
        state = result.state
        metrics = result.metrics
        print(f"Metrics: {metrics['client_work']['train']}")

if __name__ == "__main__":
    main()
