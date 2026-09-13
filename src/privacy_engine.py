import tensorflow as tf
import tensorflow_privacy as tfp
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer

def get_dp_optimizer(learning_rate, l2_norm_clip=1.0, noise_multiplier=0.1, num_microbatches=1):
    """
    Returns a Differentially Private Adam optimizer using TensorFlow Privacy.
    """
    optimizer = DPKerasAdamOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate=learning_rate
    )
    return optimizer

def apply_dp_to_model(keras_model, learning_rate=0.01, l2_norm_clip=1.0, noise_multiplier=0.5):
    """
    Recompiles the given keras_model with a Differentially Private optimizer.
    This limits how much an individual rating can affect the global model.
    """
    dp_optimizer = get_dp_optimizer(learning_rate, l2_norm_clip, noise_multiplier)
    
    keras_model.compile(
        optimizer=dp_optimizer,
        loss=tf.keras.losses.MeanSquaredError(), # For DP, consider MeanSquaredError with reduction=NONE if using microbatches
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )
    
    return keras_model
