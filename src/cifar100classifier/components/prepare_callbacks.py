import os
import tensorflow as tf

from cifar100classifier.entity import PrepareCallbacksConfig
from cifar100classifier.utils import *

class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    @property
    def _create_tb_callbacks(self):
        
        """TensorBoard is a tool for providing the measurements and visualizations 
        needed during the machine learning workflow.
        
        This callback logs events for TensorBoard, including:
            Metrics summary plots,
            Training graph visualization,
            Weight histograms,
            Sampled profiling"""
                
        timestamp = get_current_time_stamp()  #time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir) # tensorboard callbacks

    @property
    def _create_ckpt_callbacks(self):
        
        """Callback to save the Keras model or model weights at some frequency."""
        
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.checkpoint_model_filepath,
            save_best_only=True
        )
        
    @property
    def _create_early_stopping_callbacks(self):
        
        """Stop training when a monitored metric has stopped improving."""
        
        return tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.1,
            patience=5,
            verbose=1,
            mode='min',
            baseline=None,
            restore_best_weights=True
        )
        
    @property
    def _create_reduce_LR_onplateau_callbacks(self):
        
        """Reduce learning rate when a metric has stopped improving."""
        
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            verbose=1,
            mode='min',
            min_delta=0.1,
            cooldown=0,
            min_lr=1e-6
        )

    def get_callbacks(self):
        return [
            self._create_tb_callbacks,
            self._create_ckpt_callbacks,
            self._create_early_stopping_callbacks,
            self._create_reduce_LR_onplateau_callbacks
        ]
