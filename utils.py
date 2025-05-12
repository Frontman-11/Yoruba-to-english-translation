import keras
import pandas as pd
import tensorflow as tf


def read_file(filepath, delimiter='\t'):
    '''Returns pandas dataframe with Capitalized column names and stripped text.
    Drops nan rows if any, drops duplicates. Returns concatinated dataframe if 
    filepath is a list, otherwise returns dataframe for filepath'''
    
    if isinstance(filepath, list):
        df = pd.DataFrame()
        for path in filepath:
            a = pd.read_csv(path, delimiter=delimiter)
            a.columns = a.columns.str.capitalize()
            
            # Strip trailing and leading spaces from all string values
            a = a.map(lambda x: x.strip() if isinstance(x, str) else x)
            
            df = pd.concat([df, a], axis=0, ignore_index=True)
            
        df.dropna(how='any', inplace=True)
        df.drop_duplicates(inplace=True)
        
        return df
    
    else:
        df = pd.read_csv(filepath, delimiter=delimiter)
        df.dropna(how='any', inplace=True)
        df.columns = df.columns.str.capitalize()
        
        # Strip trailing and leading spaces from all string values
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
        df.drop_duplicates(inplace=True)
        
        return df
        

@keras.saving.register_keras_serializable()
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        return {
            "d_model": self.d_model.numpy(),
            "warmup_steps": self.warmup_steps
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred
    
    mask = label != 0
    
    match = match & mask
    
    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)
