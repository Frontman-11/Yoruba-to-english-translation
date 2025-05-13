import keras
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split


class ReadFile():
    def __init__(self, dropna=False, drop_duplicates=False, random_state=42):
        self.dropna = dropna
        self.random_state = random_state
        self.drop_duplicates = drop_duplicates 

    def _normalize(self, df):
        df.columns = df.columns.str.capitalize()
        return df.map(lambda x: x.strip() if isinstance(x, str) else x)
        
    def _read_single_file(self, path, delimiter):
        delimiter = '\t' if path.lower().endswith('.tsv') else ',' if path.lower().endswith('.csv') else delimiter
        df = pd.read_csv(path, delimiter=delimiter)
        df = self._normalize(df)
        return df[['Yoruba', 'English']]

    def drop_or_buffer(self, df):
        if self.dropna:      
            df.dropna(how='any', inplace=True)
        if self.drop_duplicates:
            df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
        
    def _read_file(self, filepath, delimiter, ignore_index=True): 
        df = pd.DataFrame()
        if isinstance(filepath, list):
            for path in filepath:
                a = self._read_single_file(path, delimiter=delimiter)
                df = pd.concat([df, a], axis=0, ignore_index=ignore_index)
        else:
            df = self._read_single_file(filepath, delimiter=delimiter)
        return self.drop_or_buffer(df)

    def split_file(self, df, split_ratio):
        return train_test_split(df, test_size=split_ratio, random_state=self.random_state)

    def shuffle_df(self, df):
        return df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)      

    def read_file(self, filepath, delimiter='\t', ignore_index=True, shuffle=None, split_ratio=None):
        if split_ratio:
            train_df = pd.DataFrame()
            test_df = pd.DataFrame() 
            
            for path in filepath:
                df = self._read_file(path, delimiter=delimiter, ignore_index=ignore_index)
               
                if shuffle:
                    df = self.shuffle_df(df)
                    df, df1 = self.split_file(df, split_ratio)
                    train_df = pd.concat([train_df, df], axis=0, ignore_index=ignore_index)
                    test_df = pd.concat([test_df, df1], axis=0, ignore_index=ignore_index)
            
            return self.drop_or_buffer(train_df), self.drop_or_buffer(test_df)
                        
        else:
            return self._read_file(filepath, delimiter=delimiter, ignore_index=ignore_index)            
        

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
