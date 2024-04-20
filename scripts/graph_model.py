import spektral
from tensorflow.keras.models import Model
from spektral.layers import GCNConv, GlobalAttentionPool
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

class GraphModel(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(32, activation='relu')
        self.conv2 = GCNConv(32, activation='relu')
        self.pool = GlobalAttentionPool(32)
        self.dropout = Dropout(0.5)
        self.dense = Dense(1, activation=None)

    def call(self, inputs):
        x, a = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.pool(x)
        x = self.dropout(x)
        return self.dense(x)

model = GraphModel()
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error', metrics=['mean_absolute_error'])