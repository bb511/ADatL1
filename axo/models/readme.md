# Recipe to add new model

## Step 1: Create a New `.py` File

Create a new Python file named `custom_mlp.py` where you will define your MLP model.

```bash
touch custom_mlp.py
```
## Step 2: Update the `__init__.py` file with the proper import

```python
# __init__.py

from .custom_mlp import CustomMLP
```
## Step 3: Write the MLP Model Using Model Subclassing in Keras
File: `custom_mlp.py`

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class CustomMLP(Model):
    def __init__(self, input_shape, num_classes):
        super(CustomMLP, self).__init__()
        self.dense1 = layers.Dense(64, activation='relu', input_shape=input_shape)
        self.dense2 = layers.Dense(64, activation='relu')
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.output_layer(x)
```
The output of the above model will be later handled by a corresponding script.