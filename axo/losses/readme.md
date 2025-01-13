# Recipe to add new Loss

1. **Create a new Python file** in the losses directory. Example name: `mae_loss.py`. 

    ```sh
    touch mae_loss.py
    ```

2. **Define the Loss Function**:

    - Open the `mae_loss.py` file in your text editor or IDE.
    - Define a function named `_mae_loss` (note the leading underscore) that calculates the Mean Absolute Error. This function will be used internally to compute the loss. The below code gives a minimal example. Define more complicated losses as per your convinience :)

    ```python
    # mae_loss.py

    import tensorflow as tf
    from tensorflow.keras.losses import Loss
    from tensorflow.keras import backend as K
    
    class _mae_loss(Loss):
        def __init__(self, name="MAE"):
            super().__init__(name=name)
            
        def call(self, y_true, y_pred):
            return K.mean(K.abs(y_true - y_pred), axis=-1)
    ```

    In this code:
    - `y_true` represents the true values.
    - `y_pred` represents the predicted values.
    - `tf.reduce_mean` calculates the mean of the absolute differences between `y_true` and `y_pred`.
    
    This is for a simple loss that does not need the scale files. If something more complicated is needed inherit the L1ADBaseLoss loss class.

3. **Update the `__init__.py` File**:

    - Locate the `__init__.py` file in the losses directory.
    - Open the `__init__.py` file and add a new line to import and rename the function you defined in `mae_loss.py`.

    ```python
    # __init__.py

    from .mae_loss import _mae_loss as mae_loss
    ```