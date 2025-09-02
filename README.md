# High-Frequency-Predictor

This is a demo for testing the predictive power of mid-to-high-frequency data. It is primarily divided into the following parts:

-- The Model module defines the model. All functions related to the model should be implemented under this module, and other code should call its interfaces for model training and prediction. The initial prediction target is a classification problem (regression problems may be considered later). The initial model is LightGBM.

-- The DataLoader module defines data preprocessing methods. All methods for processing raw data should be completed within this module.

-- The Stats module defines methods for statistical information. Whether this module is integrated into the Model module depends on whether other statistical metrics need to be calculated.

#### Workflow

The simplified workflow should proceed as follows:

· Test expressive factors.
· Call the DataProcess module to obtain input and output arrays X and y in the form of np.array.
· Call the Model module for training and prediction, and print statistical information.

#### Module Calls

-- The Model module is called via:
Model.fit(x, y, modelname='lgbm_regression')

### Log

##### 2025-08-17

· Set up a rapid testing framework.
