# transformer_time_series_forecasting
Transformers applied on Time Series Forecasting

## Structure
1. Data:
`train_raw.csv` and `test_raw.csv` are the raw data files. `train_dataset.csv` and `test_dataset.csv` are the refined data files. 

2. Code:
`DataLoader.py`: Load the sensor data.
`helpers.py`: Applied exponential moving average.
`inference.py`: Applied Transformers to predict the next word. Used MSE loss.
`main.py`: Main entrypoint from where the other functionalities are being called.
`model.py`: Used TransformerEncoderLayer and TransformerEncoder to build the network.
`plot.py`: For the visualization purpose. Plots for loss, prediction of forecast from the Sensor, and training behaviour of an algorithm.
`Preprocessing.py`: Created the positional encoding for the data which is divided in hour, day and month, so it will generate the pattern with sin and cosine for the hour, day, and month.
`train_teacher_forcing.py`: Teacher Forcing technique is for applying at the time of training.
`train_with_sampling.py`: Teacher Forcing technique with Sampling so that we will reduce the proportion of true values supplied to the model overtime. At 310 epoch, the model samples only 20% of it's values from the true input, the other 80% are its own predictions that are built upon. 

3. Plots:
Results are demonstrated for a forecast window of 50 timestamps.