# Time Series Forecasting using Transformers
Transformers applied on Time Series Forecasting.

## Structure
**1. Data:**
`train_raw.csv` and `test_raw.csv` are the raw data files. `train_dataset.csv` and `test_dataset.csv` are the refined data files. 

**2. Code:**</br>
`DataLoader.py`: Load the sensor data.</br>
`helpers.py`: Applied exponential moving average.</br>
`inference.py`: Applied Transformers to predict the next word. Used MSE loss.</br>
`main.py`: Main entrypoint from where the other functionalities are being called.</br>
`model.py`: Used TransformerEncoderLayer and TransformerEncoder to build the network.</br>
`plot.py`: For the visualization purpose. Plots for loss, prediction of forecast from the Sensor, and training behaviour of an algorithm.</br>
`Preprocessing.py`: Created the positional encoding for the data which is divided in hour, day and month, so it will generate the pattern with sin and cosine for the hour, day, and month.</br>
`train_teacher_forcing.py`: Teacher Forcing technique is for applying at the time of training.</br>
`train_with_sampling.py`: Teacher Forcing technique with Sampling so that we will reduce the proportion of true values supplied to the model overtime. At 310 epoch, the model samples only 20% of it's values from the true input, the other 80% are its own predictions that are built upon. 

**3. Plots:**</br>
Results are demonstrated for a forecast window of 50 timestamps. Predictions on an unseen sequence with a forecast window of 50, using a model trained for 400 epochs. Listed below in 4 graphs:
![Plot 1: from epoch 0 to epoch 275](https://github.com/Anshita1Saxena/transformer_time_series_forecasting/blob/main/Image/Forecast%20plot2.png)
![Plot 2: from epoch 275 to epoch 400](https://github.com/Anshita1Saxena/transformer_time_series_forecasting/blob/main/Image/Forecast%20plot1.png)

<b>Teacher Forcing Effect:</b> </br>
At timestamp 20, the model had moved significantly off track and predicted a humidity of 65%. The sampler selected the true output as the next input, and the model succeeded in correcting itself almost perfectly in the subsequent timestamp 21.
![Plot: Teacher Forcing](https://github.com/Anshita1Saxena/transformer_time_series_forecasting/blob/main/Image/Teacher%20Forcing%20Method.png)

This is intuitive, considering the model is predicting with very little knowledge of the specific sequence. Training with the scheduled sampler that kicks in early in the sequence was demonstrated to confuse the model, as it is penalized for poor predictions in cases where it does not have sufficient input to understand patterns in the data yet. To correct this, a threshold is set on the scheduled sampler, ensuring that only true values are used for the first 24 time-stamps, representing one full day of data, before the scheduled sampler kicks in at the normal rate demonstrated in the inverse sigmoid graph, correlated with the epoch.
Scheduled Sampler with threshold. The first 24 time stamps come only from the true input, regardless of the probability set by the scheduled sampler.
![Plot: Teacher Forcing with Scheduling](https://github.com/Anshita1Saxena/transformer_time_series_forecasting/blob/main/Image/Teacher%20Forcing%20Method(Schedular%20Sampling).png)
