# Channel State Model Estimation

This code was written in MATLAB making use of the communications toolbox for wireless channel simulation.  Further, I created an Long Short-Term Memory (LSTM) Model for predicting the channel state (that is, how much of my transmission will reach its intended target?).  

## How it works:

First, a number of transmission through a wireless channel are simulated.  This is then saved in memory and used to train a LSTM model, and its accuarcy is evaluated.  

## Upcoming

The next to-do is to make this into a classifier: rather than estimating the channel using knowledge from the receiver (which is unrealistic), can you estimate the channel using incoming transmissions only?  What about another node's transmissions?  