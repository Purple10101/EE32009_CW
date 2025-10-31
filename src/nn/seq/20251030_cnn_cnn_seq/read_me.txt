Note:
	- No noise injection.
	- Norm the whole thing (al 1.4m samples)
		- This causes issues as the highest spikes are a result of compounded n activations.
	- This model probably suck when SNR degrades!


Spike Detection:
1D CNN
[1, 16, 5] -> [16, 32, 5] -> [32, 1, 5] -> sigmoid to get a prob
Each n inputs maps to n output probabilities
If the probability at the output is > 0.5 it is a spike

Classification
CNN 
[1, 16, 5, padding=2] -> ReLu -> [16, 32, 3, padding=1] -> [32, 64, 3, padding=1] -> ReLu -> AdpAvgPool1d
Input captures (images) are 80 samples wide and normalized between 1 and -1
with respect to the whole data set.
