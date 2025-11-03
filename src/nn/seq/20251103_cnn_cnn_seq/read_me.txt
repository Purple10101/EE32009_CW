Note:
	- All noise injection.
	- Norm the whole thing (al 1.4m samples)
		- This causes issues as the highest spikes are a result of compounded n activations.
	- This model performs better when SNR degrades than previous submission.


Spike Detection:
1D CNN
    - Batch Normalization is implemented in this version.
    - Each convolutional layer output is normalized across the batch dimension.
        - This ensures each feature channel has a stable mean and variance
          before being passed to the next layer.
        - Implemented with nn.BatchNorm1d(...).
    - Dilation is used to expand the convolution’s receptive field exponentially.
        - This allows the network to capture a wider temporal context
          without increasing the number of parameters.

nn.Conv1d(input_channels, 32, kernel_size=7, padding=3, dilation=1),
nn.BatchNorm1d(32),
nn.ReLU(),
nn.Conv1d(32, 64, kernel_size=5, padding=4, dilation=2),
nn.BatchNorm1d(64),
nn.ReLU(),
nn.Conv1d(64, 128, kernel_size=3, padding=4, dilation=4),
nn.BatchNorm1d(128),
nn.ReLU(),
nn.Dropout(0.3),
nn.Conv1d(128, 1, kernel_size=1),

Each n inputs maps to n output probabilities
If the probability at the output is > 0.68 it is a spike

Classification
CNN 
[1, 16, 5, padding=2] -> ReLu -> [16, 32, 3, padding=1] -> [32, 64, 3, padding=1] -> ReLu -> AdpAvgPool1d
Input captures (images) are 80 samples wide and normalized between 1 and -1
with respect to the whole data set.
