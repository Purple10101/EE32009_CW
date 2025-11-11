Note:
	- All noise injection.
	- Norm the whole thing (al 1.4m samples)
		- This causes issues as the highest spikes are a result of compounded n activations.
	- This model performs better when SNR degrades than previous submission.


This model has a peak detection component with boosted performance on early datasets.
The submission for this seq is intended to boost performance for d2-4 but the lower SNR datasets will likely show no performance benefit.

As the classification of spikes (1-5) is untouched there is defo some signal processing stuff you can do here to remove the low frequency swing!
