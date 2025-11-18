Previous submission used a dule chan for classification which was no good.
This new seq is based on a simpler cls CNN.

Data is spectrally degraded for training and then band passed for event detection.
Data is left alone for training classification. We boost SNR in inference wit spectral matching.

Event detection chain:
	
	Training:  D1 -> Spectral degrade -> high pass -> norm window-wise (ind model per target)
	Inference: DN -> high pass -> norm

Cls chain:
	
	Training:  D1 -> norm (single model)
	Inference: DN -> high pass -> spectral boost -> norm 