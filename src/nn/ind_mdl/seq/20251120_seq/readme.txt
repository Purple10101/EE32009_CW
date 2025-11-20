
Event Detection Pipelines:

	Training: zscore each step
	
	d1 -> d1 spectral degraded = d1_dn -> d1_dn spectral suppressed = d1_dn_d1 -> d1_dn_d1 wavelet = d1_dn_d1_wt -> d1_dn_d1_wt bandpass = d1_dn_d1_wt_bp
	for dn = 5: d1_dn_d1_wt_bp adding colored noise with std=3 and increased refractory from 3 to 10
        for dn = 6: d1_dn_d1_wt_bp adding colored noise with std=5 and increased refractory from 3 to 10

	Inference: zscore each step

	dn -> dn spectral suppress = dn_d1 -> dn_d1 wavelet = dn_d1_wt -> d1_dn_d1_wt bandpass = dn_d1_wt_bp

Classification Pipelines:

	Traning: zscore each step

	d1 -> d1 spectral degraded = d1_dn -> d1_dn spectral suppressed = d1_dn_d1 -> d1_dn_d1 bandpass = d1_dn_d1_bp
	for dn = 5: d1_dn_d1_bp adding colored noise with std=3 and increased refractory from 3 to 10
        for dn = 6: d1_dn_d1_bp adding colored noise with std=5 and increased refractory from 3 to 10

	Inference: zscore each step

	dn -> dn spectral suppress = dn_d1 -> d1_dn_d1 bandpass = dn_d1_bp

	The lack of a wavelet here is because we suspect that the shape is really corrupted by this process.