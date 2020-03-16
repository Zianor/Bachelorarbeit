# Data Format

The samplerate of all data is 100 hz.

Furthermore the .mat files contain the following fields:
* 'BCG_raw_data' containing the raw bcg data
* 'q_BCG' containing a SQI which denotes the similarity of the 3 Brüser estimators (0 to 1)
* 'BBI_BCG' containing the beat-to-beat interval of the BCG signal in seconds
* 'BBI_ECG' containing the beat-to-beat interval of the ECG signal in seconds
* 'indx' containing the index of a point of interest in the BCG signal where an interval is located

