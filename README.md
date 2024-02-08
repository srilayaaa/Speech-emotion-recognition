detects the emotion through the speech signals. This is based on the fact that voice often reflects underlying emotion through tone and pitch.
For analyzing the emotion we need to extract features from audio. Therefore we are using the library Librosa. We are extracting mfcc, chroma, Mel feature from Soundfile.
Opening file from soundfile.Soundfile and read sound from that. Samplerate for obtaining sample rate. If chroma is true then we are obtaining a Short-time Fourier transform of sound. After that extracting feature from Librosa.feature and get the mean value of that feature. Now, store this feature by calling the function hstack(). 
Hstack() stores the features returns at the end of the function. 
