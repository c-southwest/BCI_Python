# BCI Alpha Wave Typing System
 The system allows paralyzed patients to type, chat and use the Linux operating system
 
 ## Introduction
 This repository include the core algorithm. 
 |Folder      | Description|
 |-----       |----------|
 |data        | raw brain wave signal data acquired by record_data.py|
 |psd_data    | Power spectral density data generated from Generate_PSD_data.py|
 |ffmpeg_stimulus_generator | a simple method to generate stable SSVEP stimulus|
 |signal_exp  | unimportant, just try signal processing API or something else |
 
 | File       | Description  |
 | ------     | -----------  |
 | Generate_PSD_data.py | generate Power spectral density data from data folder, then save PSD data to psd_data folder|
 | Online_Alpha_Detector.py | Used to detect alpha brain wave in real time, if detected, then send message to Unity GUI Program via Socket |
 | Online_MI_Detector.py | similar to Online_Alpha_Detector.py and MI stands for motor imagery |
 | Online_SSVEP_Detector.py | similar to Online_Alpha_Detector.py and SSVEP stands for steady state visually evoked potentials |
 | Train_Alpha_Detector.py  | train neural network model for detecting alpha brain wave |
 | Train_MI_Detector.py  | train neural network model for detecting motor imagery |
 | Train_SSVEP_Detector.py  | train neural network model for detecting steady state visually evoked potentials |
 | \*.keras | the Tensorflow Keras model file, can be used to classify EEG signal at real time |
 | read_data_and filter signal.py | Visualize the signal processing flow. |
 | record_data.py | acquire raw EEG signal and save to data folder. |
 | socket_server.py | unimportant, just try the Socket API. |
 | test.py | unimportant, do something temporarily. |
 | utils.py | Encapsulate the signal processing function, make code cleaner. |
 
 
 
 
 ## All related repositories
 |Repository             |Description|
 |----------     |-----------|
 |[https://github.com/c-southwest/BCI_Python](https://github.com/c-southwest/BCI_Python)| Core Algorithm |
 |[https://github.com/c-southwest/BCI_AlphaWriter_Unity3D](https://github.com/c-southwest/BCI_AlphaWriter_Unity3D)| Unity GUI Program|
 |[https://github.com/c-southwest/Discord_Bot](https://github.com/c-southwest/Discord_Bot)| Discord Bot Program|
 
 ## Functions of the complete system
* User can type characters
* Text-To-Speech
* Send messages to Discord channel
* Execute Linux command and display the result

## Youtube Demo Video
<a href="http://www.youtube.com/watch?feature=player_embedded&v=xDUA1Lh9cAE
" target="_blank"><img src="http://img.youtube.com/vi/xDUA1Lh9cAE/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>
