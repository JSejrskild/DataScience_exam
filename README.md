# Automated Speech Recognition for the Verbal Fluency Task
## Annotating and analysing a phonemic verbal fluency task from STN-DBS Parkinsons patients recorded in the MEG at Skejby

## Note:
Due to GDPR, multiple folders are not publicly available. Therefore, the published code is not ready to run, as no data is in the repo. The published code can however be used to inspect the analysis steps done in this project. 

## Files in the folders:
/src:
- ASR_annotation.py -> This file runs the ASR annotation *without* prompts
- ASR_annotation_w_prompt.py -> This file runs the ASR annotation *with* prompts
- compute_linguistics_variables.ipynb -> Computing the phonemic and semantic similarity for the words in clusters
- curve_fitting.py -> This files contains the function that fits individual clusters
- curvefit_and_condition.ipynb -> Here the clusters are fit and plotted. Conditions are appended for further analysis
- enhance_audio.py -> This script takes all audio and enhances using MetricGan+ and saves enhanced files in /enhanced_data
- inter_rater_reliability.py -> This script calculates the cohens kappa for the double annotated data
- plot_annotation.py -> This file plots the accuracy of all individual audiofiles as well as the overall mean accuracy
- transfer_files.py -> This script takes the files in the .zip folder and saves the audio recordings needed in a /data folder

/stat_modelling:
- model_fitting.rmd -> This script fits all the models of DBS condition and Phonemic/semantic similarity on number of clusrters in the VFT
