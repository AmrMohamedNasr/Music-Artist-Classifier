# MuseClass : Music Classification By Composer
---
## Overview
    Where words fail, music speaks.
    -Hans Christian Andersen
That’s why music is so interesting and why People love to listen to music coming from specific person. We decided to classify music according to composers, we will receive the music and select the composer that most likely composed this music piece from 9 composers or a class of others. This problem is much harder than genre classification due to its complexity, however, logically its models can be generalized to help in other music classifications tasks.

Our project was based on the stanford project completed in fall 2018. This project report can be found [here](http://cs230.stanford.edu/projects_fall_2018/reports/12441334.pdf). Differences between our approaches are that we introduced a class of others that wasn't available in the original project, and that we classify on 30 second intervals only while they considered the interval size to classify to be a hyperparameter that can be tuned allowing them more flexibility.

---
## Input/ output Example
Our input is a midi file. A midi file, which contains tracks and chords represented as a tensor with dimensions of #tracks, #notes, #time which we will use in our models for classification. While we can take any midi files with the length of more than 30 seconds, our models work on classifying 30 seconds intervals of music. The input dimensions are 1 track, 128 musical note, and 30 second.

Our output is the label of the artist who composed this segment. It can be one of the following composers : 'chopin', 'bartok', 'hummel', 'mendelssohn', 'bach', 'byrd', 'handel', 'schumann', 'mozart' or 'other' if it doesn't match any composer.

---
## Results
The original stanford project had an accuracy of 58%. Our best model managed to achieve 92.68293% accuracy.

---
The html page, and the various PDFs included in the repository have a lot more details about this project.

---
