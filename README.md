# Identifying and Extracting Football Features from Real-World Media Sources using Only Synthetic Training Data
Real-world images used for training machine learning algorithms are often unstructured and inconsistent. The process of analysing and tag- ging these images can be costly and error prone (also availability, gaps and legal conundrums). However, as we demonstrate in this article, the potential to generate accurate graphical images that are indistinguishable from real-world sources has a multitude of benefits in machine learning paradigms. One such example of this is football data from broadcast services (television and other streaming media sources). The football games are usually recorded from multiple sources (cameras and phones) and resolutions, not to mention, occlusion of visual details and other artefacts (like blurring, weathering and lighting conditions) which make it difficult to accurately identify features. We demonstrate an approach which is able to overcome these limitations using generated tagged and structured images. The generated images are able to simulate a variety views and conditions (including noise and blurring) which may only occur sporadically in real-world data and make it difficult for machine learning algorithm to ‘cope’ with these unforeseen problems in real-data. This approach enables us to rapidly train and prepare a robust solution that accurately extracts features (e.g., spacial locations, markers on the pitch, player positions, ball location and camera FOV) from real-world football match sources for analytical purposes.

Webpage: Access [learningfootball.github.io](https://learningfootball.github.io) 
GitHub Repository: [https://github.com/learningfootball/](https://github.com/learningfootball/learningfootball.github.io).


## Prerequisites
- This code was developed under Windows (10) and Linux (Debian) (64bit) and was tested only on these environments.  
- The generated images were done using Chrome (105/64bit) and the WebGPU API.
 
- Python (3.1)
  - PyTorch 1.11.0
  - CUDA >= 11.7
