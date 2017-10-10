# Improved Wasserstein GAN application on MRI dataset

Application of a deep generative model on MRI images of knees. The MRI database used was provided by Imperial College London, however similar databases can be found on the OAI website (http://www.oai.ucsf.edu/), an observational study dedicated to monitor the natural evolution of osteoarthritis.

# Prerequisites
- Python, Lasagne (developer version), Theano (developer version), Numpy, Matplotlib, scikit-image
- NVIDIA GPU (5.0 or above)
# Results
- **Examples of real images from the input dataset** 
<img src="results/ground_truthgan.png" alt="alt text" width="600" height="600">

- **Examples of generated images 4500 iterations**
<img src="results/examples_34gen.png" alt="alt text" width="600" height="600">

- **Evolution of generated images at various iterations (total of 35 epochs - around 4500 iterations)**
<img src="results/evolution.png" alt="alt text" width="600" height="400">

# License
This project is licensed under Imperial College London.
# Acknowledgements
The following codes were used as a base:
- for the main skeleton for a lasagne implementation of an adversarial generative network: https://github.com/ToniCreswell/AllThingsGAN/blob/master/Code/dcgan.py 
- for WGAN-GP implementations
 https://github.com/tjwei/GANotebooks and
 https://github.com/igul222/improved_wgan_training


