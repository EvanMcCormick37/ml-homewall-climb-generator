# Welcome to Beta-Zero

### Using machine-learning to set climbs on spray walls and system boards.

## Full-Stack Application (Work-In-Progress)

This is an open-source application which leverages ML models to set custom climbs on homewalls and system boards. It is also an open repository of climbs and spray-walls/system boards. The workflow is as follows:

* Upload your homewall and climb geometry
* Add climbs to the database
* Train MLP, RNN, and LSTM models on your wall's climbs
* Filter climbs by grade and style

## Making Climbs for the Sideways Wall

See my write-up about this stage of the project here: https://evmojo37.substack.com/p/beta-zero-alpha-can-ai-set-climbs

The original project was completed within `feature-generation` and `model-training`. `feature-generation` involved creating the training data (holds and climbs) using a custom React-Vite UI. `model-training` consisted of processing the holds and climbs data into useable training data, and training the ML models.

### Feature-Generation (React Vite -> JSON output)

This is a simple React-Vite application which allows the user to upload a photograph of their home-wall/spray-wall, and then manually add a set of hand/foot holds, embedded via 6 feature vectors (_**x, y, pull_x, pull_y, useability, is_hand**_). It is, unfortunately, no longer in use and is un-maintained. I am currently in the process of porting some of the functionality of `feature-generation` into `climb-front-end`, which when complete will involve a more comprehensive version of the original workflow.

### Model-Training (DDPM)

The latest version uses a Denoising Diffusion Probabilistic Model (DDPM) to generate climbs as point clouds, where each point corresponds to a hold used in the climb. It then guides the generated points to actual valid holds via Manifold Guidance. My utility classes for DDPM climb generation and model training can be found in the 'equivariant_diffusion' folder of the 'model-training' directory.
![EditingHolds](https://github.com/user-attachments/assets/878edd1c-b98b-4dd1-bf37-b0cb4e22e3fa)
