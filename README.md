# Welcome to BetaZero

### Using machine-learning to set climbs on spray walls and system boards.

## Full-Stack Application (Work-In-Progress)

This is an open-source application which leverages ML models to set custom climbs on homewalls and system boards. It is also an open repository of climbs and spray-walls/system boards. The workflow is as follows:

* Upload your homewall and climb geometry
* Add climbs to the database
* Train MLP, RNN, and LSTM models on your wall's climbs
* Filter climbs by grade and style

## Making Climbs for the Sideways Wall

The original project was completed within `feature-generation` and `model-training`, for handling the creation of the training data, and the training of the LSTM models, respectively.

### Feature-Generation (React Vite -> JSON output)

This is a simple React-Vite application which allows the user to upload a photograph of their home-wall/spray-wall, and then manually add a set of hand/foot holds, embedded via 6 feature vectors (_**x, y, pull_x, pull_y, useability, is_hand**_). It is, unfortunately, no longer in use and is un-maintained. I am currently in the process of porting some of the functionality of `feature-generation` into `climb-front-end`, which when complete will involve a more comprehensive version of the original workflow.

### Model-Training (Python+Jupyter+Torch)

The model-training involved a folder of utility functions for processing the JSON **holds**__, **moves**__ and **climbs**__ datasets into matrices for model training, and using them to train three **nn.Module** models: 

- **Multi-layer perceptron** (MLP) - A simple multi-layered neural network which predicts an output based on a given input.
- **Recursive Neural Network** (RNN) - An MLP which outputs a '**hidden state**' matrix in addition to the standard '**prediction**' output. This hidden state is passed back into the model during the next forward step, allowing it to "_remember_" its prior predictions when making a new prediction.
- **Long Short-Term Memory** network (LSTM) - A more sophisticated version of the RNN which uses a special 'forget gate' to selectively alter the **hidden state** while keeping parts of the hidden state unchanged. This addresses the "vanishing gradient" problem of RNNs: RNNs quickly 'forget' old inputs as each successive forward step alters the **hidden state**.


These models were trained to predict position **_p(i)_** of a climb, given the previous sequence of positions. They were then inserted into a **ClimbGenerator**, a special wrapper class which uses autoregressive generation to create artificial climbing sequences based on the models' predictions.
