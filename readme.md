**Training Ms. Pac-Man Agent with SEGA Genesis Emulator**
by AznIronMan @ ClarkTribeGames, LLC


This project trains a reinforcement learning agent to play the classic Ms. Pac-Man game using the SEGA Genesis emulator. Follow the steps below to set up and run the training process.


**Prerequisites**


Make sure you have Python 3 installed on your system. If you don't have it installed, you can download it from the official Python website.


**Installation**


Install the required Python packages by running the following command in the terminal:

*pip install -r _requirements.txt*


**ROM file**


Obtain a legal Ms. Pac-Man SEGA Genesis ROM file and place it in the roms directory of this project.


**Importing the ROM file**


Run the following command in the terminal to import the ROM file:

*python -m retro.import roms/(mspacman_rom_file_name)*


**Configuration**


Make sure that the .env file in the project directory is properly configured. These are the default values:

GAME_NAME=MsPacMan-Genesis 
MODEL_PREFIX=ms_pacman_ppo_ 
MODEL_FILE_SUFFIX=final.zip 
MODEL_DIR=models 
TENSORBOARD_LOG=./tensorboard/ 
LOG_DIR=logs 
RENDER=False 
TIMESTEPS=1000000 
TRIALS=50 
PARALLEL_ENVS=128 
LEARNING_RATE=0.0005 
N_EPOCHS=10 
N_EVAL_EPISODES=10 


**Training**


Run the following command in the terminal to start training the agent:

*python train_genesis.py*

The training process will save the trained model to the MODEL_DIR directory and log the training progress to the LOG_DIR directory.


**Conclusion**


Following the above instructions should enable you to set up and run the Ms. Pac-Man SEGA Genesis emulator reinforcement learning agent training process. 

**More Info on Gym-Retro:**  https://retro.readthedocs.io/en/latest/
