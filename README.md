# captcha
general character recogonizer

## Dependency
- Tensorflow
- PIL

## Description
This project's destination is to recogonize any character sequence in one image.

Util reaching the final goal, it may need a number of steps. Espeacially, the final result may not satisfy us.

Charactor recogonition task is very hard in uncertain context, the training set is not same with the test set.

In this project, I will adopt a new policy to do this thing.

Firstly, I will train the network in some steps seperately with a generated training set which will be not related with the captcha world.

Secondly, I will try to introduce attention mechinism into the Neural Network.

Thirdly, I will employ Dynamic Conventional Neural Network(DCNN) and give the network time to focus, think and infer.

Fourthly, I will visualize all the middle layers so that the NN is under my control.

## Structure
The Generator will generates all the training sets, the Trainer will train every layer for DCNN, the Runner will run the DCNN to do the captha recognition tasks.

- util.py - common utils for images and text
- generator.py - The Generator
- trainer.py - The Trainer
- runner.py - The Runner
- nn.net - The trained and saved network parameters
