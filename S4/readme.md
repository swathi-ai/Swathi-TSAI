This is an assignment on MNIST data and the target is 
            to create a model that achieves a 99.4% test accuracy 
            with in or equal to 15 epochs 
            and no of parameters must be less than 10K
            
As a first step i created a basic skeleton of the model with less no of parameters. To do this, i used very less no of kernels. Model was little overfit 

As a second step, batchnormalization is added to the model hence accuracy is improved a little but still model is overfit

As a third step, Dropout with a value of 0.05 is added to the model hence the difference between train and test accuracy is decreased a lot.

As a fourth step, image augmentation technique -  random rotation has been applied hence the difference between train and test accuracy is decreased furthur.

To improve the accuarcy furthur, tweaked learning rate and optimizer hence accuarcy has reached to 99.4%  with 8K+ parameters and in 15 epochs which is our target
