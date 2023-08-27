# Privacy_learning 
This directory contains 3 python files 1 notebook and a directory that is made up of the datasets.

Main.py is the code repository that stores the functions to be imported by the other files. 
It is made up of the ESA function which is the feature inference attack, The read_data function which is 
used to open the different datasets in their respective ways, The log reg function which generates the logistic 
regression model, and The random minibatches which is used in training the model.

The Training.py is where the training and testing of the model is done, when this file is run a 
pytorch file is the output which is created which is the final model. it is what the attack.py uses to carry out the ESA attack.

The attack.py file is where the ESA attack was carried ouyt. It shows the configuration of the ESA function which was imported from the main.py file.
It makes use of the model generated from the training.py file and the output is a pickle file containing a list of the MSE per feature.

The graph.py file is used to display the graphs of the MSE using the pickle files from attack.py.
The IPYNb file shows the preprocessing of the bank additional dataset to study the correlation between featues
