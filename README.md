# DigitIdentifierAI
A neural network that identifies hadnwritten digits (0-9) from 28x28 monochromatic images. The images are represented as 784 pixel values (0-255) and an answer value.
The program takes each data set, inputs them into the neural network, which generates a 1x10 array of what digit it most likely is. It then compares if the answer is correct,
and computes gradients through backpropagation, which are then applied to all the weights and biases for adjustment, which allows the network to learn. 
This program allows the user to train the network, save and upload weights and biases from a .txt file, test the network with MNIST testing and training data, and to test the network
while seeing each input image with ASCII art. 

The network is currently a 784x50x10 neural network. Each layer of the network has its own weights and biases arrays to facilitate calculations and to make adjustments easier while training.

Training:
To train the network, Stochastic Gradient Descent is utilized, and the training data is broken into mini-batches to make this process quicker. The size of each mini-batch is 10 images,
and the network is trained on 30 epochs of 6000 minibatches, totalling 60,000 images. After each mini-batch, the weight and bias gradients are then applied to the existing weights and biases of each neuron in the network. 

The current accuracy of the neural network after 2 training sessions is 97%.

Testing:
To test the network, 10,000 images from the MNIST Testing data set is fed to the network. The network has never seen these image before, so the accuracy is slightly reduced at 94% accuracy.

Observing Categorization:
The program allows the user to see how the network categorized each image, and what the image looked like through ASCII art. The user can also specify to look at just the incorrectly classified images to see
where the network was wrong.

Saving Weights and Biases:
The user can decide to save the current weights and biases to a .txt file at any time while the program is running. If a file already exists with the same name, the program will overwrite the existing file. Weights
and biases can also be uploaded from a .txt file of the same name that the program saves to, which removes the need to restart training each time the program is run.
