//***********************************
// Name: Jacob Stein
// CWID: 10391595
// Class: CSC 475 001
// Assignment #2
// Date: 10/23/2023
// Description: A neural network that identifies hadnwritten digits (0-9) from 28x28 monochromatic images. The images are represented as 784 pixel values (0-255) and an answer value.
// The program takes each data set, inputs them into the neural network, which generates a 1x10 array of what digit it most likely is. It then compares if the answer is correct,
// and computes gradients through backpropagation, which are then applied to all the weights and biases for adjustment, which allows the network to learn. 
// This program allows the user to train the network, save and upload weights and biases from a .txt file, test the network with MNIST testing and training data, and to test the network
// while seeing each input image with ASCII art. 
//***********************************

import java.lang.Math;
import java.util.*;
import java.io.*;

class DigitIdentifier{
    //2d arrays for the weights and biases associated with each neuron in the network
    public static double[][] L1Weights = new double[50][784];
    public static double[] L1Bias = new double[50];
    public static double[][] L2Weights = new double[10][50];
    public static double[] L2Bias = new double[10];

    //The batches 2d array used for training, and the 2d array for the neurons in the network
    public static int[][] batches = new int[6000][10];
    public static double[][] neuralNetwork = new double[3][];

    //Unchanging values, including metadata
    public static final int L1neurons = 50;
    public static final int L2neurons = 10;
    public static final int learningRate = 12;
    public static final int miniBatchSize = 10;
    public static final int epochs = 30;

    //Arrays for gradients used for backpropogation
    public static double[] L2BiasGrad;
    public static double[][] L2WeightGrad;
    public static double[] L1BiasGrad;
    public static double[][] L1WeightGrad;

    public static void main(String[] args){

        //Menu; must first either train the network or upload pre-trained weights and biases
        Scanner myScan = new Scanner(System.in);
        boolean running = true;

        System.out.println("Choose an option:");
        System.out.println("[1] Train network");
        System.out.println("[2] Upload pre-trained network");
        System.out.println("[0] Exit");
        String choice = myScan.nextLine();
        int choiceInt = 0;
        choiceInt = Integer.valueOf(choice);

        //Initializes the neural network with 15 in the hidden layer and 10 in the output layer. Input layer is initialized later.
        DigitIdentifier.initNeuralNetwork();
        //Choice to initially train the network
        if(choiceInt == 1){
            //Input training data inputs from MNIST training file
            double[][] inputArrays = new double[60000][784];
            inputArrays = DigitIdentifier.inputDataInputs("mnist_train.csv", inputArrays);
            //Input training data answers from MNIST training file
            int[][] answerArrays = new int[60000][10];
            answerArrays = DigitIdentifier.inputDataAnswers("mnist_train.csv", answerArrays);
            //Input randomized weights and biases with a range of -1.0 to 1.0
            double minW = -1.0;
            double maxW = 1.0;
            Random randD = new Random();
            //Input weights to Layer 1 (Hidden Layer) and Layer 2 (Output Layer)
            DigitIdentifier.L1Weights = DigitIdentifier.initRandWeights(minW, maxW, randD, DigitIdentifier.L1Weights);
            DigitIdentifier.L2Weights = DigitIdentifier.initRandWeights(minW, maxW, randD, DigitIdentifier.L2Weights);
            //Input biases for Layer 1 and Layer 2
            DigitIdentifier.L1Bias = DigitIdentifier.initRandBiases(minW, maxW, randD, DigitIdentifier.L1Bias);
            DigitIdentifier.L2Bias = DigitIdentifier.initRandBiases(minW, maxW, randD, DigitIdentifier.L2Bias);
            //Train Network
            DigitIdentifier.networkTraining(inputArrays, answerArrays);
        } else if(choiceInt == 2){
            //Choice to upload pre-trained weights and biases
            DigitIdentifier.uploadFromFile();
        } else {
            running = false;
        }

        //Constant loop until the choice to exit the program is made
        while(running){
            //Menu options
            System.out.println("Choose an option:");
            System.out.println("[1] Train network further");
            System.out.println("[2] Upload pre-trained network");
            System.out.println("[3] Test network on TESTING data");
            System.out.println("[4] Test network on TRAINING data");
            System.out.println("[5] See all network test data with images");
            System.out.println("[6] See incorrect network test data with images");
            System.out.println("[7] Save network to file");
            System.out.println("[8] Exit");
            choice = myScan.nextLine();
            choiceInt = Integer.valueOf(choice);

            //Input default activation values for subsequent layers into neural network
            DigitIdentifier.initNeuralNetwork();

            //Currently further trains the network with current weights and biases
            if(choiceInt == 1){
                //Input training data inputs from MNIST training file
                double[][] inputArrays = new double[60000][784];
                inputArrays = DigitIdentifier.inputDataInputs("mnist_train.csv", inputArrays);
                //Input training data answers from MNIST training file
                int[][] answerArrays = new int[60000][10];
                answerArrays = DigitIdentifier.inputDataAnswers("mnist_train.csv", answerArrays);

                //Train Network
                DigitIdentifier.networkTraining(inputArrays, answerArrays);
            } else if(choiceInt == 2){
                //Uploads weights and biases from the save file again. Useful if you want to start training again at a checkpoint
                DigitIdentifier.uploadFromFile();
            } else if(choiceInt == 3) {
                //Test the network on testing data
                //Inputs data into the inputArrays and answerArrays
                double[][] inputArrays = new double[10000][784];
                inputArrays = DigitIdentifier.inputDataInputs("mnist_test.csv", inputArrays);
                int[][] answerArrays = new int[10000][10];
                answerArrays = DigitIdentifier.inputDataAnswers("mnist_test.csv", answerArrays);

                //Tests the network. (Testing involves a straight run through the data. There are no batches and no backpropogation.)
                DigitIdentifier.networkTesting(inputArrays, answerArrays);
            } else if(choiceInt == 4) {
                //Test the network on training data
                //Inputs data into the inputArrays and answerArrays
                double[][] inputArrays = new double[60000][784];
                inputArrays = DigitIdentifier.inputDataInputs("mnist_train.csv", inputArrays);
                int[][] answerArrays = new int[60000][10];
                answerArrays = DigitIdentifier.inputDataAnswers("mnist_train.csv", answerArrays);

                //Tests the network.
                DigitIdentifier.networkTesting(inputArrays, answerArrays);
            } else if(choiceInt == 5 || choiceInt == 6) {
                //Choice to run through each test image. Choice 5 is to run through every image in the testing data set, 
                //while Choice 6 is to see only the incorrect images
                double[][] inputArrays = new double[10000][784];
                inputArrays = DigitIdentifier.inputDataInputs("mnist_test.csv", inputArrays);
                int[][] answerArrays = new int[10000][10];
                answerArrays = DigitIdentifier.inputDataAnswers("mnist_test.csv", answerArrays);

                if(choiceInt == 5){
                    DigitIdentifier.networkTestWithASCII(inputArrays, answerArrays, false, myScan);
                } else {
                    DigitIdentifier.networkTestWithASCII(inputArrays, answerArrays, true, myScan);
                }
            } else if(choiceInt == 7) {
                //Saves current weights and biases to a save file
                DigitIdentifier.saveToFile();
            } else {
                //Ends the program
                running = false;
            }
        } 
        myScan.close();
        System.out.println("Goodbye!");  
    }

    //Input default activation values for subsequent layers into neural network.
    public static void initNeuralNetwork(){
        //Input default values for hidden layer.
        DigitIdentifier.neuralNetwork[1] = new double[DigitIdentifier.L1neurons];
        for(int j = 0; j < DigitIdentifier.L1neurons; j++){
            DigitIdentifier.neuralNetwork[1][j] = 0;
        }
        //Input default values for hidden layer.
        DigitIdentifier.neuralNetwork[2] = new double[DigitIdentifier.L2neurons];
        for(int k = 0; k < DigitIdentifier.L2neurons; k++){
            DigitIdentifier.neuralNetwork[2][k] = 0;
        }
    }

    public static double[][] initRandWeights(double minW, double maxW, Random randD, double[][] weights){
        //Input random weights for a layer
        for(int row = 0; row < weights.length; row++){
            for(int col = 0; col < weights[row].length; col++){
                //Formula ensures double is within the allowed range
                weights[row][col] = minW + (randD.nextDouble() * (maxW - minW));
            }
        }
        return weights;
    }
    public static double[] initRandBiases(double minW, double maxW, Random randD, double[] biases){
        //Input random biases for a layer
        for(int i = 0; i < biases.length; i++){
            //Formula ensures double is within the allowed range
            biases[i] = minW + (randD.nextDouble() * (maxW - minW));
        }
        return biases;
    }

    public static double[][] inputDataInputs(String filename, double[][] inputArrays){
        //Input MNIST input values for inputArrays
        //Create bufferedReader
        String separator = ",";
        try{
        File file = new File(filename);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String[] strArray;

        //Iterate through each array in inputArrays
        for(int i = 0; i < inputArrays.length; i++){
            //Create string array of every value in line, iterate through the array
            strArray = br.readLine().split(separator);
            //Iterate through the pixels of the image (not index 0, because that is the correct classification of the image)
            for(int o = 1; o < strArray.length; o++){
                int currInt = Integer.valueOf(strArray[o]);
                //Intensity of color is divided by 255 to make input values the same range as the weights & biases, allowing for the network to learn.
                inputArrays[i][o - 1] = (double) (currInt) / 255.0;
            }   
        }
        br.close();
        } catch(IOException ioe) {
            ioe.printStackTrace();
        }
        return inputArrays;
    }
    public static int[][] inputDataAnswers(String filename, int[][] answerArrays){
        //Input MNIST input values for answerArrays
        //Create bufferedReader
        String separator = ",";
        try{
        File file = new File(filename);
        FileReader fr = new FileReader(file);
        BufferedReader br = new BufferedReader(fr);
        String[] strArray;

        //Iterate through each array in answerArrays
        for(int i = 0; i < answerArrays.length; i++){
            //Create string array of every value in line, iterate through the array
            strArray = br.readLine().split(separator);
            //Creates an array of length 10, one for each digit. Inputs a '1' for the classification. Used for backpropogation gradient creation.
            //Ex. If the answer is 5, a '1' is put into answerArrays[i][5]
            for(int index = 0; index < answerArrays[i].length; index++){
                if(Integer.valueOf(strArray[0]) == index){
                    answerArrays[i][index] = 1;
                }
                else{
                    answerArrays[i][index] = 0;
                }
            }   
        }
        br.close();
        } catch(IOException ioe) {
            ioe.printStackTrace();
        }
        return answerArrays;
    }

    //Trains the network using MNIST training data. Uses backpropagation to make the neural network learn.
    public static void networkTraining(double[][] inputArrays, int[][] answerArrays){
        //Iterate through 30 epochs
        for (int epoch = 0; epoch < epochs; epoch++){
            System.out.println("Epoch # " + epoch);
            //Create new randomized batches
            DigitIdentifier.randomizeBatches(inputArrays.length);
            //Create accuracy tracker
            int[] correctArray = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            int[] occurrenceArray = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
            //Iterate through all minibatches per epoch
            for(int batchNum = 0; batchNum < DigitIdentifier.batches.length; batchNum++){
                //Create/refresh bias and weight gradient arrays for each mini-batch
                DigitIdentifier.L2BiasGrad = new double[10];
                DigitIdentifier.L2WeightGrad = new double[10][50];
                DigitIdentifier.L1BiasGrad = new double[50];
                DigitIdentifier.L1WeightGrad = new double[50][784];
                //Iterate through each set of input data in mini-batch
                for(int inputSet = 0; inputSet < DigitIdentifier.batches[batchNum].length; inputSet++){
                    //Insert input data into first layer (Layer 0) of the neural network matrix
                    int inputIndex = DigitIdentifier.batches[batchNum][inputSet];
                    //Input layer of neural network is made into a double array of length 784, then values of each pixel are inputted
                    DigitIdentifier.neuralNetwork[0] = new double[inputArrays[inputIndex].length];
                    for(int i = 0; i < inputArrays[inputIndex].length; i++){
                        DigitIdentifier.neuralNetwork[0][i] = inputArrays[inputIndex][i];
                    }
                    //Add the correct answer to the occurrence array. Tracks how many time is number appears in the entire MNIST data set.
                    for(int i = 0; i < occurrenceArray.length; i++){
                        occurrenceArray[i] += answerArrays[inputIndex][i];
                    }

                    //Forward pass through neural network. Creates output.
                    DigitIdentifier.forwardPass();

                    //Check for accuracy. Compares the network results to the correct answer.
                    correctArray = isCorrect(answerArrays, inputIndex, correctArray);

                    //Backwards pass. Adds to the gradients of weights and biases to allow the network to make adjustments after the minibatch.
                    DigitIdentifier.backPropogation(answerArrays, inputIndex);
                }
                //After mini-batch, update the weights and biases using the gradients. Causes the network to learn.
                DigitIdentifier.updateWeightsAndBiases();
            }
            //Prints the accuracy of the network after each epoch
            DigitIdentifier.printEpochResults(occurrenceArray, correctArray, inputArrays.length);
        }
    }

    //Tests the neural network with the current weights and biases. Does not use batches or backpropagation for learning.
    public static void networkTesting(double[][] inputArrays, int[][] answerArrays){
        //Create accuracy tracker
        int[] correctArray = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        int[] occurrenceArray = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for(int inputSet = 0; inputSet < inputArrays.length; inputSet++){
            DigitIdentifier.neuralNetwork[0] = new double[inputArrays[inputSet].length];
            for(int i = 0; i < inputArrays[inputSet].length; i++){
                DigitIdentifier.neuralNetwork[0][i] = inputArrays[inputSet][i];
            }
            //Add the correct answer to the occurrence array
            for(int i = 0; i < occurrenceArray.length; i++){
                occurrenceArray[i] += answerArrays[inputSet][i];
            }
            //Forward pass through neural network. Creates output.
            DigitIdentifier.forwardPass();

            //Check for accuracy. Compares the network results to the correct answer.
            correctArray = isCorrect(answerArrays, inputSet, correctArray);
        }
        DigitIdentifier.printEpochResults(occurrenceArray, correctArray, inputArrays.length);
    }

    //Tests the network with testing data. This function iterates through each set of input data (each image) and shows the network classification,
    //the correct classification, whether the network got the correct answer, and ASCII art of the image in question. Depending on the 'onlyMissed' booleans,
    //the function will either show all input sets (image), or only the input sets that the network misclassified.
    public static void networkTestWithASCII(double[][] inputArrays, int[][] answerArrays, boolean onlyMissed, Scanner myScan){
        boolean scanOpen = true;
        String scanStr = "";
        //Create accuracy tracker
        int[] correctArray = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        int[] occurrenceArray = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for(int inputSet = 0; inputSet < inputArrays.length; inputSet++){
            DigitIdentifier.neuralNetwork[0] = new double[inputArrays[inputSet].length];
            for(int i = 0; i < inputArrays[inputSet].length; i++){
                DigitIdentifier.neuralNetwork[0][i] = inputArrays[inputSet][i];
            }
            //Add the correct answer to the occurrence array
            for(int i = 0; i < occurrenceArray.length; i++){
                occurrenceArray[i] += answerArrays[inputSet][i];
            }
            //Forward pass through neural network. Creates output.
            DigitIdentifier.forwardPass();

            //Check for accuracy. Allows this function to translate each classification from arrays of length 10 to single digits used for easier printing and comparison.
            correctArray = DigitIdentifier.isCorrect(answerArrays, inputSet, correctArray);
            boolean correctOutput = DigitIdentifier.correctBool(answerArrays, inputSet);
            //Finds the correct classification and converts it into a single digit.
            int classification = 0;
            for(int i = 0; i < answerArrays[inputSet].length; i++){
                if(answerArrays[inputSet][i] == 1){
                    classification = i;
                }
            }
            //Finds the network classification and converts it into a single digit.
            int greatestIndex = 0;
            double greatest = neuralNetwork[2][greatestIndex];
            for(int i = 1; i < neuralNetwork[2].length; i++){
                if(neuralNetwork[2][i] > greatest){
                    greatest = neuralNetwork[2][i];
                    greatestIndex = i;
                }
            }

            //Checks if the user is still iterating through the input sets.
            if(scanOpen){
                //If the user only wants to see the network's misclassifications (Option 6 in menu).
                if(onlyMissed){
                    //If the network misclassified the image
                    if(!correctOutput){
                        //Prints out the resulting data and the image in question.
                        System.out.println("Image #" + inputSet + ",\t Result: INCORRECT,\tCorrect Classification: " + classification + ",\tNetwork Classification: " + greatestIndex);
                        DigitIdentifier.printASCIIImage(inputArrays[inputSet]);

                        //Allows the user to stop iterating through each input set and skip to the end of the testing data to see the final results.
                        System.out.println("To continue, press Enter. If you wish to skip to the end results, enter anything.");
                        scanStr = myScan.nextLine();
                    }
                //If the user wants to see every input set (Option 5 in menu).
                } else {
                    String resultStr = "INCORRECT";
                    if(correctOutput){
                        resultStr = "CORRECT";
                    }
                    //Prints out the resulting data and the image in question.
                    System.out.println("Image #" + inputSet + ",\t Result: " + resultStr + ",\tCorrect Classification: " + classification + ",\tNetwork Classification: " + greatestIndex);
                    DigitIdentifier.printASCIIImage(inputArrays[inputSet]);

                    //Allows the user to stop iterating through each input set and skip to the end of the testing data to see the final results.
                    System.out.println("To continue, press Enter. If you wish to skip to the end results, enter anything.");
                    scanStr = myScan.nextLine();
                }
                //If the user does not enter nothing, stop scanning and skip to the end results of the testing data.
                if(scanStr != ""){
                    scanOpen = false;
                }
            }
        }
        //Print the final results of the network test.
        DigitIdentifier.printEpochResults(occurrenceArray, correctArray, inputArrays.length);
    }

    //Creates minibatches of randomized indexes that correspond to indexes of inputArrays and answerArrays.
    //Minibatches are used to more efficiently use backpropagation for faster learning.
    public static void randomizeBatches(int inputArraysLength){
        //Creates an ArrayList of indexes and shuffles them with no repeats.
        ArrayList<Integer> randomIndexes = new ArrayList<Integer>();
        for(int i = 0; i < inputArraysLength; i++){
            randomIndexes.add(i);
        }
        //Shuffles the contents of an array
        Collections.shuffle(randomIndexes);

        //Divide random indexes into batches of size miniBatchSize
        int indexTrack = 0;
        for(int row = 0; row < DigitIdentifier.batches.length; row++){
            for(int col = 0; col < DigitIdentifier.batches[row].length; col++){
                DigitIdentifier.batches[row][col] = randomIndexes.get(indexTrack);
                indexTrack++;
            }
        }
    }

    //Using current weights and biases, the network generates output on what it thinks the digit in the image is into an array of length 10, based on the input data.
    public static void forwardPass(){
        //Iterate through each layer of the neural network. Starts on hidden layer
        for(int currLayer = 1; currLayer < DigitIdentifier.neuralNetwork.length; currLayer++){
            //Iterate through each neuron in current layer
            for(int currNeuron = 0; currNeuron < DigitIdentifier.neuralNetwork[currLayer].length; currNeuron++){
                double summationTotal = 0;
                //Iterate through each input (activation) from previous layer
                for(int currInput = 0; currInput < DigitIdentifier.neuralNetwork[currLayer - 1].length; currInput++){
                    //Summation of weights * activations of previous layer
                    if(currLayer == 1){
                        summationTotal += (DigitIdentifier.L1Weights[currNeuron][currInput] * DigitIdentifier.neuralNetwork[currLayer - 1][currInput]); 
                    }
                    else{
                        summationTotal += (DigitIdentifier.L2Weights[currNeuron][currInput] * DigitIdentifier.neuralNetwork[currLayer - 1][currInput]);
                    }
                }
                //Adds bias of current neuron to summation
                if(currLayer == 1){
                    summationTotal += DigitIdentifier.L1Bias[currNeuron];
                }
                else{
                    summationTotal += DigitIdentifier.L2Bias[currNeuron];
                }
                //Apply sigmoid function and insert value into each neuron in the current layer
                DigitIdentifier.neuralNetwork[currLayer][currNeuron] = 1/(1 + Math.exp(-summationTotal));
            }
        }
    }

    //Checks if the network's classification is correct, and updates the correctArray.
    public static int[] isCorrect(int[][] answerArrays, int inputIndex, int[] correctArray){
        //Finds the highest value in the network's classification and its index.
        int resultIndex = 0;
        double resultValue = DigitIdentifier.neuralNetwork[2][0];
        for(int outNeuron = 1; outNeuron < DigitIdentifier.neuralNetwork[2].length; outNeuron++){
            if(DigitIdentifier.neuralNetwork[2][outNeuron] > resultValue){
                resultValue = DigitIdentifier.neuralNetwork[2][outNeuron];
                resultIndex = outNeuron;
            }
        }
        //If the network's classification matches the correct one, update the correctArray.
        for(int i = 0; i < answerArrays[inputIndex].length; i++){
            if((answerArrays[inputIndex][i] == 1) && (i == resultIndex)){
                correctArray[i] += 1;
            }
        }
        return correctArray;
    }

    //Similar function to isCorrect(), but return a boolean depending on the result. Used for networkTestWithASCII().
    public static boolean correctBool(int[][] answerArrays, int inputIndex){
        //Finds the highest value in the network's classification and its index.
        int resultIndex = 0;
        double resultValue = DigitIdentifier.neuralNetwork[2][0];
        for(int outNeuron = 1; outNeuron < DigitIdentifier.neuralNetwork[2].length; outNeuron++){
            if(DigitIdentifier.neuralNetwork[2][outNeuron] > resultValue){
                resultValue = DigitIdentifier.neuralNetwork[2][outNeuron];
                resultIndex = outNeuron;
            }
        }
        //If the network's classification matches the correct on, return true, otherwise return false.
        for(int i = 0; i < answerArrays[inputIndex].length; i++){
            if((answerArrays[inputIndex][i] == 1) && (i == resultIndex)){
                return true;
            }
        }
        return false;
    }

    //Allows the neural network to learn. By comparing the network's classification of the image to the correct answer, creates gradients that are distributed
    //to each weight and bias in the network. After each mini-batch, the weights and biases are adjusted using these gradients so the network learns.
    public static void backPropogation(int[][] answerArrays, int inputIndex){
        //Create bias gradient arrays for the current input set to calculate all weight and bias gradients.
        double[] L2TempBiasG = new double[10];
        double[] L1TempBiasG = new double[50];
        //Iterate through neural network layers
        for(int currLayer = 2; currLayer > 0; currLayer--){
            //If in output layer (Layer 2)
            if(currLayer == 2){
                //Update Layer 2 Bias Gradients. Compares the network's output to the correct answer.
                for(int outNeuron = 0; outNeuron < DigitIdentifier.neuralNetwork[currLayer].length; outNeuron++){
                    L2TempBiasG[outNeuron] = (DigitIdentifier.neuralNetwork[currLayer][outNeuron] - answerArrays[inputIndex][outNeuron]) * DigitIdentifier.neuralNetwork[currLayer][outNeuron] * (1 - DigitIdentifier.neuralNetwork[currLayer][outNeuron]);
                    DigitIdentifier.L2BiasGrad[outNeuron] += L2TempBiasG[outNeuron];
                }
            }
            //If in hidden layer (Layer 1)
            else{
                //Update Layer 1 Bias Gradients. Takes the summation of the weights from the neuron and bias gradient of the next layer, 
                //and applies the neuron's activation and adds it to the gradient.
                for(int currNeuron = 0; currNeuron < DigitIdentifier.neuralNetwork[currLayer].length; currNeuron++){
                    double summationTotal = 0;
                    for(int destNeuron = 0; destNeuron < DigitIdentifier.neuralNetwork[currLayer + 1].length; destNeuron++){
                        summationTotal += DigitIdentifier.L2Weights[destNeuron][currNeuron] * L2TempBiasG[destNeuron];
                    }
                    L1TempBiasG[currNeuron] = summationTotal * DigitIdentifier.neuralNetwork[currLayer][currNeuron] * (1 - DigitIdentifier.neuralNetwork[currLayer][currNeuron]);
                    DigitIdentifier.L1BiasGrad[currNeuron] += L1TempBiasG[currNeuron];
                }
            }
            //Update weight gradients. Each weight gradient multiplies the activation of the previous layer and the mias gradient of the neuron.
            for(int i = 0; i < DigitIdentifier.neuralNetwork[currLayer].length; i++){
                for(int o = 0; o < DigitIdentifier.neuralNetwork[currLayer - 1].length; o++){
                    if(currLayer == 2){
                        DigitIdentifier.L2WeightGrad[i][o] += DigitIdentifier.neuralNetwork[currLayer - 1][o] * L2TempBiasG[i];
                    }
                    else{
                        DigitIdentifier.L1WeightGrad[i][o] += DigitIdentifier.neuralNetwork[currLayer - 1][o] * L1TempBiasG[i];
                    }
                }
            }
        }
    }

    //Takes the current weight and bias gradients of the network and uses it to adjust the weights and biases.
    // Subtracts the summation of the weights or biases multiplied by (learning rate / mini-batch size) from the current weight or bias.
    public static void updateWeightsAndBiases(){
        //Update biases for Hidden Layer (Layer 1)
        for(int index = 0; index < DigitIdentifier.L1Bias.length; index++){
            DigitIdentifier.L1Bias[index] = DigitIdentifier.L1Bias[index] - ((DigitIdentifier.learningRate/DigitIdentifier.miniBatchSize) * DigitIdentifier.L1BiasGrad[index]);
        }
        //Update biases for Output Layer (Layer 2)
        for(int index = 0; index < DigitIdentifier.L2Bias.length; index++){
            DigitIdentifier.L2Bias[index] = DigitIdentifier.L2Bias[index] - ((DigitIdentifier.learningRate/DigitIdentifier.miniBatchSize) * DigitIdentifier.L2BiasGrad[index]);
        }
        //Update weights to Hidden Layer (Layer 1)
        for(int row = 0; row < DigitIdentifier.L1Weights.length; row++){
            for(int col = 0; col < DigitIdentifier.L1Weights[row].length; col++){
                DigitIdentifier.L1Weights[row][col] = DigitIdentifier.L1Weights[row][col] - ((DigitIdentifier.learningRate/DigitIdentifier.miniBatchSize) * DigitIdentifier.L1WeightGrad[row][col]);
            }
        }
        //Update weights to Output Layer (Layer 2)
        for(int row = 0; row < DigitIdentifier.L2Weights.length; row++){
            for(int col = 0; col < DigitIdentifier.L2Weights[row].length; col++){
                DigitIdentifier.L2Weights[row][col] = DigitIdentifier.L2Weights[row][col] - ((DigitIdentifier.learningRate/DigitIdentifier.miniBatchSize) * DigitIdentifier.L2WeightGrad[row][col]);
            }
        }
    }

    //Prints the results of the network going through an entire data set (epoch in training). Prints the accuracy for each digit and overall accuracy.
    public static void printEpochResults(int[] occurrenceArray, int[] correctArray, int divisor){
        int totalCorrect = 0;
        for(int index = 0; index < occurrenceArray.length; index++){
            totalCorrect += correctArray[index];
            System.out.print(index + " = " + correctArray[index] + "/" + occurrenceArray[index] + "\t");
            if(index == 4){
                System.out.print("\n");
            }
        }
        System.out.println();
        double percentage = ((double) totalCorrect / (double) divisor) * 100;
        System.out.println("Accuracy = " + totalCorrect + "/" + divisor + " = " + percentage + "%");
    }

    //Saves the current weights and biases to a .txt file. Saves in the order: L1Weights, L2Weights, L1Bias, L2Bias.
    public static void saveToFile(){
        try{
            PrintWriter writer = new PrintWriter("networksettings.txt", "UTF-8");
            //Upload weights to Hidden Layer
            for(int i = 0; i < DigitIdentifier.L1Weights.length; i++){
                for(int o = 0; o < DigitIdentifier.L1Weights[i].length; o++){
                    writer.println(DigitIdentifier.L1Weights[i][o]);
                }
            }
            //Upload weights to Output Layer
            for(int i = 0; i < DigitIdentifier.L2Weights.length; i++){
                for(int o = 0; o < DigitIdentifier.L2Weights[i].length; o++){
                    writer.println(DigitIdentifier.L2Weights[i][o]);
                }
            }
            //Upload biases of Hidden Layer
            for(int i = 0; i < DigitIdentifier.L1Bias.length; i++){
                writer.println(DigitIdentifier.L1Bias[i]);
            }
            //Upload biases of Output Layer
            for(int i = 0; i < DigitIdentifier.L2Bias.length; i++){
                writer.println(DigitIdentifier.L2Bias[i]);
            }
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Successfully saved to file.");
    }

    //Takes weights and biases from a .txt file and inputs them into the neural network. Inputs in the order: L1Weights, L2Weights, L1Bias, L2Bias.
    public static void uploadFromFile(){
        try{
            BufferedReader reader = new BufferedReader(new FileReader("networksettings.txt"));
            for(int i = 0; i < DigitIdentifier.L1Weights.length; i++){
                for(int o = 0; o < DigitIdentifier.L1Weights[i].length; o++){
                    double value = Double.valueOf(reader.readLine());
                    DigitIdentifier.L1Weights[i][o] = value;
                }
            }

            for(int i = 0; i < DigitIdentifier.L2Weights.length; i++){
                for(int o = 0; o < DigitIdentifier.L2Weights[i].length; o++){
                    double value = Double.valueOf(reader.readLine());
                    DigitIdentifier.L2Weights[i][o] = value;
                }
            }

            for(int i = 0; i < DigitIdentifier.L1Bias.length; i++){
                double value = Double.valueOf(reader.readLine());
                DigitIdentifier.L1Bias[i] = value;
            }

            for(int i = 0; i < DigitIdentifier.L2Bias.length; i++){
                double value = Double.valueOf(reader.readLine());
                DigitIdentifier.L2Bias[i] = value;
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Successfully uploaded.");
    }

    //Used for networkTestWithASCII(). Prints the image of the current input set with ASCII art. 
    public static void printASCIIImage(double[] inputArray){
        char c = '*';
        int index = 0;
        for(int row = 0; row < 28; row++){
            for(int col = 0; col < 28; col++){
                //Changes the character based on the intensity of the color white in the monochrome image.
                if(inputArray[index] > 0.9){
                    c = 'A';
                } else if(inputArray[index] > 0.8){
                    c = 'S';
                } else if(inputArray[index] > 0.7){
                    c = 't';
                } else if(inputArray[index] > 0.6){
                    c = 'q';
                } else if(inputArray[index] > 0.5){
                    c = 'n';
                } else if(inputArray[index] > 0.4){
                    c = 'u';
                } else if(inputArray[index] > 0.3){
                    c = 's';
                } else if(inputArray[index] > 0.2){
                    c = 'a';
                } else if(inputArray[index] > 0.1){
                    c = 'i';
                } else {
                    c = '.';
                }

                //If the color of the pixel is not black, print the current character. If it is, print nothing.
                if(inputArray[index] != 0){
                    System.out.print(c);
                } else {
                    System.out.print(" ");
                }
                index++;
            }
            System.out.print("\n");
        }
    }
}
