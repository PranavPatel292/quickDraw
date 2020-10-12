# BASIC REQUIREMENT(S)
# Python version: - 3.7
# PACKAGES
# Numpy, PILLOW-PIL
# the import statements are used to import the Numpy (for the mathematical calculation) and some drawing library to create the drawing board (poped at the end of the program to take user input)
# If you are confused about the Drawing Board check this link (https://www.youtube.com/watch?v=rriS3dYe02g) for Visual representation of working and use of drawing board in this project.
import numpy as np
from tkinter import *
from PIL import ImageTk, Image, ImageDraw
import PIL
loop_counter = 3    # a counter deciding the numbers of time the training should perform

# loading the data from the local device
data_airplane = np.load("./src/dataset/airplane_quickdraw.npy")
data_computer = np.load("./src/dataset/computer_quickdraw.npy")
data_line = np.load("./src/dataset/line_quickdaw.npy")
data_lighting = np.load("./src/dataset/lighting_quickdraw.npy")
data_birthday_cake = np.load("./src/dataset/birthdaycake_quickdraw.npy")
data_star = np.load("./src/dataset/star_quickdraw.npy")

# function for dividing the training and testing data (80 to 20 ration)
def divide_traing_testing(a1, a2, data):
    test_limit = (data.shape[0] * 80) / 100
    for i_ in range(len(data)):
        if i_ <= test_limit:
            # print(np.array(data[i_]))
            a1.append(data[i_] / 255.0)
            # print(a1[i_])
        else:
            a2.append(data[i_] / 255.0)
    return a1, a2


def make_training_array(a1, data, label, counter_output_neurons):
    for i_ in range(len(data)):
        a1.append([data[i_], label])
    counter_output_neurons += 1
    return a1, counter_output_neurons

# defining the sigmoid function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# defining the derivation of the sigmoid function
def derivate_sigmoid(x):
    return x * (1.0 - x)

# defining the loss function
def loss(x, y):
    a = []
    for i_ in range(len(x)):
        a.append(y[i_] - x[i_])
    return a


# this function is used for the ImageCreation for the first time, then after it will use to rewrite the Image;
def createImage():
    def paint(event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        c.create_oval(x1, y1, x2, y2, fill='black', width=16)
        draw.line([x1, y1, x2, y2], fill='black', width=10)

    def save():
        file_name = "./src/images/Google_Doodle_two.png"
        image1.thumbnail((128, 128))
        i1 = image1.resize((28, 28), Image.ANTIALIAS)
        i1.save(file_name)
        print("Image is created")

    root = Tk()
    c = Canvas(root, bg='white', height=300, width=400, cursor='pencil')
    c.bind("<B1-Motion>", paint)
    c.pack()
    image1 = PIL.Image.new("L", (400, 300), 255)
    draw = ImageDraw.Draw(image1)
    button = Button(text="Save", command=save)
    button.pack()
    root.mainloop()
    open1 = Image.open("./src/images/Google_Doodle_two.png")
    b = list(open1.getdata())
    m1 = np.reshape(b, (784, 1))
    m1 = (255 - m1) / 255.0
    return m1


print("Loading the data into the Python Environment!")

# the below section will load the data-set and divide it into train and test data!
print("Dividing the data into the training and testing sets")
data_airplane_test = []
data_airplane_training = []
data_airplane_training, data_airplane_test = divide_traing_testing(data_airplane_training, data_airplane_test,
                                                                   data_airplane)

data_computer_test = []
data_computer_training = []
data_computer_training, data_computer_test = divide_traing_testing(data_computer_training, data_computer_test,
                                                                   data_computer)

data_line_test = []
data_line_training = []
data_line_training, data_line_test = divide_traing_testing(data_line_training, data_line_test, data_line)

data_birthdaycake_test = []
data_birthdaycake_training = []
data_birthdaycake_training, data_birthdaycake_test = divide_traing_testing(data_birthdaycake_training,
                                                                           data_birthdaycake_test, data_birthday_cake)

# Here, I am not using this four data-sets!
training_data = []
test_data = []
output_label = []
counter_output_neurons = 0
x = 0


# splitting the data into one training_data for all the class
training_data, counter_output_neurons = make_training_array(training_data, data_airplane_training, 0,
                                                            counter_output_neurons)
training_data, counter_output_neurons = make_training_array(training_data, data_computer_training, 1,
                                                            counter_output_neurons)
training_data, counter_output_neurons = make_training_array(training_data, data_line_training, 2,
                                                            counter_output_neurons)
training_data, counter_output_neurons = make_training_array(training_data, data_birthdaycake_training, 3,
                                                            counter_output_neurons)

# combining the data into one test_data for all the class
test_data, x = make_training_array(test_data, data_airplane_test, 0, x)
test_data, x = make_training_array(test_data, data_computer_test, 1, x)
test_data, x = make_training_array(test_data, data_line_test, 2, x)
test_data, x = make_training_array(test_data, data_birthdaycake_test, 3, x)

# Print total numbers of training simples and testing simples as well as the numbers of input(s)
print("Number of training data simples: - ", len(training_data))  # 423139 total number of training data
print("Number of testing data simples: -", len(test_data))  # 105781 total number of testing data
print("Numbers of output nodes: -", counter_output_neurons)   # total number of output class (in this case 04)


# This for loop will pop to the use to take an label for the data-set(please make sure to enter in a correct order)
# mapping the output class to the user define input.
for i_ in range(0, counter_output_neurons):
    a = input("Enter your Label for the Data, Make sure it is in correct order => Aeroplane, Computer, Line and Cake")  # these 2 lines will create the output label and map it with the target line
    output_label.append(a)

# The actual training process will start from here!
for loop in range(0, loop_counter):   # this for loop does the entire training for 3 times
    if x == counter_output_neurons:
        input_neurons = 784  # defining the input size of the network as the data itself is the size of 784, one can confirm this by uncommenting the line 22.
        hidden_neurons_1 = 64  # defining the hidden layer size of the network
        hidden_neurons_2 = 64  # defining the hidden layer  size of the network
        output_neurons = counter_output_neurons  # defining the required output neurons
        LR = 0.01  # defining the learning rate;
        wih = np.random.uniform(-1, 1, size=(
        hidden_neurons_1, input_neurons))  # initialization of weights for the input to hidden layer 1
        whh = np.random.uniform(-1, 1, size=(
        hidden_neurons_2, hidden_neurons_1))  # initialization of weights for the hidden layer 1  to hidden layer 2
        who = np.random.uniform(-1, 1, size=(
        output_neurons, hidden_neurons_2))  # initialization of weights for the hidden layer 2 to output layer

        print("Training in Progress ")

        for e_ in range(0, 5):  # here, the model will take total of 5 epochs to train.
            print("Current number of epoch: - ", e_ + 1, "for Round:- ", loop + 1)
            for da_ in range(len(training_data)):
                c_error = []
                da_ = int(np.random.uniform(0, len(training_data)))
                m1 = np.reshape(training_data[da_][0],
                                (784, 1))  # resizing the data to fit appropriately as per the input layer
                hidden_01 = sigmoid(np.dot(wih, m1))
                hidden_02 = sigmoid(np.dot(whh, hidden_01))
                output = sigmoid(np.dot(who, hidden_02))
                index_ = training_data[da_][1]
                # print("Predication: - ", np.argmax(output),"Target: - ", index_)    # this print line can be uncomment in order to see the model predication and traget value
                for i_ in range(0, output_neurons):
                    if i_ == index_:
                        c_error.append(1)
                    else:
                        c_error.append(0)

                # Update the weight at output layer
                error_output = loss(output,
                                    c_error)  # first term of the equation of output to hidden layer 2 backpropagation (the loss)
                error_output = np.reshape(error_output, (output_neurons, 1))
                gradient_output = derivate_sigmoid(output)
                delta_error_output = gradient_output * error_output  # penultimate term of the equation of output to hidden layer 2 backpropagation (derivate term)
                f1 = delta_error_output * LR  # multiplying with LR
                delata_who = f1.dot(
                    hidden_02.T)  # last term of of the equation of output to hidden layer 2 backpropagation equation
                who += delata_who  # adding the newly calculated weight to the previous weight

                # Update the weight at hidden to hidden layer
                delta_hidden_2 = error_output.T.dot(
                    who)  # first term of the equation of hidden_layer_2 to hidden layer 1 backpropagation
                delta_hidden_2 = np.reshape(delta_hidden_2, (64, 1))
                gradient_hidden_2 = derivate_sigmoid(hidden_02)
                f = delta_hidden_2 * gradient_hidden_2  # penultimate term of the equation of hidden_layer_2 to hidden layer 1 backpropagation (derivate term)
                f1 = f * LR
                deltla_whh = f1.dot(
                    hidden_01.T)  # last term of of the equation of hidden_layer_2 to hidden layer 1 backpropagation equation
                whh += deltla_whh  # adding the newly calculated weight to the previous weight

                # Update Hidden to Input
                delta_hidden_1 = f.T.dot(whh)
                delta_hidden_1 = np.reshape(delta_hidden_1, (64, 1))
                gradient_hidden_1 = derivate_sigmoid(hidden_01)
                f = delta_hidden_1 * gradient_hidden_1
                f1 = f * LR
                delta_wih = f1.dot(m1.T)
                wih += delta_wih  # adding the newly calculated weight to the previous weight

        counter = 0     # this will use to calculate the accuracy for the testing data-set

        # Evaluating the train model on the testing data
        for da_ in range(len(test_data)):
            da_ = int(np.random.uniform(0, len(test_data)))
            m1 = np.reshape(test_data[da_][0], (784, 1))
            hidden_01 = sigmoid(np.dot(wih, m1))
            hidden_02 = sigmoid(np.dot(whh, hidden_01))
            output = sigmoid(np.dot(who, hidden_02))
            test1 = test_data[da_][1]
            test2 = np.argmax(output)

            if test1 == test2:
                counter += 1

        print("Round:- ", loop + 1, "Accuracy", (counter / float(len(test_data))) * 100)

    # Now from here on, the program will popup the drawing board to take user input! If you don not want, one can comment out the rest of the code.
    if loop == (loop_counter - 1):
        ch = input("Enter any Number: ")

        while ch != 0:
            user_image = createImage()
            # print(user_image)
            hidden_01 = sigmoid(np.dot(wih, user_image))
            hidden_02 = sigmoid(np.dot(whh, hidden_01))
            output = sigmoid(np.dot(who, hidden_02))
            test1 = np.argmax(output)
            print(output_label[test1])
            ch = input("Enter any number; 0 for exit")
