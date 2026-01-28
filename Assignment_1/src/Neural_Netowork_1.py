from numpy import fix
import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision 
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time 

# Define the Neural Network class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input 
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x 

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, epochs):
    accuracy_history = []
    correctly_classified_images = []
    incorrectly_classified_images = []
    images_to_be_displayed = 5

    #Record the start time 
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Test the model and calculate accuracy
        model.eval()
        correct = 0 
        total = 0 

        with torch.no_grad():
            for inputs, labels, in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Track correct and incorrect classifications only after the last epoch
                if epoch == epochs - 1:
                    for i in range(labels.size(0)):
                        image = inputs[i].numpy().transpose((1, 2, 0))  # Convert to numpy and rearrange dimensions
                        actual_class = cifar_test.classes[labels[i]]
                        predicted_class = cifar_test.classes[predicted[i]]
                
                        if predicted[i] == labels[i] and len(correctly_classified_images) < images_to_be_displayed:
                            correctly_classified_images.append((image, actual_class, predicted_class))
                        elif predicted[i] != labels[i] and len(incorrectly_classified_images) < images_to_be_displayed:
                            incorrectly_classified_images.append((image, actual_class, predicted_class))


        accuracy = correct / total
        accuracy_history.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Accuracy on the test set: {100 * accuracy:.2f}%")
    
    #Record the end time 
    end_time = time.time()
    
    #Calculate and print the training time 
    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")

    #  display_images(correctly_classified_images, "Correctly Classified Examples")
    #display_images(incorrectly_classified_images,"Incorrectly Classified Examples")


    return accuracy_history

# Function to plot the accuracy history
def plot_accuracy(accuracy_history, title):
    plt.plot(range(1, len(accuracy_history) + 1), accuracy_history)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()


# Hyperparameters
input_size = 3 * 32 * 32
output_size = 10

criterion = nn.CrossEntropyLoss()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

#Downloading the dataset
cifar_train = datasets.CIFAR10(root='./data', train = True, download = True, transform = transform)
cifar_test = datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)

#Configuring the dataloader --> beneficial to use it, it helps to load and iterat over the dataset efficiently 
train_loader = torch.utils.data.DataLoader(cifar_train, batch_size = 64, shuffle = True, num_workers = 2)
test_loader = torch.utils.data.DataLoader(cifar_test, batch_size = 64, shuffle = False, num_workers = 2)



# Function to display images
def display_images(images, title):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(12, 4))

    for i, (image, actual_class, predicted_class) in enumerate(images):
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(f'Actual: {actual_class}\nPredicted: {predicted_class}')

    fig.suptitle(title)
    plt.show()



#*******************************README************************************
#COMMENT/UNCOMMENT AND MODIFY ANY LINES DOWN BELOW TO RUN SOME TESTS 



#Experiment with different hidden layer sizes
#hidden_sizes = [400, 500]

#for hidden_size in hidden_sizes:
#  model = NeuralNetwork(input_size, hidden_size, output_size)
#   optimizer = optim.SGD(model.parameters(), lr=0.01)
#   accuracy_history = train_model(model, train_loader, criterion, optimizer, epochs=30)
#   plot_accuracy(accuracy_history, f'Accuracy with Hidden Size = {hidden_size}')
#
    
#Experiment with different learning rates 
#learning_rates = [0.001, 0.005, 0.01, 0.015, 0.02, 0.05, 0.1]

#for learning_rate in learning_rates:
#    model = NeuralNetwork(input_size, 300,output_size)
#   optimizer = optim.SGD(model.parameters(), lr = learning_rate)
#   accuracy_history = train_model(model, train_loader, criterion, optimizer, epochs = 30)
#   plot_accuracy(accuracy_history, f'Accuracy with learning rate = {learning_rate}')

#model = NeuralNetwork(input_size, 200, output_size)
#optimizer = optim.SGD(model.parameters(), lr = 0.01)
#accuracy_history = train_model(model, train_loader, criterion, optimizer, epochs =40)
#plot_accuracy(accuracy_history, f'Accuracy with lr = 0.01, 200 hidden neurons, 40 epochs')

model = NeuralNetwork(input_size, 200, output_size)
optimizer = optim.SGD(model.parameters(), lr = 0.015)
accuracy_history = train_model(model, train_loader, criterion, optimizer, epochs=40)
plot_accuracy(accuracy_history, f'Accuracy with learning rate = 0,015, hidden_size = 200')
