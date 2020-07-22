import numpy as np
import matplotlib.pyplot as plt
from simpleNN import NeuralNetwork
from simpleNN import load_data

net = NeuralNetwork([784, 30, 10])

data = load_data()

print('--- TRAINING ---')

training_data = data['train']

training_cost, training_accuracy, test_cost, test_accuracy = net.SGD(
    training_data=training_data,
    epochs=10,
    batch_size=10,
    learning_rate=0.25,
    test_data=data['test']
)

x = [i for i in range(1, len(training_cost)+1)]

plt.plot(x, training_cost)
plt.title('Training Cost')
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.show()

if len(test_cost) > 0:
    plt.plot(x, test_cost)
    plt.title('Test Cost')
    plt.ylabel('Cost')
    plt.xlabel('Epoch')
    plt.show()

training_accuracy = [
    float(a) / len(training_data) for a in training_accuracy
]

plt.plot(x, training_accuracy, label='training')
if len(test_accuracy) > 0:
    test_accuracy = [float(a) / len(data['test']) for a in test_accuracy]
    plt.plot(x, test_accuracy, label='test') 
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()

print('training accuracy: ', training_accuracy[-1])
print('test accuracy: ',  net.evaluate(data['test']) / len(data['test']))

print('--- PREDICTIONS ---')

def ascii_print(img):
    for row in img:
        line = ''
        for col in row:
            if col > 0:
                line += '@'
            else:
                line += '·'
        print(line)

for i in range(0, 10):
    img = np.array(data['validate'][i][0])
    img = img.reshape(28, 28)
    ascii_print(img)
    print('prediction ', np.argmax(net.predict(data['validate'][i][0])))
    print('actual ', data['validate'][i][1])
    print('---')
