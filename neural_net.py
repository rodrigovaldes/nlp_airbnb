
import torch;
import torchvision;
import torch.nn as nn;
import torch.nn.functional as F;
import numpy as np;
import torch.optim as optim;
import re;

EMBEDDING_SIZE = 300;
HIDDEN_WIDTH = 256;
CV_WIDTH = 10 + 1;
#LR = 0.00001;
LR = 0.00001;
WD = 0#0.0001;

NUM_ITER_CHECK = 50;

def load_vector(path, limit):
    data = open(path, "r");
    listings = [];
    scores = [];
    vectors = np.empty((0));
    i = 0;
    for line in data:
        line = line.rstrip();
        tokens = re.split(r'\t+', line);
        listings.append(tokens.pop(0));
        scores.append(int(float(tokens.pop(0))));
        tokens = re.split(r' ', tokens[0]);
        vector = list(map(float, tokens));
        vector = np.resize(vector, (EMBEDDING_SIZE));
        #s = np.sum(vector);
        #if s > 0:
        #    vector = vector / s;
        vectors = np.concatenate((vectors, vector));
        i += 1;
        if limit > 0 and i >= limit:
            break;
    data.close();
    print("Loaded ", i, "vectors");
    scores = np.asarray(scores);
    scores = torch.from_numpy(scores).long();
    vectors = np.reshape(vectors, (-1, EMBEDDING_SIZE));
    vectors = torch.from_numpy(vectors).double();
    return listings, scores, vectors;

def test_classify(net, test_vectors, test_labels):
    num_test = 0.0;
    correct = 0.0;
    with torch.no_grad():
        data = net(test_vectors);
        #print(data[0]);
        _, predicted = torch.max(net(test_vectors), 1);
        num_test += len(test_labels);
        correct += (predicted == test_labels).sum().item();
    return correct / num_test;



# The whole process starts from here

seed = 128;
rng = np.random.RandomState(seed);
torch.manual_seed(seed);

print("Load train vectors");
_, train_scores, train_vectors = load_vector("results/vector_train.txt", -1);
print("Load dev vectors");
_, dev_scores, dev_vectors = load_vector("results/vector_dev.txt", -1);
print("Load test vectors");
_, test_scores, test_vectors = load_vector("results/vector_test.txt", -1);

neural_net = torch.nn.Sequential(
            torch.nn.Linear(EMBEDDING_SIZE, HIDDEN_WIDTH),
            torch.nn.Tanh(),
            #torch.nn.ReLU(),
            #torch.nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH),
            #torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_WIDTH, CV_WIDTH),
            torch.nn.Softmax(1)
        ).double();

best_neural_net = torch.nn.Sequential(
            torch.nn.Linear(EMBEDDING_SIZE, HIDDEN_WIDTH),
            torch.nn.Tanh(),
            #torch.nn.ReLU(),
            #torch.nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH),
            #torch.nn.ReLU(),
            torch.nn.Linear(HIDDEN_WIDTH, CV_WIDTH),
            torch.nn.Softmax(1)
        ).double();

criterion = nn.CrossEntropyLoss();
optimizer = optim.SGD(neural_net.parameters(), lr = LR, momentum = 0.9, weight_decay=WD);

num_epochs = 100;
batch_size = 10;

train_batch_vectors = torch.split(train_vectors, batch_size);
print(len(train_batch_vectors));
print(train_batch_vectors[0].size());
train_batch_scores = torch.split(train_scores, batch_size);
print(len(train_batch_scores));
print(train_batch_scores[0].size());

# Train FFNN
num_iter = 0;
ha = -1;
for epoch in range(num_epochs):
    print('Epoch', epoch);
    for i in range(len(train_batch_vectors)):
        # get data
        x = train_batch_vectors[i];
        labels = train_batch_scores[i];

        # zero the parameter gradients
        optimizer.zero_grad();

        # forward + backward + optimize
        y = neural_net(x);
        loss = criterion(y, labels);
        loss.backward();
        optimizer.step();
        
        # print statistics
        num_iter += 1;
        if num_iter == NUM_ITER_CHECK:
            num_iter = 0;
            accuracy = test_classify(neural_net, dev_vectors, dev_scores);
            if (accuracy > ha):
                print("   --> Loss: %.7f Accuracy: %.4f%%" % (loss, accuracy * 100));
                ha = accuracy;
                best_neural_net.load_state_dict(neural_net.state_dict());

accuracy = test_classify(best_neural_net, test_vectors, test_scores);
print("Accuracy of the best Feed-forward Neural Network on TESTDEV: %.4f%%" % (accuracy * 100));





