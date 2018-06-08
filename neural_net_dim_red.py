
import torch;
import torchvision;
import torch.nn as nn;
import torch.nn.functional as F;
import numpy as np;
import torch.optim as optim;
import re;
import pandas as pd

'''
--------------------------------------------------------------------------------
CONTROLS HERE
--------------------------------------------------------------------------------
'''
# Define Globals
EMBEDDING_SIZE = 512;
HIDDEN_WIDTH = 80; # No influence, I tried 80, and the original 256?
CV_WIDTH = 10 + 1;
#LR = 0.00001;
LR = 0.0001;
WD = 0#0.0001;
NUM_ITER_CHECK = 50;
ADD_SIZE = True; # Takes into account the size of the review in the vector
DIM_REDUCTION = True # Reduce the dimensions before optimize
DESCRIPTION = False # By default, if DESCRIPTION == False, you use a vector with
# size of the vector on it.
num_epochs = 100; # Change to 100!
batch_size = 10;

'''
--------------------------------------------------------------------------------
END OF CONTROLS
--------------------------------------------------------------------------------
'''

if DESCRIPTION == False:
    if ADD_SIZE:
        suffix = "_size_nd"
    else:
        suffix = "_nd"
elif DESCRIPTION:
    if ADD_SIZE:
        suffix = "_size"
    else:
        suffix = ""

if DIM_REDUCTION:
    EMBEDDING_SIZE = 900
    new_embedding_size = 300

# Load vector (some  optimiozations)
def load_vector_df(path, limit):
    '''
    '''
    vector_df = pd.read_csv(path, sep = "\t", header = None)
    # listings
    listings = list(vector_df[0])
    # Scores
    scores = torch.from_numpy(np.array(vector_df[1])).long()

    # Vectors
    matrix_df = vector_df[2].str.split(" ", expand = True)
    subset_matrix_df = matrix_df[np.arange(0,EMBEDDING_SIZE)].as_matrix()
    subset_matrix_np = subset_matrix_df.astype(np.float)
    vectors = torch.from_numpy(subset_matrix_np).double()

    return listings, scores, vectors


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

# Find out which are misclassified
def summary_classified(net, test_vectors, test_labels):
    '''
    Generates two arrays. 1) Labels according the the data
    2) Predictions according to the model
    '''

    with torch.no_grad():
        data = net(test_vectors);
        _, predicted = torch.max(net(test_vectors), 1);
        mask_correct = (predicted == test_labels)
        good_real = test_labels[mask_correct]
        good_prediction = predicted[mask_correct]

        bad_real = test_labels[~mask_correct]
        bad_prediction = predicted[~mask_correct]

    return good_real, good_prediction, bad_real, bad_prediction

def gen_good_bad(real, pred, test_labels):

    RealVsPred = pd.DataFrame({'real': np.array(real), 'pred': np.array(pred)})
    RealVsPred['category_name'] = 1
    sum_rp = RealVsPred.groupby(['real', 'pred']).count().reset_index()

    y_real = pd.DataFrame({'real': np.array(test_labels), 'count': 1}
        ).groupby('real').count().reset_index()

    compiler = sum_rp.merge(y_real, on = 'real')
    compiler['ratio'] = compiler['category_name'] / compiler['count']

    return compiler

# Dimensionality reduction
def dimensionality_reduction_one_vec(vec, new_dim):
    '''
    '''
    size_vec = vec.shape[0]
    num_important = int(size_vec / new_dim)
    dim_transformer = (size_vec, new_dim)

    transformer = np.zeros(dim_transformer)

    for initial in range(new_dim):
        for j in range(num_important):
            transformer[num_important*initial + j, initial] = 1 / num_important

    return np.matmul(vec, transformer)

def reduce_size_vectors(vectors, new_len):
    '''
    '''
    new_vecs = np.zeros(0)
    for i, vec_item in enumerate(vectors):
        n_vec = dimensionality_reduction_one_vec(vec_item, new_len)
        if i == 0:
            new_vecs = np.insert(new_vecs, 0, n_vec)
        else:
            new_vecs = np.insert(new_vecs, -1, n_vec)

    new_vecs_aux = new_vecs.reshape((len(vectors), new_len))
    new_vecs = torch.from_numpy(new_vecs_aux)

    return new_vecs

# The whole process starts from here
from datetime import datetime
print("Load train vectors");
print(str(datetime.now()))
_, train_scores, train_vectors = load_vector_df("results/vector_train{}.txt".format(suffix), -1);
print("Load dev vectors");
print(str(datetime.now()))
_, dev_scores, dev_vectors = load_vector_df("results/vector_dev{}.txt".format(suffix), -1);
print("Load test vectors");
print(str(datetime.now()))

# from datetime import datetime
# print("1. START")
# print(str(datetime.now()))
# _, test_scores, test_vectors = load_vector("results/vector_test{}.txt".format(suffix), -1);
# print("1. END")
# print(str(datetime.now()))
# print("2. START")
# print(str(datetime.now()))
_, test_scores, test_vectors = load_vector_df("results/vector_test{}.txt".format(suffix), -1);
# print("2. END")
# print(str(datetime.now()))
print(str(datetime.now()))



## Make some transformations to the vectors
if DIM_REDUCTION:
    print("Reducing the size of train")
    print(str(datetime.now()))
    train_vectors = reduce_size_vectors(train_vectors, new_embedding_size)
    print("Reducing the size of dev")
    print(str(datetime.now()))
    dev_vectors = reduce_size_vectors(dev_vectors, new_embedding_size)
    print("Reducing the size of test")
    print(str(datetime.now()))
    test_vectors = reduce_size_vectors(test_vectors, new_embedding_size)
    EMBEDDING_SIZE = new_embedding_size
    print(str(datetime.now()))


seed = 128;
rng = np.random.RandomState(seed);
torch.manual_seed(seed);

# The output of neural net is the "classification"
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
        # loss.backward() computes dloss/dx for every parameter x
        loss.backward();
        # optimizer.step() performs parameter update
        optimizer.step();

        # print statistics
        num_iter += 1;
        if num_iter == NUM_ITER_CHECK:
            num_iter = 0;
            accuracy = test_classify(neural_net, dev_vectors, dev_scores);
            if (accuracy > ha):
                print("   --> Loss: %.7f Accuracy: %.4f%%" % (loss, accuracy * 100));
                ha = accuracy;
                # "Save" the best neural net (this is nicer than Tensorflow)
                best_neural_net.load_state_dict(neural_net.state_dict());

accuracy = test_classify(best_neural_net, test_vectors, test_scores);
print("Accuracy of the best Feed-forward Neural Network on TESTDEV: %.4f%%" % (accuracy * 100));
print(str(datetime.now()))

good_r, good_p, bad_r, bad_p = summary_classified(
    best_neural_net, test_vectors, test_scores)
compiler_good = gen_good_bad(good_r, good_p, test_scores)
print(compiler_good)
compiler_bad = gen_good_bad(bad_r, bad_p, test_scores)
print(compiler_bad)
# The biggest problem: 9 classified as 10
