import numpy as np
import torch
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import rmse_score

dataset = get_movielens_dataset(variant='1M')
print(dataset)



model = ExplicitFactorizationModel(loss='regression',
                                   embedding_dim=128,  # latent dimensionality
                                   n_iter=10,  # number of epochs of training
                                   batch_size=1024,  # minibatch size
                                   l2=1e-9,  # strength of L2 regularization
                                   learning_rate=1e-3,
                                   use_cuda=torch.cuda.is_available())

train, test = random_train_test_split(dataset, random_state=np.random.RandomState(42))

print('Split into \n {} and \n {}.'.format(train, test))
model.fit(train, verbose=True)

train_rmse = rmse_score(model, train)
test_rmse = rmse_score(model, test)

print('Train RMSE {:.3f}, test RMSE {:.3f}'.format(train_rmse, test_rmse))
