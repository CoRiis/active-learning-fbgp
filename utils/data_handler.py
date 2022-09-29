import numpy as np
import torch

from utils.transformations import transform


class Dataset:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class MyDataLoader:
    def __init__(self, args, oracle):
        self.args = args
        self.oracle = oracle
        self.pool_labeled = None
        self.oracle_labels = None
        self.search_space = None

        self.true_mean = None
        self.true_std = None
        self.true_mean_cp = None
        self.true_std_cp = None
        self.candidate_points = None

        self.train = Dataset(None, None)
        self.test = Dataset(None, None)
        self.train_trans = Dataset(None, None)
        self.test_trans = Dataset(None, None)

        # Dataloaders
        self.gp_train_loader_ori = None
        self.gp_test_loader_ori = None
        self.gp_train_loader = None
        self.gp_test_loader = None

        self.x_mu, self.x_sigma = None, None
        self.y_mu, self.y_sigma = None, None

    def get_initial_data(self, seed=None):
        self.search_space = self.oracle.search_space()
        train_x, train_y = self.oracle.sample_initial_data(n_samples=self.args.initial_samples,
                                                           space_filling_design=self.args.space_filling_design,
                                                           seed=seed)
        test_x, test_y = self.oracle.sample_initial_data(n_samples=self.args.test_samples,
                                                         space_filling_design="random",
                                                         seed=seed)

        self.train = Dataset(train_x, train_y)
        self.test = Dataset(test_x, test_y)

    def compute_true_mean_and_stddev(self):
        self.true_mean = torch.Tensor(self.oracle.mean(x=self.search_space))
        self.true_std = torch.Tensor(self.oracle.stddev(x=self.search_space))

    def transform(self):
        train_x_trans, self.x_mu, self.x_sigma = transform(self.train.x, method=self.args.transformation_x)
        train_y_trans, self.y_mu, self.y_sigma = transform(self.train.y, method=self.args.transformation_y)
        test_x_trans, _, _ = transform(self.test.x, self.x_mu, self.x_sigma, method=self.args.transformation_x)
        test_y_trans, _, _ = transform(self.test.y, self.y_mu, self.y_sigma, method=self.args.transformation_y)

        train_y_trans = train_y_trans.squeeze(-1)
        test_y_trans = test_y_trans.squeeze(-1)

        self.train_trans = Dataset(train_x_trans, train_y_trans)
        self.test_trans = Dataset(test_x_trans, test_y_trans)

    def make_dataloader(self):
        self.gp_train_loader = (self.train_trans.x, self.train_trans.y)
        self.gp_test_loader = (self.test_trans.x, self.test_trans.y)
        # Original space
        self.gp_train_loader_ori = (self.train.x, self.train.y)
        self.gp_test_loader_ori = (self.test.x, self.test.y)

    def get_candidate_points(self, seed):
        # Points of interest for querying
        candidate_points = self.search_space
        if self.search_space.shape[0] > 10000:
            rng = np.random.default_rng(seed=seed)
            subset_indices = rng.choice(np.arange(self.search_space.shape[0]), size=10000, replace=False)
            candidate_points = self.search_space[subset_indices]

            # recompute true mean and stddev
            toy_simulator = True
            if toy_simulator:
                self.true_mean_cp = self.true_mean[subset_indices]
                self.true_std_cp = self.true_std[subset_indices]
        else:
            self.true_std_cp = self.true_std
            self.true_mean_cp = self.true_mean

        self.candidate_points = candidate_points
