import numpy as np
from itertools import product
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from scipy.spatial import cKDTree

def fit_logistic_decoder(activity, labels, n_labels):

    n_steps, n_units = activity[0].shape
    X = np.zeros((len(labels), n_units))
    for i, this_example_activity in enumerate(activity):
        X[i,:] = np.sum(this_example_activity, axis=0)

    param_grid = {
        "pca__n_components": range(10, 150, 10),
        "logisticregression__C": np.logspace(-4, 4, 4)
    }
        
    pca = PCA()
    scaler = StandardScaler()
    reg_model = LogisticRegression(tol=0.01, penalty='l1', solver='saga')
    ppl = make_pipeline(scaler, pca, reg_model)
    clf = GridSearchCV(ppl, param_grid, n_jobs=-1)
    clf.fit(X, labels)

    return clf

def predict_logistic(activity, decoder):

    n_steps, n_units = activity[0].shape
    X = np.zeros((len(labels), n_units))
    for i, this_example_activity in enumerate(activity):
        X[i] = np.mean(this_example_activity, axis=0)

    return clf.predict(X)



def fit_rate_decoder(activity, labels, n_labels, rate_decoder, ncap=20):

    n_steps, n_units = activity[0].shape
    for i, this_example_activity in enumerate(activity):
        rate_sum = np.sum(this_example_activity, axis=0)
        sequence = tuple(sorted(np.argsort(rate_sum)[::-1][:ncap]))
        if sequence not in rate_decoder:
            rate_decoder[sequence] = np.zeros(n_labels)

        rate_decoder[sequence][int(labels[i])] += 1

    kdt_matrix = np.vstack(tuple(rate_decoder.keys()))
    print(kdt_matrix.shape)
    kdt = cKDTree(kdt_matrix)
    
    return rate_decoder, kdt, kdt_matrix


def predict_rate(activity, rate_decoder, kdt, kdt_matrix, n_labels, ncap=20):
    n_examples = len(activity)
    n_steps, n_units = activity[0].shape
    
    predictions = []
    for i, this_example_activity in enumerate(activity):
        
        score = np.zeros(n_labels)
        rate_sum = np.sum(this_example_activity, axis=0)
        sequence = tuple(sorted(np.argsort(rate_sum)[::-1][:ncap]))

        nn = kdt.query(sequence, k=1)[1]
        key = tuple(kdt_matrix[nn])
        score += rate_decoder[key]

        predictions.append(np.argmax(score))
        
    return predictions


def predict_ngram(activity, ngram_decoder, n_labels, n):
    """
    Predicts between ``n_labels`` using ``ngram_decoder``.
    :param activity: Spike activity of shape ``(n_examples, time, n_neurons)``.
    :param ngram_decoder: Previously recorded ngram score model.
    :param n_labels: The number of target labels in the data.
    :param n: The max size of n-gram to use.
    :return: Predictions per example.
    """
    n_examples = len(activity)
    n_steps, n_units = activity[0].shape
    
    predictions = []
    for i, this_example_activity in enumerate(activity):
        
        score = np.zeros(n_labels)

        # Aggregate all of the firing neurons' indices
        this_example_orders_per_step = []
        for step in range(n_steps):
            step_nz = np.nonzero(this_example_activity[step])[0]
            if len(step_nz) > 0:
                step_order = step_nz[np.argsort(-this_example_activity[step][step_nz])]
                this_example_orders_per_step.append(step_order)
                
        for order in zip(*(this_example_orders_per_step[k:] for k in range(n))):
            for sequence in product(*order):
                if sequence in ngram_decoder:
                    score += ngram_decoder[sequence]

        predictions.append(np.argmax(score))
        
    return predictions


def fit_ngram_decoder(activity, labels, n_labels, n, ngram_decoder, dropout=None):
    """
    Fits ngram scores model by adding the count of each firing sequence of length n from the past ``n_examples``.

    :param activity: Firing activity of shape ``(n_examples, time, n_neurons)``.
    :param labels: The ground truth labels of shape ``(n_examples)``.
    :param n_labels: The number of target labels in the data.
    :param n: The max size of n-gram to use.
    :param ngram_decoder: Previously recorded scores to update.
    :return: Dictionary mapping n-grams to vectors of per-class unit activity.
    """

    n_steps, n_units = activity[0].shape
    act_units = {}
    
    for i, this_example_activity in enumerate(activity):

        inv_labels = np.ones(n_labels)
        inv_labels[int(labels[i])] = 0.
        inv_labels_inds = np.nonzero(inv_labels)[0]

        this_example_orders_per_step = []
        for step in range(n_steps):
            step_nz = np.nonzero(this_example_activity[step])[0]
            if len(step_nz) > 0:
                step_order = step_nz[np.argsort(-this_example_activity[step][step_nz])]
                n_order = len(step_order)
                for u in step_order:
                    n_act = act_units.get(u, 0)
                    act_units[u] = n_act + 1
                if (n_order > 1) and (dropout is not None) and (dropout > 0.0):
                    n_choice = int(round(n_order*dropout))
                    n_acts = np.asarray([act_units.get(u, 0) for u in step_order], dtype=np.float32)
                    sum_acts = np.sum(n_acts)
                    prob_acts = None
                    if sum_acts > 0.:
                        prob_acts = n_acts / np.sum(n_acts)
                    dropout_selection = np.random.choice(range(n_order), size=n_choice, p=prob_acts, replace=False)
                    step_order = np.delete(step_order, dropout_selection)
                this_example_orders_per_step.append(step_order)
                
        for order in zip(*(this_example_orders_per_step[k:] for k in range(n))):
            for sequence in product(*order):
                if sequence not in ngram_decoder:
                    ngram_decoder[sequence] = np.zeros(n_labels)

                ngram_decoder[sequence][int(labels[i])] += 1
                ngram_decoder[sequence][inv_labels_inds] -= 2.0

    return ngram_decoder

