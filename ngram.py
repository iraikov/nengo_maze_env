import numpy as np
from itertools import product

def predict_ngram(activity, ngram_model, n_labels, n):
    """
    Predicts between ``n_labels`` using ``ngram_model``.
    :param activity: Spike activity of shape ``(n_examples, time, n_neurons)``.
    :param ngram_model: Previously recorded ngram score model.
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
                
        if len(this_example_orders_per_step) > 0:
            this_example_order = np.concatenate(this_example_orders_per_step)
            # Consider all n-gram sequences.
            for j in range(len(this_example_order) - n):
                sequence = tuple(this_example_order[j:j + n])
                if sequence in ngram_model:
                    score += ngram_model[sequence]

        predictions.append(np.argmax(score))
        
    return predictions


def fit_ngram_model(activity, labels, n_labels, n, ngram_model):
    """
    Fits ngram scores model by adding the count of each firing sequence of length n from the past ``n_examples``.

    :param activity: Firing activity of shape ``(n_examples, time, n_neurons)``.
    :param labels: The ground truth labels of shape ``(n_examples)``.
    :param n_labels: The number of target labels in the data.
    :param n: The max size of n-gram to use.
    :param ngram_model: Previously recorded scores to update.
    :return: Dictionary mapping n-grams to vectors of per-class unit activity.
    """

    n_steps, n_units = activity[0].shape
    for i, this_example_activity in enumerate(activity):

        this_example_orders_per_step = []
        for step in range(n_steps):
            step_nz = np.nonzero(this_example_activity[step])[0]
            if len(step_nz) > 0:
                step_order = step_nz[np.argsort(-this_example_activity[step][step_nz])]
                this_example_orders_per_step.append(step_order)

        for order in zip(*(this_example_orders_per_step[k:] for k in range(n))):
            for sequence in product(*order):
                if sequence not in ngram_model:
                    ngram_model[sequence] = np.zeros(n_labels)

                ngram_model[sequence][int(labels[i])] += 1

    return ngram_model

def predict_ngram_rates(activity, ngram_model, n_labels, n):
    """
    Predicts between ``n_labels`` using ``ngram_model``.
    :param activity: Spike activity of shape ``(n_examples, time, n_neurons)``.
    :param ngram_model: Previously recorded ngram score model.
    :param n_labels: The number of target labels in the data.
    :param n: The max size of n-gram to use.
    :return: Predictions per example.
    """
    n_examples = len(activity)
    n_steps, n_units = activity[0].shape
    
    predictions = []
    for this_example_activity in activity:
        
        score = np.zeros(n_labels)

        # Aggregate all of the firing neurons' indices
        step_nz = np.nonzero(np.any(this_example_activity > 0.,axis=0))[0]
        if len(step_nz) > 0:
            this_example_order = step_nz[np.argsort(-np.argmax(this_example_activity[:,step_nz],axis=0))]
        else:
            this_example_order = None

        if this_example_order is not None:
            for j in range(len(this_example_order) - n):
                sequence = tuple(this_example_order[j:j + n])
                if sequence in ngram_model:
                    score += ngram_model[sequence]

        predictions.append(np.argmax(score))
        
    return predictions


def fit_ngram_model_rates(activity, labels, n_labels, n, ngram_model):
    """
    Fits ngram scores model by adding the count of each firing sequence of length n from the past ``n_examples``.

    :param activity: Firing activity of shape ``(n_examples, time, n_neurons)``.
    :param labels: The ground truth labels of shape ``(n_examples)``.
    :param n_labels: The number of target labels in the data.
    :param n: The max size of n-gram to use.
    :param ngram_model: Previously recorded scores to update.
    :return: Dictionary mapping n-grams to vectors of per-class unit activity.
    """
    n_steps, n_units = activity[0].shape
    for i, this_example_activity in enumerate(activity):

        step_nz = np.nonzero(np.any(this_example_activity > 0.,axis=0))[0]
        if len(step_nz) > 0:
            this_example_order = step_nz[np.argsort(-np.argmax(this_example_activity[:,step_nz],axis=0))]
        else:
            this_example_order = None

        if this_example_order is not None:
            for j in range(len(this_example_order) - n):
                sequence = tuple(this_example_order[j:j + n])
                if sequence not in ngram_model:
                    ngram_model[sequence] = np.zeros(n_labels)

                ngram_model[sequence][int(labels[i])] += 1

    return ngram_model

