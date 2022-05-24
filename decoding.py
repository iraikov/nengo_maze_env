import numpy as np
from itertools import product


def fit_rank_order_decoder(activity, labels, n_labels, n, rank_order_decoder):
    """
    Fits rank order model by sorting each neuron according to latency of first spike.

    :param activity: Firing activity of shape ``(n_examples, time, n_neurons)``.
    :param labels: The ground truth labels of shape ``(n_examples)``.
    :param n_labels: The number of target labels in the data.
    :param n: The max size of n-gram to use.
    :param rank_order_model: Previously recorded rank orders to update.
    :return: Dictionary mapping rank orders to vectors of per-class unit activity.
    """

    n_steps, n_units = activity[0].shape
    for i, this_example_activity in enumerate(activity):

        unit_act, unit_no = np.nonzero(this_example_activity)
        u, ind, ct = np.unique(unit_no, return_index=True, return_counts=True)
        rank_order = np.argsort(ind)
        activity_order = u[rank_order]
        
        for j in range(len(activity_order) - n):
            sequence = tuple(activity_order[j:j + n])
            if sequence not in rank_order_decoder:
                rank_order_decoder[sequence] = np.zeros(n_labels)

            rank_order_decoder[sequence][int(labels[i])] += 1

    return rank_order_decoder


def predict_rank_order(activity, rank_order_decoder, n_labels, n):
    """
    Predicts between ``n_labels`` using ``rank_order_decoder``.
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

        unit_act, unit_no = np.nonzero(this_example_activity)
        u, ind, ct = np.unique(unit_no, return_index=True, return_counts=True)
        rank_order = np.argsort(ind)
        activity_order = u[rank_order]
                
        for j in range(len(activity_order) - n):
            sequence = tuple(activity_order[j:j + n])
            if sequence in rank_order_decoder:
                score += rank_order_decoder[sequence]

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
                
        if len(this_example_orders_per_step) > 0:
            this_example_order = np.concatenate(this_example_orders_per_step)
            # Consider all n-gram sequences.
            for j in range(len(this_example_order) - n):
                sequence = tuple(this_example_order[j:j + n])
                if sequence in ngram_decoder:
                    score += ngram_decoder[sequence]

        predictions.append(np.argmax(score))
        
    return predictions


def fit_ngram_decoder(activity, labels, n_labels, n, ngram_decoder):
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
    for i, this_example_activity in enumerate(activity):

        this_example_orders_per_step = []
        for step in range(n_steps):
            step_nz = np.nonzero(this_example_activity[step])[0]
            if len(step_nz) > 0:
                step_order = step_nz[np.argsort(-this_example_activity[step][step_nz])]
                this_example_orders_per_step.append(step_order)

        for order in zip(*(this_example_orders_per_step[k:] for k in range(n))):
            for sequence in product(*order):
                if sequence not in ngram_decoder:
                    ngram_decoder[sequence] = np.zeros(n_labels)

                ngram_decoder[sequence][int(labels[i])] += 1

    return ngram_decoder

def predict_ngram_rates(activity, ngram_decoder, n_labels, n):
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
                if sequence in ngram_decoder:
                    score += ngram_decoder[sequence]

        predictions.append(np.argmax(score))
        
    return predictions


def fit_ngram_decoder_rates(activity, labels, n_labels, n, ngram_decoder):
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
    for i, this_example_activity in enumerate(activity):

        step_nz = np.nonzero(np.any(this_example_activity > 0.,axis=0))[0]
        if len(step_nz) > 0:
            this_example_order = step_nz[np.argsort(-np.argmax(this_example_activity[:,step_nz],axis=0))]
        else:
            this_example_order = None

        if this_example_order is not None:
            for j in range(len(this_example_order) - n):
                sequence = tuple(this_example_order[j:j + n])
                if sequence not in ngram_decoder:
                    ngram_decoder[sequence] = np.zeros(n_labels)

                ngram_decoder[sequence][int(labels[i])] += 1

    return ngram_decoder

