#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    # first approach just using list comprehension
    import timeit
    start = timeit.default_timer()
    cleaned_data = []
    cleaned_data = [(age, net_worth, (prediction-net_worth)**2)\
        for age, net_worth, prediction \
        in zip(ages, net_worths, predictions)]
    last_index = int(len(cleaned_data)*0.9-1)
    cleaned_data = sorted(cleaned_data, key = lambda t: t[2])
    cleaned_data = cleaned_data[0:last_index]
    stop = timeit.default_timer()
    runtime1 = stop - start

    # second approach. converting lists to numpy arrays, doing the math and
    # converting back to a list of tuples.
    start = timeit.default_timer()
    import numpy as np
    predictions_array = np.reshape(np.array(predictions), (len(predictions), 1))
    ages_array = np.reshape(np.array(ages), (len(ages), 1))
    net_worths_array = np.reshape(np.array(net_worths), (len(net_worths), 1))
    cleaned_data = np.concatenate((ages_array, net_worths_array, (predictions_array-net_worths_array)**2), axis=1)
    cleaned_data = cleaned_data[cleaned_data[:,2].argsort()]
    cleaned_data = cleaned_data[:last_index+1, :]
    #print "cleaned_data size:", cleaned_data.size, "dimens: ", cleaned_data.shape
    cleaned_data = [tuple(row) for row in cleaned_data]
    #print "length of cleaned_data:", len(cleaned_data)
    runtime2 = stop - start
    #print "runtime with lists: ", runtime1
    #print "runtime with np.arrays: ", runtime2
    #
    # The second approach with numpy arrays seems to work a bit faster.
    return cleaned_data
