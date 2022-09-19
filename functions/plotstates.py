import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plotTimeSeries(Q, hidden_states, ylabel):
 
    sns.set_theme(style='white')
    fig = plt.figure()
    ax = fig.add_subplot(111)
 
    xs = np.arange(len(Q))+1909
    masks = hidden_states == 0
    ax.scatter(xs[masks], Q[masks], c='r', label='Dry State')
    masks = hidden_states == 1
    ax.scatter(xs[masks], Q[masks], c='b', label='Wet State')
    ax.plot(xs, Q, c='k')
     
    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    fig.subplots_adjust(bottom=0.2)
    handles, labels = plt.gca().get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=True)
 
    return None
 