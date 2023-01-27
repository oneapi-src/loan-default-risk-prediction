import pandas
import matplotlib.pyplot as plt
import numpy as np

def milify(number):
    milified = number
    milis = ['','K',"M","B"]
    for n,m in enumerate(milis):
        _milified = milified/10**(3*n)
        if _milified < 1:
            _milified = milified/10**(3*(n-1))
            return str(_milified)+milis[n-1]

def bar_comparison(stock_values,intel_values,stock_label='stock',intel_label='intel'
                            ,ax=None,title=None,xlabel=None,xticks=None,legend=False, relative=False):
    ax = ax or plt.gca()
    width = 0.35 
    x = np.arange(len(stock_values))
    
    if relative:
        intel_values = np.round(stock_values.values/intel_values.values,2)
        stock_values = np.ones(len(stock_values))
    rects1 = ax.bar(x - width/2, stock_values, width, label=stock_label, color='b')
    rects2 = ax.bar(x + width/2, intel_values, width, label=intel_label, color='deepskyblue')
    ax.bar_label(rects1, padding=3)
    if relative:
        ax.bar_label(rects2, labels=[str(i) + 'x' for i in intel_values], padding=3)
    else:
        ax.bar_label(rects2, padding=3)
    if title is not None:
        ax.title(title)
    if xticks is not None:
        ax.set_xticks(x, xticks)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if legend:
        ax.legend()