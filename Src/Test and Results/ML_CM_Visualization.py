import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

def cm_collage(classifier):    
    f, axes = plt.subplots(1, 8, figsize=(20, 5), sharey='row')

    for i, (key, classifiers) in enumerate(classifier.items()):
        disp=ConfusionMatrixDisplay(classifiers)
        disp.plot(ax=axes[i])
        disp.ax_.set_title(key)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel('')
        if i!=0:
            disp.ax_.set_ylabel('')

    f.text(0.4, 0.1, 'Predicted label', ha='left')
    plt.subplots_adjust(wspace=0.40, hspace=0.1)


    f.colorbar(disp.im_, ax=axes)
    plt.show()

    classifier={
    'SVM1_normal':cm_SVM1_normal,
    'SVM1_Stratified':cm_SVM1_Stratified,
    'SVM1_Systematic':cm_SVM1_Systematic,
    'SVM1_Cyclic':cm_SVM1_Cyclic,
    'SVM1_UT':cm_SVM1_ut,
    'SVM1_DT':cm_SVM1_dt,
    'SVM1_US':cm_SVM1_us,
    'SVM1_DS':cm_SVM1_ds,
}