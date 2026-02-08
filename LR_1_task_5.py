import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_curve,
    roc_auc_score
)

df = pd.read_csv('data.csv')
print(df.head())


thresh = 0.5
df['predicted_RF'] = (df.model_RF >= thresh).astype('int')
df['predicted_LR'] = (df.model_LR >= thresh).astype('int')
print(df.head())


def find_TP(y_true, y_pred):
    # true positives: y_true=1 and y_pred=1
    return int(np.sum((y_true == 1) & (y_pred == 1)))

def find_FN(y_true, y_pred):
    # false negatives: y_true=1 and y_pred=0
    return int(np.sum((y_true == 1) & (y_pred == 0)))

def find_FP(y_true, y_pred):
    # false positives: y_true=0 and y_pred=1
    return int(np.sum((y_true == 0) & (y_pred == 1)))

def find_TN(y_true, y_pred):
    # true negatives: y_true=0 and y_pred=0
    return int(np.sum((y_true == 0) & (y_pred == 0)))

y_true_rf = df.actual_label.values
y_pred_rf = df.predicted_RF.values
y_pred_lr = df.predicted_LR.values

print("Sklearn confusion_matrix (RF):")
print(confusion_matrix(y_true_rf, y_pred_rf))

print('TP:', find_TP(y_true_rf, y_pred_rf))
print('FN:', find_FN(y_true_rf, y_pred_rf))
print('FP:', find_FP(y_true_rf, y_pred_rf))
print('TN:', find_TN(y_true_rf, y_pred_rf))


def find_conf_matrix_values(y_true, y_pred):
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN


def Piatnytsia_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP],
                     [FN, TP]])

assert np.array_equal(
    Piatnytsia_confusion_matrix(y_true_rf, y_pred_rf),
    confusion_matrix(y_true_rf, y_pred_rf)
), 'Piatnytsia_confusion_matrix() is not correct for RF'

assert np.array_equal(
    Piatnytsia_confusion_matrix(y_true_rf, y_pred_lr),
    confusion_matrix(y_true_rf, y_pred_lr)
), 'Piatnytsia_confusion_matrix() is not correct for LR'

print("Piatnytsia_confusion_matrix (RF):")
print(Piatnytsia_confusion_matrix(y_true_rf, y_pred_rf))


def Piatnytsia_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

assert Piatnytsia_accuracy_score(y_true_rf, y_pred_rf) == accuracy_score(y_true_rf, y_pred_rf), \
    'Piatnytsia_accuracy_score failed on RF'
assert Piatnytsia_accuracy_score(y_true_rf, y_pred_lr) == accuracy_score(y_true_rf, y_pred_lr), \
    'Piatnytsia_accuracy_score failed on LR'

print('Accuracy RF: %.3f' % (Piatnytsia_accuracy_score(y_true_rf, y_pred_rf)))
print('Accuracy LR: %.3f' % (Piatnytsia_accuracy_score(y_true_rf, y_pred_lr)))

def Piatnytsia_recall_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    denom = TP + FN
    return TP / denom if denom != 0 else 0.0

assert Piatnytsia_recall_score(y_true_rf, y_pred_rf) == recall_score(y_true_rf, y_pred_rf), \
    'Piatnytsia_recall_score failed on RF'
assert Piatnytsia_recall_score(y_true_rf, y_pred_lr) == recall_score(y_true_rf, y_pred_lr), \
    'Piatnytsia_recall_score failed on LR'

print('Recall RF: %.3f' % (Piatnytsia_recall_score(y_true_rf, y_pred_rf)))
print('Recall LR: %.3f' % (Piatnytsia_recall_score(y_true_rf, y_pred_lr)))


def Piatnytsia_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    denom = TP + FP
    return TP / denom if denom != 0 else 0.0

assert Piatnytsia_precision_score(y_true_rf, y_pred_rf) == precision_score(y_true_rf, y_pred_rf), \
    'Piatnytsia_precision_score failed on RF'
assert Piatnytsia_precision_score(y_true_rf, y_pred_lr) == precision_score(y_true_rf, y_pred_lr), \
    'Piatnytsia_precision_score failed on LR'

print('Precision RF: %.3f' % (Piatnytsia_precision_score(y_true_rf, y_pred_rf)))
print('Precision LR: %.3f' % (Piatnytsia_precision_score(y_true_rf, y_pred_lr)))


def Piatnytsia_f1_score(y_true, y_pred):
    r = Piatnytsia_recall_score(y_true, y_pred)
    p = Piatnytsia_precision_score(y_true, y_pred)
    denom = p + r
    return (2 * p * r / denom) if denom != 0 else 0.0

assert np.isclose(Piatnytsia_f1_score(y_true_rf, y_pred_rf), f1_score(y_true_rf, y_pred_rf)), \
    'Piatnytsia_f1_score failed on RF'
assert np.isclose(Piatnytsia_f1_score(y_true_rf, y_pred_lr), f1_score(y_true_rf, y_pred_lr)), \
    'Piatnytsia_f1_score failed on LR'

print('F1 RF: %.3f' % (Piatnytsia_f1_score(y_true_rf, y_pred_rf)))
print('F1 LR: %.3f' % (Piatnytsia_f1_score(y_true_rf, y_pred_lr)))


print('\nScores with threshold = 0.5 (RF)')
print('Accuracy RF: %.3f' % (Piatnytsia_accuracy_score(df.actual_label.values, df.predicted_RF.values)))
print('Recall RF: %.3f' % (Piatnytsia_recall_score(df.actual_label.values, df.predicted_RF.values)))
print('Precision RF: %.3f' % (Piatnytsia_precision_score(df.actual_label.values, df.predicted_RF.values)))
print('F1 RF: %.3f' % (Piatnytsia_f1_score(df.actual_label.values, df.predicted_RF.values)))

print('\nScores with threshold = 0.25 (RF)')
pred_rf_025 = (df.model_RF >= 0.25).astype('int').values
print('Accuracy RF: %.3f' % (Piatnytsia_accuracy_score(df.actual_label.values, pred_rf_025)))
print('Recall RF: %.3f' % (Piatnytsia_recall_score(df.actual_label.values, pred_rf_025)))
print('Precision RF: %.3f' % (Piatnytsia_precision_score(df.actual_label.values, pred_rf_025)))
print('F1 RF: %.3f' % (Piatnytsia_f1_score(df.actual_label.values, pred_rf_025)))


fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)


plt.figure()
plt.plot(fpr_RF, tpr_RF, 'r-', label='RF')
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR')
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)

print('AUC RF: %.3f' % auc_RF)
print('AUC LR: %.3f' % auc_LR)

plt.figure()
plt.plot(fpr_RF, tpr_RF, 'r-', label='RF AUC: %.3f' % auc_RF)
plt.plot(fpr_LR, tpr_LR, 'b-', label='LR AUC: %.3f' % auc_LR)
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')
plt.legend()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
