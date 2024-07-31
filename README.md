# DLBasics

## F1 Score:
    The F1 score is the harmonic mean of precision and recall, which means it is a good measure of the balance between precision and recall.
    The formula for the F1 score is: 
    F1 = 2 * ((Precision * Recall)/ (Precision + Recall))

### Micro F1 Score: 
    The micro-average aggregates the contributions of all classes to compute the average metric. It considers the total number of true positives, false negatives, and false positives.

### Macro F1 Score: 
    The macro-average calculates the F1 score for each class independently and then takes the average (unweighted).

### Weighted F1 Score: 
    The weighted average calculates the F1 score for each class independently and then takes the average, weighted by the number of true instances for each class.