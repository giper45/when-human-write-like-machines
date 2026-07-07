class Predictions:
    def __init__(self, 
                 predicted_labels: list,
                 true_labels: list,
                 pred_probs: list, negate=False):
        if negate: 
            print("Negating predictions (for RADAR predictions)")
            predicted_labels = [1 - label for label in predicted_labels]
            pred_probs = [1 - prob for prob in pred_probs]

        self.predicted_labels = predicted_labels
        self.true_labels = true_labels
        self.pred_probs = pred_probs
        self.error_probs = []
        self.error_preds = []
        self.error_threshold = None
        self.is_error_based_classifier = False
        self.ids = []


    def set_ids(self, ids):
        self.ids = ids

    def set_errors(self, error_probs, error_preds, error_threshold):
        self.error_probs = error_probs
        self.error_preds = error_preds
        self.error_threshold = error_threshold
        self.is_error_based_classifier = True
