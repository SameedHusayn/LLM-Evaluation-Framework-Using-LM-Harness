import sklearn.metrics


def mean_3class_f1(predictions, references):  # This is a passthrough function

    string_label = ["entailment", "contradiction", "neutral"]
    predictions = string_label.index(predictions[0])
    references = string_label.index(references[0])

    return (predictions, references)


def agg_mean_3class_f1(items):

    predictions, references = zip(*items)

    """Computes the unweighted average of the F1 per class."""
    metric_str = "fbeta_score"
    metric_post_process_fn = lambda x: x
    metric_fn_kwargs = {
        "beta": 1,
        "labels": range(3),
        "average": "macro",
    }

    def _fn(predictions, references):
        metric_fn = getattr(sklearn.metrics, metric_str)
        metric_val = metric_fn(references, predictions, **metric_fn_kwargs)
        if metric_post_process_fn is not None:
            metric_val = metric_post_process_fn(metric_val)
        return metric_val

    return _fn(predictions, references)
