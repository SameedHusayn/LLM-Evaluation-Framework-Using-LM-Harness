import re


def doc_to_text(x):
    def _mark_span(text, span_str, span_idx, mark):
        pattern_tmpl = r"^((?:\S+\s){N})(W)"
        pattern = re.sub("N", str(span_idx), pattern_tmpl)
        pattern = re.sub("W", span_str, pattern)
        return re.sub(pattern, r"\1{0} \2 {0}".format(mark), text)

    text = x["text"]
    text = _mark_span(text, x["span1_text"], x["span1_index"], "*")
    # Compensate for 2 added "words" added in previous step.
    span2_index = x["span2_index"] + 2 * (x["span1_index"] < x["span2_index"])
    text = _mark_span(text, x["span2_text"], span2_index, "#")

    return text
