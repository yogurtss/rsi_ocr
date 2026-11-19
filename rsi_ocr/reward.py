import math

from .formats import NUMBER, normalize, parse_html, parse_otsl, similarity, table_cells, table_metrics


def reward_components(task, completion, solution, table_format):
    if task == "text":
        left, right = normalize(completion), normalize(solution)
        applicable = bool(NUMBER.search(left) or NUMBER.search(right))
        numeric = (left or not right) and NUMBER.findall(left) == NUMBER.findall(right)
        return {"character": similarity(left, right), "numeric": float(bool(numeric))}, applicable
    if task == "table":
        reference = table_cells(solution)
        try:
            parse = parse_otsl if table_format == "otsl" else parse_html
            predicted = parse(completion)
        except ValueError:
            return {"structure": 0.0, "content": 0.0, "numeric": 0.0}, False
        applicable = any(NUMBER.search(cell.text) for cell in predicted + reference)
        return dict(zip(("structure", "content", "numeric"), table_metrics(predicted, reference), strict=True)), applicable
    raise ValueError(f"unknown reward task: {task}")


def score(task, completion, solution, config):
    components, numeric_present = reward_components(task, completion, solution, config["table_format"])
    weights = config[task]
    numerator = sum(components[key] * weights[key] for key in components)
    denominator = sum(weights[key] for key in components)
    if not numeric_present:
        numerator -= components["numeric"] * weights["numeric"]
        denominator -= weights["numeric"]
    return {"score": numerator / denominator, **components}


def validate_weights(config):
    if config["table_format"] not in {"otsl", "html"}:
        raise ValueError("table_format must be otsl or html")
    keys_by_task = {"text": {"character", "numeric"}, "table": {"structure", "content", "numeric"}}
    for task, keys in keys_by_task.items():
        weights = config[task]
        if weights.keys() != keys or any(not math.isfinite(value) or value < 0 for value in weights.values()):
            raise ValueError(f"invalid {task} reward weights: expected {sorted(keys)} with finite nonnegative values")
        if sum(weights[key] for key in keys - {"numeric"}) <= 0:
            raise ValueError(f"{task} reward must include a character, structure or content weight")
