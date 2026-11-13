"""Validates a brick build, expands big bricks and works out every tensor shape."""
from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from .catalogue import BRICKS, IMAGE, PORT_TYPES, SERIES, SUMMARY, TEXT, TRAINING_DIAL, WRITERS, Brick, Param

MAX_NODES = 60
MAX_SUMMARY_SIZE = 1_000_000
NAME_RE = re.compile(r"^[\w .\-/%()]{1,40}$", re.UNICODE)
PROJECT_RE = re.compile(r"^[\w \-]{1,60}$", re.UNICODE)


class NodeIn(BaseModel):
    id: str = Field(min_length=1, max_length=64)
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class EdgeIn(BaseModel):
    source: str
    target: str
    targetHandle: str = "in"


class GraphIn(BaseModel):
    projectName: str = "My model"
    training: str = "balanced"
    trainingOverride: dict[str, Any] | None = None
    nodes: list[NodeIn] = Field(default_factory=list)
    edges: list[EdgeIn] = Field(default_factory=list)


@dataclass(frozen=True)
class Shape:
    kind: str
    channels: int = 0
    height: int = 0
    width: int = 0
    steps: int = 0
    features: int = 0
    size: int = 0
    vocab: int = 0  # set by a pretrained writer: how many words its own tokenizer knows

    def code(self) -> str:
        if self.kind == IMAGE:
            return f"(batch, {self.channels}, {self.height}, {self.width})"
        if self.kind == TEXT:
            return f"(batch, {self.steps})"
        if self.kind == SERIES:
            return f"(batch, {self.steps}, {self.features})"
        return f"(batch, {self.size})"

    def plain(self) -> str:
        if self.kind == IMAGE:
            return f"{self.height}×{self.width} image, {self.channels} channel{'s' if self.channels != 1 else ''}"
        if self.kind == TEXT:
            return f"{self.steps} tokens, vocabulary of {self.features:,}"
        if self.kind == SERIES:
            return f"{self.steps} steps, {self.features} value{'s' if self.features != 1 else ''} each"
        return f"summary of {self.size} numbers"


@dataclass
class Issue:
    message: str
    node: str | None = None


@dataclass
class Step:
    """One small brick in execution order, after big bricks are expanded."""

    id: str
    origin: str  # id of the node on the canvas
    brick: Brick
    params: dict[str, Any]
    inputs: list[str]
    input_shapes: list[Shape]
    output: Shape | None = None


@dataclass
class Analysis:
    issues: list[Issue]
    steps: list[Step]
    shapes: dict[str, str]
    socket: Step | None = None
    head: Step | None = None
    target: Step | None = None

    @property
    def ok(self) -> bool:
        return not self.issues


def parse_names(value: Any) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _clean_param(param: Param, value: Any) -> Any:
    """Returns the validated value, or raises ValueError. Nothing unvalidated reaches the generated code."""
    if param.kind == "int":
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
            raise ValueError(f"{param.label} must be a whole number.")
        value = int(value)
        if not param.min <= value <= param.max:
            raise ValueError(f"{param.label} must be between {param.min} and {param.max}.")
        return value
    if param.kind == "choice":
        if value not in param.choices or isinstance(value, bool):
            raise ValueError(f"{param.label} must be one of: {', '.join(map(str, param.choices))}.")
        return value
    if param.kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{param.label} must be on or off.")
        return value
    names = parse_names(value) if isinstance(value, str) else None
    if names is None:
        raise ValueError(f"{param.label} must be a comma-separated list.")
    if not param.min_items <= len(names) <= param.max_items:
        raise ValueError(f"{param.label} needs between {param.min_items} and {param.max_items} names.")
    if len(set(names)) != len(names):
        raise ValueError(f"{param.label} contains a duplicate name.")
    for name in names:
        if not NAME_RE.match(name):
            raise ValueError(f'"{name[:40]}" is not allowed in {param.label}. Use letters, numbers, spaces and . - / % ( ) _ only.')
    return ", ".join(names)


def _clean_params(brick: Brick, raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    clean, problems = {}, []
    for param in brick.params:
        try:
            clean[param.key] = _clean_param(param, raw.get(param.key, param.default))
        except ValueError as err:
            clean[param.key] = param.default
            problems.append(str(err))
    return clean, problems


def _expand(graph: GraphIn, issues: list[Issue]):
    """Replaces big bricks with their small parts and returns (nodes, edges) as plain data."""
    nodes: dict[str, tuple[str, Brick, dict[str, Any]]] = {}
    entry: dict[str, str] = {}
    exit_: dict[str, str] = {}
    edges: list[tuple[str, str, str]] = []
    for node in graph.nodes:
        brick = BRICKS[node.type]
        params, problems = _clean_params(brick, node.params)
        issues.extend(Issue(p, node.id) for p in problems)
        if not brick.parts:
            nodes[node.id] = (node.id, brick, params)
            entry[node.id] = exit_[node.id] = node.id
            continue
        previous = None
        for index, part in enumerate(brick.parts):
            part_brick = BRICKS[part.brick]
            part_id = f"{node.id}::{index}"
            merged = {p.key: p.default for p in part_brick.params} | part.params
            merged |= {k: v for k, v in params.items() if k in merged}
            nodes[part_id] = (node.id, part_brick, merged)
            if previous:
                edges.append((previous, part_id, "in"))
            previous = part_id
        entry[node.id] = f"{node.id}::0"
        exit_[node.id] = previous
    for edge in graph.edges:
        edges.append((exit_[edge.source], entry[edge.target], edge.targetHandle))
    return nodes, edges


def _check_heads(step: Step, features: int) -> None:
    heads = step.params["heads"]
    if features % heads:
        raise ValueError(f"{step.brick.name} receives {features} numbers per step, which doesn't divide evenly by {heads} attention heads. Change the heads or the Embedding size.")


def _infer(step: Step, socket: Shape | None, target: Shape | None = None) -> Shape | None:
    """Output shape for one brick. Raises ValueError with a plain-English message."""
    t, p, inp = step.brick.id, step.params, step.input_shapes
    x = inp[0] if inp else None
    if t == "image_folder":
        return Shape(IMAGE, channels=p["channels"], height=p["image_size"], width=p["image_size"])
    if t == "time_series":
        return Shape(SERIES, steps=p["sequence_length"], features=len(parse_names(p["columns"])))
    if t in ("text_input", "target_text"):
        return Shape(TEXT, steps=p["max_length"], features=p["vocab_size"])
    if t == "embedding":
        return Shape(SERIES, steps=x.steps, features=p["size"])
    if t == "positional":
        return x
    if t in ("attention", "transformer_encoder", "transformer_decoder"):
        _check_heads(step, x.features)
        return x
    if t == "pretrained_language":
        return Shape(SERIES, steps=x.steps, features=768)
    if t == "pretrained_writer":
        return Shape(SERIES, steps=x.steps, features=WRITERS[p["model"]]["hidden"], vocab=WRITERS[p["model"]]["vocab"])
    if t == "pretrained_vision":
        if x.channels != 3 or (x.height, x.width) != (224, 224):
            raise ValueError(f"The pretrained vision transformer needs 224×224 colour images, but it receives {x.plain()}. Update the Image folder settings.")
        return Shape(SUMMARY, size=768)
    if t == "mean_pool":
        return Shape(SUMMARY, size=x.features)
    if t == "generate":
        expected = target or socket
        if expected is None or expected.kind != TEXT:
            raise ValueError("Generating text needs a Text input.")
        if x.steps != expected.steps:
            raise ValueError(f"This head must produce {expected.steps} tokens but receives {x.steps}. Keep the sequence length unchanged on the way here.")
        return None
    if t == "conv":
        return Shape(IMAGE, channels=p["filters"], height=x.height, width=x.width)
    if t == "pool":
        if min(x.height, x.width) < 2:
            raise ValueError("The image is already 1 pixel wide, so it can't be pooled further. Remove a Pooling layer or use larger images.")
        return Shape(IMAGE, channels=x.channels, height=x.height // 2, width=x.width // 2)
    if t == "residual":
        return x
    if t == "upsample":
        if max(x.height, x.width) * 2 > 2048:
            raise ValueError("The image would grow past 2048 pixels. Remove an Upsampling layer.")
        return Shape(IMAGE, channels=p["filters"], height=x.height * 2, width=x.width * 2)
    if t == "backbone":
        if x.channels != 3:
            raise ValueError(f"The pretrained backbone needs 3 colour channels, but it receives {x.channels}. Set the Image folder input to 3 channels and place the backbone first.")
        if min(x.height, x.width) < 32:
            raise ValueError("The pretrained backbone needs images of at least 32×32 pixels.")
        return Shape(IMAGE, channels=512, height=max(x.height // 32, 1), width=max(x.width // 32, 1))
    if t == "cnn_decoder":
        if socket is None or socket.kind != IMAGE:
            raise ValueError("A CNN decoder rebuilds an image, so this build needs an Image folder input.")
        return Shape(IMAGE, channels=p["filters"], height=socket.height, width=socket.width)
    if t in ("lstm", "gru", "rnn"):
        return Shape(SERIES, steps=x.steps, features=p["hidden"] * (2 if p["bidirectional"] else 1))
    if t == "lstm_decoder":
        steps = p["steps"]
        if steps == 0:
            if socket is None or socket.kind != SERIES:
                raise ValueError("Only a Time series input has a sequence length to match. Set Steps to generate above 0.")
            steps = socket.steps
        return Shape(SERIES, steps=steps, features=p["hidden"])
    if t == "global_pool":
        return Shape(SUMMARY, size=x.channels)
    if t == "flatten":
        size = x.channels * x.height * x.width
        if size > MAX_SUMMARY_SIZE:
            raise ValueError(f"Flatten would create {size:,} numbers, which is too many. Use Global pooling or add Pooling layers first.")
        return Shape(SUMMARY, size=size)
    if t == "last_step":
        return Shape(SUMMARY, size=x.features)
    if t == "dense":
        return Shape(SUMMARY, size=p["size"])
    if t == "dropout":
        return x
    if t == "combine":
        return Shape(SUMMARY, size=inp[0].size + inp[1].size)
    if t == "classify":
        return None
    if t == "predict_number":
        return None
    if t == "forecast":
        if socket is None or socket.kind != SERIES:
            raise ValueError("Forecasting predicts future values of a time series, so this build needs a Time series input.")
        if x.kind == SERIES and x.steps != p["horizon"]:
            raise ValueError(f"This head forecasts {p['horizon']} steps but receives {x.steps}. Make the two numbers match.")
        return None
    if t == "detect_anomalies":
        if socket is None or x.kind != socket.kind:
            raise ValueError("This head rebuilds your input, so the component before it must produce the same kind of data as the input. Add a matching decoder.")
        if x.kind == IMAGE and (x.height, x.width) != (socket.height, socket.width):
            raise ValueError(f"This head must rebuild a {socket.height}×{socket.width} image but receives {x.height}×{x.width}. Use a CNN decoder or match the Upsampling and Pooling bricks.")
        if x.kind == SERIES and x.steps != socket.steps:
            raise ValueError(f"This head must rebuild {socket.steps} steps but receives {x.steps}. Change Steps to generate on the decoder.")
        return None
    raise ValueError(f"Unknown brick {t}.")  # pragma: no cover


OVERRIDE_LIMITS = {
    "epochs": ("Epochs", 1, 10000, True),
    "batch_size": ("Batch size", 1, 8192, True),
    "learning_rate": ("Learning rate", 1e-7, 1.0, False),
}


def _check_override(override: dict[str, Any] | None) -> list[str]:
    problems = []
    for key, value in (override or {}).items():
        if key not in OVERRIDE_LIMITS:
            problems.append("Unknown training value.")
            continue
        label, low, high, whole = OVERRIDE_LIMITS[key]
        valid = isinstance(value, (int, float)) and not isinstance(value, bool) and low <= value <= high and (not whole or int(value) == value)
        if not valid:
            kind = "a whole number" if whole else "a number"
            problems.append(f"{label} must be {kind} between {low} and {high}.")
    return problems


def analyse(graph: GraphIn) -> Analysis:
    issues: list[Issue] = []
    if len(graph.nodes) > MAX_NODES:
        return Analysis([Issue(f"A build can have at most {MAX_NODES} components.")], [], {})
    if not PROJECT_RE.match(graph.projectName.strip() or "x"):
        issues.append(Issue("The project name can only use letters, numbers, spaces, - and _."))
    if graph.training not in TRAINING_DIAL:
        issues.append(Issue("Unknown training setting."))
    issues.extend(Issue(message) for message in _check_override(graph.trainingOverride))

    ids = [n.id for n in graph.nodes]
    if len(ids) != len(set(ids)):
        return Analysis([Issue("Two components share the same id.")], [], {})
    known = set(ids)
    for node in graph.nodes:
        if node.type not in BRICKS:
            return Analysis([Issue(f"Unknown component type: {node.type[:40]}", node.id)], [], {})
    by_id = {n.id: n for n in graph.nodes}
    for edge in graph.edges:
        if edge.source not in known or edge.target not in known:
            return Analysis([Issue("A connection points at a component that no longer exists.")], [], {})
        ports = {p.id for p in BRICKS[by_id[edge.target].type].inputs}
        if edge.targetHandle not in ports:
            return Analysis([Issue("A connection uses an input that doesn't exist.", edge.target)], [], {})
        if BRICKS[by_id[edge.source].type].output is None:
            return Analysis([Issue("A task head has no output to connect.", edge.source)], [], {})

    nodes, edges = _expand(graph, issues)

    incoming: dict[str, dict[str, str]] = defaultdict(dict)
    outgoing: dict[str, list[str]] = defaultdict(list)
    for src, dst, handle in edges:
        if handle in incoming[dst]:
            issues.append(Issue("Two components are connected to the same input.", nodes[dst][0]))
        incoming[dst][handle] = src
        outgoing[src].append(dst)

    # Topological order (Kahn). Anything left over is part of a loop.
    degree = {nid: len(incoming[nid]) for nid in nodes}
    queue = deque(nid for nid in nodes if degree[nid] == 0)
    order: list[str] = []
    while queue:
        nid = queue.popleft()
        order.append(nid)
        for dst in outgoing[nid]:
            degree[dst] -= 1
            if degree[dst] == 0:
                queue.append(dst)
    if len(order) != len(nodes):
        return Analysis(issues + [Issue("These components form a cycle. Data must flow one way, from the input to the task head.")], [], {})

    sockets = [nid for nid in order if nodes[nid][1].group == "socket"]
    heads = [nid for nid in order if nodes[nid][1].output is None and nodes[nid][1].group == "head"]
    target_sockets = [nid for nid in sockets if nodes[nid][1].id == "target_text"]
    sockets = [nid for nid in sockets if nodes[nid][1].id != "target_text"]
    if not sockets:
        issues.append(Issue("Add a data input so the model knows what data it receives."))
    elif len(sockets) > 1:
        for nid in sockets[1:]:
            issues.append(Issue("This version supports one data input per build.", nodes[nid][0]))
    for nid in target_sockets[1:]:
        issues.append(Issue("This version supports one Target text input per build.", nodes[nid][0]))
    if not heads:
        issues.append(Issue("Add a task head to say what the model should do."))
    elif len(heads) > 1:
        for nid in heads[1:]:
            issues.append(Issue("This version supports one task head per build.", nodes[nid][0]))

    socket_shape: Shape | None = None
    if sockets:
        first_socket = nodes[sockets[0]]
        socket_shape = _infer(Step(sockets[0], first_socket[0], first_socket[1], first_socket[2], [], []), None)

    target_shape: Shape | None = None
    if target_sockets:
        first_target = nodes[target_sockets[0]]
        target_shape = _infer(Step(target_sockets[0], first_target[0], first_target[1], first_target[2], [], []), None)

    produced: dict[str, Shape] = {}
    lineage: dict[str, set[str]] = {}
    steps: list[Step] = []
    shapes: dict[str, str] = {}
    for nid in order:
        origin, brick, params = nodes[nid]
        missing = [port for port in brick.inputs if port.id not in incoming[nid] and not port.optional]
        if missing:
            issues.append(Issue(f"{BRICKS[by_id[origin].type].name} has no input connected.", origin))
            continue
        inputs = [incoming[nid][port.id] for port in brick.inputs if port.id in incoming[nid]]
        lineage[nid] = set().union(*(lineage.get(i, set()) for i in inputs)) if inputs else {nid}
        if any(i not in produced for i in inputs):
            continue  # an earlier brick failed; its issue is already reported
        input_shapes = [produced[i] for i in inputs]
        step = Step(nid, origin, brick, params, inputs, input_shapes)
        bad_type = next((s for s, port in zip(input_shapes, brick.inputs) if s.kind not in port.accepts), None)
        if bad_type:
            wanted = " or ".join(PORT_TYPES[a]["name"] for a in brick.inputs[0].accepts)
            issues.append(Issue(f"{brick.name} takes {wanted}, but receives {PORT_TYPES[bad_type.kind]['name']}.", origin))
            continue
        try:
            step.output = _infer(step, socket_shape, target_shape)
        except ValueError as err:
            issues.append(Issue(str(err), origin))
            continue
        if step.output is not None:
            produced[nid] = step.output
            shapes[origin] = step.output.plain()
        steps.append(step)

    for nid in order:
        origin, brick, _ = nodes[nid]
        if brick.group != "head" and nid in produced and not outgoing[nid]:
            issues.append(Issue(f"The output of {BRICKS[by_id[origin].type].name} is not connected.", origin))

    decoders = [s for s in steps if s.brick.id == "transformer_decoder"]
    with_memory = [s for s in decoders if len(s.inputs) == 2]
    target_id = target_sockets[0] if target_sockets else None
    if target_id and not with_memory:
        issues.append(Issue("Target text is only used by a Transformer decoder with an encoder connected.", nodes[target_id][0]))
    for decoder in with_memory:
        if target_id is None:
            issues.append(Issue("A decoder with an encoder connected needs a Target text input to learn from.", decoder.origin))
        elif lineage[decoder.inputs[0]] != {target_id} or target_id in lineage[decoder.inputs[1]]:
            issues.append(Issue("Connect Target text to the decoder's Tokens input, and your input text to its Encoder input.", decoder.origin))
    head_step = next((s for s in steps if s.brick.group == "head"), None)
    if head_step and target_id and head_step.brick.id != "generate":
        issues.append(Issue("Target text needs a Generate text head.", head_step.origin))
    if head_step and head_step.brick.id == "generate" and target_id and target_id not in lineage.get(head_step.id, set()):
        issues.append(Issue("Generate text must follow the decoder that reads Target text.", head_step.origin))
    writers = [s for s in steps if s.brick.id == "pretrained_writer"]
    for writer in writers:
        if any(s.brick.id == "embedding" for s in steps):
            issues.append(Issue("The pretrained writer has its own tokenizer, so it can't share a build with an Embedding layer.", writer.origin))
        elif target_id:
            issues.append(Issue("The pretrained writer continues text, so it doesn't use Target text. Remove the Target text input.", writer.origin))
        elif not (head_step and head_step.brick.id == "generate" and head_step.inputs == [writer.id]):
            issues.append(Issue("The pretrained writer writes text, so connect its output straight to Generate text. "
                                "To sort text into categories, use the Pretrained language model.", writer.origin))
    if writers and socket_shape is not None and sockets:
        shapes[nodes[sockets[0]][0]] = f"{socket_shape.steps} tokens, using {writers[0].params['model']}'s own tokenizer"
    if head_step and head_step.brick.id == "generate" and not decoders and not writers and not any(s.brick.id in ("lstm", "gru", "rnn") for s in steps):
        issues.append(Issue("Generate text needs a Transformer decoder or a sequence layer before it, otherwise it can't learn from earlier tokens.", head_step.origin))
    pretrained = [s for s in steps if s.brick.id == "pretrained_language"]
    if pretrained and any(s.brick.id == "embedding" for s in steps):
        issues.append(Issue("The pretrained language model has its own tokenizer, so it can't share a build with an Embedding layer.", pretrained[0].origin))
    if pretrained and head_step and head_step.brick.id == "generate":
        issues.append(Issue("This version uses the pretrained language model for classifying and predicting numbers, not generating text.", pretrained[0].origin))

    socket = next((s for s in steps if s.brick.group == "socket" and s.brick.id != "target_text"), None)
    target = next((s for s in steps if s.brick.id == "target_text"), None)
    head = next((s for s in steps if s.brick.group == "head"), None)
    return Analysis(issues, steps, shapes, socket, head, target)
