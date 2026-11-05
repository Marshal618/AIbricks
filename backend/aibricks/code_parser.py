"""Turns model.py source code back into blocks.

The code is only parsed with `ast`, never imported or run, so anything typed here is safe.
Only code built from AI Bricks components can become blocks. Everything else is reported
with its line number.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

from .catalogue import BRICKS
from .graph import parse_names


@dataclass(frozen=True)
class Component:
    brick: str
    args: tuple[str, ...]  # constructor arguments, in order
    params: dict[str, str] = field(default_factory=dict)  # argument name -> brick setting
    signature: str = ""


_RECURRENT = ("in_features", "hidden", "layers", "bidirectional")
_RECURRENT_MAP = {"hidden": "hidden", "layers": "layers", "bidirectional": "bidirectional"}
_FREEZE = ("pretrained", "freeze")

COMPONENTS: dict[str, Component] = {
    "ConvBlock": Component("conv", ("in_channels", "out_channels", "kernel_size"), {"out_channels": "filters", "kernel_size": "kernel_size"}, "ConvBlock(in_channels, out_channels, kernel_size=3)"),
    "Pool": Component("pool", ("mode",), {"mode": "mode"}, 'Pool(mode="max")'),
    "ResidualBlock": Component("residual", ("channels",), {}, "ResidualBlock(channels)"),
    "Upsample": Component("upsample", ("in_channels", "out_channels"), {"out_channels": "filters"}, "Upsample(in_channels, out_channels)"),
    "ResNet18Backbone": Component("backbone", _FREEZE, {"freeze": "freeze"}, "ResNet18Backbone(pretrained=use_pretrained_weights, freeze=True)"),
    "CNNDecoder": Component("cnn_decoder", ("in_features", "channels", "height", "width"), {"channels": "filters"}, "CNNDecoder(in_features, channels, height=64, width=64)"),
    "LSTMLayer": Component("lstm", _RECURRENT, _RECURRENT_MAP, "LSTMLayer(in_features, hidden, layers=1, bidirectional=False)"),
    "GRULayer": Component("gru", _RECURRENT, _RECURRENT_MAP, "GRULayer(in_features, hidden, layers=1, bidirectional=False)"),
    "RNNLayer": Component("rnn", _RECURRENT, _RECURRENT_MAP, "RNNLayer(in_features, hidden, layers=1, bidirectional=False)"),
    "LSTMDecoder": Component("lstm_decoder", ("in_features", "hidden", "steps"), {"hidden": "hidden", "steps": "steps"}, "LSTMDecoder(in_features, hidden, steps=24)"),
    "TokenEmbedding": Component("embedding", ("vocab_size", "size"), {"size": "size"}, "TokenEmbedding(vocab_size=8000, size=128)"),
    "PositionalEncoding": Component("positional", ("size", "max_length"), {}, "PositionalEncoding(size, max_length=64)"),
    "AttentionBlock": Component("attention", ("size", "heads"), {"heads": "heads"}, "AttentionBlock(size, heads=4)"),
    "TransformerEncoder": Component("transformer_encoder", ("size", "heads", "layers"), {"heads": "heads", "layers": "layers"}, "TransformerEncoder(size, heads=4, layers=2)"),
    "TransformerDecoderOnly": Component("transformer_decoder", ("size", "heads", "layers"), {"heads": "heads", "layers": "layers"}, "TransformerDecoderOnly(size, heads=4, layers=2)"),
    "TransformerDecoder": Component("transformer_decoder", ("size", "heads", "layers", "memory_size"), {"heads": "heads", "layers": "layers"}, "TransformerDecoder(size, heads=4, layers=2, memory_size=128)"),
    "PretrainedLanguageModel": Component("pretrained_language", _FREEZE, {"freeze": "freeze"}, "PretrainedLanguageModel(pretrained=use_pretrained_weights, freeze=True)"),
    "PretrainedVisionTransformer": Component("pretrained_vision", _FREEZE, {"freeze": "freeze"}, "PretrainedVisionTransformer(pretrained=use_pretrained_weights, freeze=True)"),
    "GlobalPool": Component("global_pool", (), {}, "GlobalPool()"),
    "Flatten": Component("flatten", (), {}, "Flatten()"),
    "LastStep": Component("last_step", (), {}, "LastStep()"),
    "MeanPool": Component("mean_pool", (), {}, "MeanPool()"),
    "Combine": Component("combine", (), {}, "Combine()"),
    "DenseLayer": Component("dense", ("in_features", "out_features"), {"out_features": "size"}, "DenseLayer(in_features, out_features)"),
    "Dropout": Component("dropout", ("probability",), {}, "Dropout(0.2)"),
    "ClassifyHead": Component("classify", ("in_features", "num_classes"), {}, "ClassifyHead(in_features, num_classes=len(CLASS_NAMES))"),
    "PredictNumberHead": Component("predict_number", ("in_features",), {}, "PredictNumberHead(in_features)"),
    "ForecastHead": Component("forecast", ("in_features", "horizon", "num_features"), {"horizon": "horizon"}, "ForecastHead(in_features, horizon=24, num_features=1)"),
    "SequenceForecastHead": Component("forecast", ("in_features", "num_features"), {}, "SequenceForecastHead(in_features, num_features=1)"),
    "RebuildImageHead": Component("detect_anomalies", ("in_channels", "out_channels"), {}, "RebuildImageHead(in_channels, out_channels)"),
    "RebuildSeriesHead": Component("detect_anomalies", ("in_features", "out_features"), {}, "RebuildSeriesHead(in_features, out_features)"),
    "GenerateHead": Component("generate", ("in_features", "vocab_size"), {}, "GenerateHead(in_features, vocab_size=8000)"),
}

ALLOWED_IMPORTS = {"torch", "bricks", "torch.nn"}
START_COMMENT = re.compile(r"#\s*(x|target)\s+starts\s+as\s+\(\s*batch\s*((?:,\s*\d+\s*)*)\)")
PRETRAINED_VOCAB = 30522
STARTER = '''"""Write your model here, then convert it to blocks."""
from torch import nn

from bricks import ClassifyHead, ConvBlock, GlobalPool, Pool

CLASS_NAMES = ['OK', 'Defect']


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBlock(3, 32, kernel_size=3)
        self.pool = Pool(mode="max")
        self.global_pool = GlobalPool()
        self.classify_head = ClassifyHead(32, num_classes=len(CLASS_NAMES))

    def forward(self, x):
        # x starts as (batch, 3, 64, 64)
        x = self.conv(x)
        x = self.pool(x)
        x = self.global_pool(x)
        x = self.classify_head(x)
        return x
'''


@dataclass
class ParseResult:
    errors: list[dict[str, Any]]
    nodes: list[dict[str, Any]]
    edges: list[dict[str, str]]
    constructors: list[tuple[str, str, str, int]]  # class, typed source, attribute, line


def _fail(result: ParseResult, node: ast.AST | None, message: str) -> None:
    result.errors.append({"line": getattr(node, "lineno", None), "message": message})


def _literal(value: ast.expr) -> Any:
    return ast.literal_eval(value)  # only ever reads plain constants


def _bind(result: ParseResult, call: ast.Call, name: str) -> dict[str, Any] | None:
    """Maps a constructor call onto the brick's settings."""
    component = COMPONENTS[name]
    named: dict[str, ast.expr] = {}
    if len(call.args) > len(component.args):
        _fail(result, call, f"{name} takes at most {len(component.args)} arguments: {component.signature}")
        return None
    for arg_name, value in zip(component.args, call.args):
        if isinstance(value, ast.Starred):
            _fail(result, call, f"{name} can't use * arguments here.")
            return None
        named[arg_name] = value
    for keyword in call.keywords:
        if keyword.arg not in component.args:
            label = f"**" if keyword.arg is None else keyword.arg
            _fail(result, call, f"{name} has no argument {label}. Use: {component.signature}")
            return None
        named[keyword.arg] = keyword.value

    params: dict[str, Any] = {}
    for arg_name, param in component.params.items():
        if arg_name not in named:
            continue
        try:
            params[param] = _literal(named[arg_name])
        except (ValueError, SyntaxError):
            _fail(result, call, f"{name}: {arg_name} must be a plain value such as a number, not a calculation.")
            return None
    if name == "Dropout" and "probability" in named:
        try:
            params["percent"] = round(float(_literal(named["probability"])) * 100)
        except (ValueError, SyntaxError, TypeError):
            _fail(result, call, "Dropout: the probability must be a plain number such as 0.2.")
            return None
    # Size arguments are worked out from the model, except these two which define the input.
    for arg_name in ("vocab_size",):
        if name == "TokenEmbedding" and arg_name in named:
            try:
                params["__vocab"] = _literal(named[arg_name])
            except (ValueError, SyntaxError):
                _fail(result, call, "TokenEmbedding: vocab_size must be a plain number.")
                return None
    return params


def _parse_init(result: ParseResult, method: ast.FunctionDef) -> dict[str, tuple[str, dict[str, Any], ast.stmt]]:
    modules: dict[str, tuple[str, dict[str, Any], ast.stmt]] = {}
    for statement in method.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        if isinstance(statement, ast.Expr) and ast.unparse(statement.value) == "super().__init__()":
            continue
        if not (
            isinstance(statement, ast.Assign) and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Attribute)
            and isinstance(statement.targets[0].value, ast.Name) and statement.targets[0].value.id == "self"
            and isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name)
        ):
            _fail(result, statement, "Blocks can only be made from lines like: self.name = Component(...)")
            continue
        attribute, class_name = statement.targets[0].attr, statement.value.func.id
        if class_name not in COMPONENTS:
            _fail(result, statement, f"{class_name} isn't an AI Bricks component, so it can't become a block. Choose one from the reference list.")
            continue
        if attribute in modules:
            _fail(result, statement, f"self.{attribute} is defined twice.")
            continue
        params = _bind(result, statement.value, class_name)
        if params is None:
            continue
        modules[attribute] = (class_name, params, statement)
        result.constructors.append((class_name, ast.unparse(statement.value), attribute, statement.lineno))
    return modules


def _parse_forward(result: ParseResult, method: ast.FunctionDef, modules, source_comments) -> None:
    names = [arg.arg for arg in method.args.args]
    if names not in (["self", "x"], ["self", "x", "target"]):
        _fail(result, method, "forward must be: def forward(self, x) or def forward(self, x, target)")
        return
    variables = {"x": "input"}
    if "target" in names:
        variables["target"] = "target"
    used: set[str] = set()
    returned = False
    for statement in method.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        if (
            isinstance(statement, ast.Assign) and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id.endswith("padding_mask")
            and isinstance(statement.value, ast.Compare)
        ):
            continue  # padding masks are regenerated automatically
        if isinstance(statement, ast.Return):
            if not isinstance(statement.value, ast.Name) or statement.value.id not in variables:
                _fail(result, statement, "forward must end by returning the output of the last component.")
            returned = True
            continue
        call = statement.value if isinstance(statement, ast.Assign) else None
        if not (
            isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name)
            and isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name) and call.func.value.id == "self"
        ):
            _fail(result, statement, "Blocks can only be made from lines like: x = self.name(x). Move custom logic into the exported project.")
            continue
        attribute = call.func.attr
        if attribute not in modules:
            _fail(result, statement, f"self.{attribute} isn't defined in __init__.")
            continue
        if attribute in used:
            _fail(result, statement, f"self.{attribute} is used twice. Each block appears once, so define a second component.")
            continue
        class_name, params, _ = modules[attribute]
        brick = BRICKS[COMPONENTS[class_name].brick]
        inputs = []
        bad = False
        for arg in call.args:
            if not isinstance(arg, ast.Name) or arg.id not in variables:
                _fail(result, statement, f"self.{attribute} must receive the output of an earlier line.")
                bad = True
                break
            inputs.append(variables[arg.id])
        if bad:
            continue
        for keyword in call.keywords:
            if keyword.arg not in ("padding_mask", "memory_padding_mask"):
                _fail(result, statement, f"self.{attribute}: only padding masks can be passed by name.")
                bad = True
        required = [port for port in brick.inputs if not port.optional]
        if not bad and not len(required) <= len(inputs) <= len(brick.inputs):
            _fail(result, statement, f"{brick.name} takes {len(required)} input(s), but this line gives it {len(inputs)}.")
            bad = True
        if bad:
            continue
        used.add(attribute)
        for port, source in zip(brick.inputs, inputs):
            result.edges.append({"source": source, "target": attribute, "targetHandle": port.id})
        result.nodes.append({"id": attribute, "type": brick.id, "params": {k: v for k, v in params.items()}, "__class": class_name})
        variables[statement.targets[0].id] = attribute
    if not returned:
        _fail(result, method, "forward must end with a return line.")
    for attribute, (_, _, statement) in modules.items():
        if attribute not in used and not any(e["line"] == statement.lineno for e in result.errors):
            _fail(result, statement, f"self.{attribute} is defined but never used in forward.")


def _sockets(result: ParseResult, source: str, current: list[dict[str, Any]]) -> None:
    starts = {}
    for number, line in enumerate(source.splitlines(), start=1):
        match = START_COMMENT.search(line)
        if match:
            dims = [int(d) for d in re.findall(r"\d+", match.group(2))]
            starts[match.group(1)] = (dims, number)

    def existing(*types: str) -> dict[str, Any] | None:
        return next((n for n in current if n["type"] in types), None)

    def vocab_for(socket_id: str) -> int | None:
        for edge in result.edges:
            if edge["source"] != socket_id:
                continue
            node = next((n for n in result.nodes if n["id"] == edge["target"]), None)
            if node and "__vocab" in node["params"]:
                return node["params"]["__vocab"]
            if node and node["type"] == "pretrained_language":
                return PRETRAINED_VOCAB
        return None

    sockets = []
    if "x" not in starts:
        previous = existing("image_folder", "time_series", "text_input")
        if previous is None:
            _fail(result, None, "Add a comment in forward describing the input, such as: # x starts as (batch, 3, 64, 64)")
            return
        sockets.append({"id": "input", "type": previous["type"], "params": dict(previous.get("params", {}))})
    else:
        dims, line = starts["x"]
        previous = None
        if len(dims) == 3:
            if dims[1] != dims[2]:
                result.errors.append({"line": line, "message": "Images must be square, so height and width must match."})
                return
            previous = existing("image_folder")
            params = dict(previous.get("params", {})) if previous else {}
            params |= {"channels": dims[0], "image_size": dims[1]}
            sockets.append({"id": "input", "type": "image_folder", "params": params})
        elif len(dims) == 2:
            previous = existing("time_series")
            params = dict(previous.get("params", {})) if previous else {}
            names = parse_names(params.get("columns", ""))
            count = dims[1]
            names = names[:count] + [f"value_{i + 1}" for i in range(len(names), count)]
            params |= {"sequence_length": dims[0], "columns": ", ".join(names)}
            sockets.append({"id": "input", "type": "time_series", "params": params})
        elif len(dims) == 1:
            previous = existing("text_input")
            params = dict(previous.get("params", {})) if previous else {}
            params["max_length"] = dims[0]
            vocab = vocab_for("input")
            if vocab is not None:
                params["vocab_size"] = vocab
            sockets.append({"id": "input", "type": "text_input", "params": params})
        else:
            result.errors.append({"line": line, "message": "The input should look like (batch, channels, height, width), (batch, steps, values) or (batch, tokens)."})
            return
    if any(edge["source"] == "target" for edge in result.edges):
        previous = existing("target_text")
        params = dict(previous.get("params", {})) if previous else {}
        if "target" in starts:
            params["max_length"] = starts["target"][0][0] if starts["target"][0] else params.get("max_length", 64)
        vocab = vocab_for("target")
        if vocab is not None:
            params["vocab_size"] = vocab
        sockets.append({"id": "target", "type": "target_text", "params": params})
    result.nodes = sockets + result.nodes


def parse_model(source: str, current_nodes: list[dict[str, Any]] | None = None) -> ParseResult:
    result = ParseResult([], [], [], [])
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        _fail(result, None, f"Python syntax error: {error.msg}.")
        result.errors[-1]["line"] = error.lineno
        return result
    except (ValueError, RecursionError, MemoryError):
        _fail(result, None, "This code can't be read.")
        return result

    class_names: list[str] = []
    model_class: ast.ClassDef | None = None
    for statement in tree.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        if isinstance(statement, ast.Import):
            bad = [alias.name for alias in statement.names if alias.name not in ALLOWED_IMPORTS]
            if bad:
                _fail(result, statement, f"Importing {bad[0]} can't become a block. Only torch and bricks are allowed here.")
            continue
        if isinstance(statement, ast.ImportFrom):
            if statement.module not in ALLOWED_IMPORTS:
                _fail(result, statement, f"Importing from {statement.module} can't become a block. Only torch and bricks are allowed here.")
            continue
        if isinstance(statement, ast.Assign) and ast.unparse(statement.targets[0]) == "CLASS_NAMES":
            try:
                value = _literal(statement.value)
                if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                    raise ValueError
                class_names = value
            except (ValueError, SyntaxError):
                _fail(result, statement, "CLASS_NAMES must be a plain list of names.")
            continue
        if isinstance(statement, ast.FunctionDef) and statement.name == "build_model":
            continue  # regenerated
        if isinstance(statement, ast.ClassDef):
            if model_class is not None:
                _fail(result, statement, "Write one model class. Custom components can be added after export.")
                continue
            model_class = statement
            continue
        _fail(result, statement, "Only imports, CLASS_NAMES and one model class can become blocks.")

    if model_class is None:
        if not result.errors:
            _fail(result, None, "Write a model class with __init__ and forward.")
        return result
    methods = {}
    for statement in model_class.body:
        if isinstance(statement, ast.FunctionDef) and statement.name in ("__init__", "forward"):
            methods[statement.name] = statement
        elif not (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant)):
            _fail(result, statement, "The model class can only contain __init__ and forward.")
    if set(methods) != {"__init__", "forward"}:
        _fail(result, model_class, "The model class needs both __init__ and forward.")
        return result

    modules = _parse_init(result, methods["__init__"])
    _parse_forward(result, methods["forward"], modules, source)
    if result.errors:
        return result

    # Settings that live outside the constructor.
    for node in result.nodes:
        if node["type"] == "classify":
            if not class_names:
                _fail(result, None, "Add CLASS_NAMES = ['First', 'Second'] above the model class.")
                return result
            node["params"]["classes"] = ", ".join(class_names)
        if node["__class"] == "SequenceForecastHead":
            source_id = next(e["source"] for e in result.edges if e["target"] == node["id"])
            decoder = next((n for n in result.nodes if n["id"] == source_id), None)
            steps = (decoder or {}).get("params", {}).get("steps")
            if isinstance(steps, int) and steps > 0:
                node["params"]["horizon"] = steps
    _sockets(result, source, current_nodes or [])
    for node in result.nodes:
        node.pop("__class", None)
        node.get("params", {}).pop("__vocab", None)
    return result


def correction_notes(result: ParseResult, regenerated: str) -> list[dict[str, Any]]:
    """Explains size arguments that were replaced with what the connected model actually needs."""
    tree = ast.parse(regenerated)
    model = next(s for s in tree.body if isinstance(s, ast.ClassDef))
    init = next(s for s in model.body if isinstance(s, ast.FunctionDef) and s.name == "__init__")
    produced: dict[str, list[str]] = {}
    for statement in init.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call) and isinstance(statement.value.func, ast.Name):
            produced.setdefault(statement.value.func.id, []).append(ast.unparse(statement.value))
    notes = []
    seen: dict[str, int] = {}
    for class_name, typed, attribute, line in result.constructors:
        index = seen.get(class_name, 0)
        seen[class_name] = index + 1
        options = produced.get(class_name, [])
        if typed in options or index >= len(options):
            continue
        notes.append({"line": line, "message": f"self.{attribute} was changed to {options[index]} to match the components connected to it."})
    return notes


def reference() -> list[dict[str, str]]:
    return [{"brick": c.brick, "name": name, "signature": c.signature} for name, c in COMPONENTS.items()]
