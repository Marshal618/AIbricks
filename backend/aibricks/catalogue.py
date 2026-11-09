"""The brick catalogue: every brick, its typed ports, its settings and the starter kits.

This file is the single source of truth. The editor reads it over the API, and the
graph checker and code generator read it directly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

IMAGE = "image"
SERIES = "series"
SUMMARY = "summary"
TEXT = "text"

PORT_TYPES = {
    IMAGE: {"name": "Image", "description": "A grid of pixels or image features."},
    SERIES: {"name": "Time series", "description": "Values that unfold step by step."},
    SUMMARY: {"name": "Summary", "description": "A compact list of numbers describing the input."},
    TEXT: {"name": "Text", "description": "Text converted to token numbers."},
}


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    kind: str  # "int" | "choice" | "bool" | "names"
    default: Any
    help: str = ""
    min: int | None = None
    max: int | None = None
    choices: tuple[Any, ...] | None = None
    # names only: how many comma-separated entries are allowed.
    min_items: int = 1
    max_items: int = 64
    # Shown only while this other on/off setting is on.
    shown_when: str | None = None


@dataclass(frozen=True)
class Port:
    id: str
    accepts: tuple[str, ...]
    label: str = ""
    optional: bool = False


@dataclass(frozen=True)
class Part:
    brick: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Brick:
    id: str
    name: str
    group: str
    description: str
    inputs: tuple[Port, ...] = ()
    output: str | None = None
    params: tuple[Param, ...] = ()
    size: str = "small"  # "small" | "big"
    parts: tuple[Part, ...] = ()  # big bricks only


def _in(*types: str) -> tuple[Port, ...]:
    return (Port("in", tuple(types)),)


GROUPS = [
    {"id": "socket", "name": "Data inputs", "label": "Input"},
    {"id": "big", "name": "Composite modules", "label": "Composite"},
    {"id": "cnn", "name": "Convolutional", "label": "Convolutional"},
    {"id": "rnn", "name": "Recurrent", "label": "Recurrent"},
    {"id": "transformer", "name": "Transformer", "label": "Transformer"},
    {"id": "connector", "name": "Adapters", "label": "Adapter"},
    {"id": "head", "name": "Task heads", "label": "Task head"},
]

FILTERS = Param("filters", "Feature channels", "int", 32, "More channels can spot more patterns but run slower.", 1, 1024)
HIDDEN = Param("hidden", "Memory size", "int", 64, "How many numbers the layer remembers at each step.", 1, 2048)
LAYERS = Param("layers", "Stacked layers", "int", 1, "Layers stacked inside this brick.", 1, 4)
BIDIRECTIONAL = Param("bidirectional", "Read both directions", "bool", False, "Also reads the sequence backwards. Not suitable when the brick must only see the past.")
CLASSES = Param("classes", "Categories", "names", "Cat, Dog", "Comma-separated category names.", min_items=2, max_items=100)
HEADS = Param("heads", "Attention heads", "int", 4, "Parallel attention views. The input size must divide evenly by this number.", 1, 32)
BLOCKS = Param("layers", "Layers", "int", 2, "Stacked transformer layers.", 1, 12)
FREEZE = Param("freeze", "Keep its weights fixed", "bool", True, "Train only the layers you added. Recommended for small datasets.")
HORIZON = Param("horizon", "Steps to forecast", "int", 24, "How many future steps to predict.", 1, 1024)

_BRICKS = [
    # Data sockets
    Brick("image_folder", "Image folder", "socket", "Folder of images, loaded and resized for the model.", output=IMAGE, params=(
        Param("image_size", "Image size (pixels)", "int", 64, "Every image is resized to this square size.", 16, 512),
        Param("channels", "Colour channels", "choice", 3, "3 for colour, 1 for greyscale.", choices=(1, 3)),
        Param("augment", "Vary training photos", "bool", False,
              "Changes each training photo a little every time the model sees it, so it learns the object, not one exact picture. "
              "Held-back photos are never changed."),
        Param("vary_flip", "Mirror left and right", "bool", True, "Leave this off if left and right matter, as with arrows or writing.", shown_when="augment"),
        Param("vary_crop", "Crop and zoom (up to %)", "int", 30, "How much of each photo may be cut away. 0 keeps the whole photo.", 0, 60, shown_when="augment"),
        Param("vary_rotate", "Turn (up to degrees)", "int", 0, "Useful when things can appear at an angle.", 0, 45, shown_when="augment"),
        Param("vary_light", "Light and colour (up to %)", "int", 25, "Brighter, darker and more or less colourful, as in different rooms.", 0, 60, shown_when="augment"),
        Param("vary_auto", "Automatic mix", "bool", False,
              "Also adds one random change of a random strength to each photo, such as sharpening or a colour shift. "
              "This is TrivialAugment, which works well without tuning.", shown_when="augment"),
    )),
    Brick("time_series", "Time series", "socket", "CSV table of time-ordered measurements.", output=SERIES, params=(
        Param("columns", "Measurement columns", "names", "value", "Comma-separated column names in your CSV file.", max_items=256),
        Param("sequence_length", "Steps per sample", "int", 48, "How many past rows the model sees at once.", 2, 2048),
    )),
    Brick("text_input", "Text", "socket", "CSV or text file, converted to token numbers.", output=TEXT, params=(
        Param("vocab_size", "Vocabulary size", "int", 8000, "How many distinct words the model knows. Others become unknown.", 16, 200000),
        Param("max_length", "Maximum length (tokens)", "int", 64, "Longer text is cut off and shorter text is padded.", 2, 512),
    )),
    Brick("target_text", "Target text", "socket", "The text a translation or summary model learns to produce.", output=TEXT, params=(
        Param("vocab_size", "Vocabulary size", "int", 8000, "How many distinct output words the model knows.", 16, 200000),
        Param("max_length", "Maximum length (tokens)", "int", 64, "Longer outputs are cut off during training.", 2, 512),
    )),
    # Big bricks
    Brick("cnn_encoder", "CNN encoder", "big", "Compresses an image into a summary.", _in(IMAGE), SUMMARY, size="big", parts=(
        Part("conv", {"filters": 32}), Part("pool"), Part("conv", {"filters": 64}), Part("pool"), Part("global_pool"),
    )),
    Brick("lstm_encoder", "LSTM encoder", "big", "Compresses a time series into a summary.", _in(SERIES), SUMMARY, (HIDDEN,), size="big", parts=(
        Part("lstm"), Part("last_step"),
    )),
    Brick("image_recogniser", "Image recogniser", "big", "A complete model that sorts images into categories.", _in(IMAGE), None, (CLASSES,), size="big", parts=(
        Part("conv", {"filters": 32}), Part("pool"), Part("conv", {"filters": 64}), Part("pool"), Part("global_pool"), Part("classify"),
    )),
    Brick("forecaster", "Time-series forecaster", "big", "A complete model that predicts upcoming values.", _in(SERIES), None, (HORIZON,), size="big", parts=(
        Part("lstm"), Part("lstm"), Part("last_step"), Part("forecast"),
    )),
    Brick("transformer_text_encoder", "Transformer text encoder", "big", "Compresses text into a summary with self-attention.", _in(TEXT), SUMMARY, (HEADS,), size="big", parts=(
        Part("embedding"), Part("positional"), Part("transformer_encoder"), Part("mean_pool"),
    )),
    Brick("text_classifier", "Text classifier", "big", "A complete transformer that sorts text into categories (BERT-style).", _in(TEXT), None, (CLASSES,), size="big", parts=(
        Part("embedding"), Part("positional"), Part("transformer_encoder"), Part("mean_pool"), Part("classify"),
    )),
    # CNN
    Brick("conv", "Convolution", "cnn", "Scans the image for small visual patterns.", _in(IMAGE), IMAGE, (
        FILTERS,
        Param("kernel_size", "Pattern size", "choice", 3, "Width of the pattern it scans for, in pixels.", choices=(3, 5, 7)),
    )),
    Brick("pool", "Pooling", "cnn", "Halves the image size and keeps the strongest signals.", _in(IMAGE), IMAGE, (
        Param("mode", "Keep", "choice", "max", "Keep the strongest signal or the average.", choices=("max", "average")),
    )),
    Brick("residual", "Residual block", "cnn", "Two convolutions with a shortcut, which helps deeper models train.", _in(IMAGE), IMAGE),
    Brick("upsample", "Upsampling", "cnn", "Doubles the image size.", _in(IMAGE), IMAGE, (FILTERS,)),
    Brick("backbone", "Pretrained image backbone", "cnn", "ResNet-18, already trained on millions of photos.", _in(IMAGE), IMAGE, (
        Param("freeze", "Keep its weights fixed", "bool", True, "Train only the layers you added. Recommended for small datasets."),
    )),
    Brick("cnn_decoder", "CNN decoder", "cnn", "Turns a summary back into an image the same size as your input.", _in(SUMMARY), IMAGE, (FILTERS,)),
    # RNN family
    Brick("lstm", "LSTM layer", "rnn", "Reads a sequence and remembers long-range patterns.", _in(SERIES), SERIES, (HIDDEN, LAYERS, BIDIRECTIONAL)),
    Brick("gru", "GRU layer", "rnn", "A lighter LSTM that often trains faster.", _in(SERIES), SERIES, (HIDDEN, LAYERS, BIDIRECTIONAL)),
    Brick("rnn", "RNN layer", "rnn", "The simplest sequence reader, best for short patterns.", _in(SERIES), SERIES, (HIDDEN, LAYERS, BIDIRECTIONAL)),
    Brick("lstm_decoder", "LSTM decoder", "rnn", "Unrolls a summary into a sequence.", _in(SUMMARY), SERIES, (
        HIDDEN,
        Param("steps", "Steps to generate", "int", 0, "Use 0 to match your input length.", 0, 2048),
    )),
    # Transformers
    Brick("positional", "Positional encoding", "transformer", "Tells attention layers the order of the tokens.", _in(SERIES), SERIES),
    Brick("attention", "Attention block", "transformer", "One self-attention layer, where each position looks at the others.", _in(SERIES), SERIES, (HEADS,)),
    Brick("transformer_encoder", "Transformer encoder", "transformer", "Stacked self-attention layers that read the whole sequence.", _in(SERIES), SERIES, (HEADS, BLOCKS)),
    Brick("transformer_decoder", "Transformer decoder", "transformer", "Generates a sequence one token at a time, seeing only earlier tokens. Connect an encoder for translation.",
          (Port("in", (SERIES,), "Tokens"), Port("memory", (SERIES,), "Encoder (optional)", optional=True)), SERIES, (HEADS, BLOCKS)),
    Brick("pretrained_language", "Pretrained language model", "transformer", "DistilBERT, already trained on English text. Uses its own tokenizer.", _in(TEXT), SERIES, (FREEZE,)),
    Brick("pretrained_vision", "Pretrained vision transformer", "transformer", "ViT-B/16, already trained on photos. Needs 224×224 colour images.", _in(IMAGE), SUMMARY, (FREEZE,)),
    # Connectors
    Brick("embedding", "Embedding", "connector", "Turns each token number into a learned vector.", _in(TEXT), SERIES, (
        Param("size", "Vector size", "int", 128, "Numbers per token. Must divide evenly by the attention heads.", 2, 4096),
    )),
    Brick("mean_pool", "Mean pooling", "connector", "Averages every step of a sequence into a summary, ignoring padding.", _in(SERIES), SUMMARY),
    Brick("global_pool", "Global pooling", "connector", "Averages a whole image into a summary. Light and robust.", _in(IMAGE), SUMMARY),
    Brick("flatten", "Flatten", "connector", "Unrolls every pixel into a long summary.", _in(IMAGE), SUMMARY),
    Brick("last_step", "Last step", "connector", "Keeps only the final step of a sequence as a summary.", _in(SERIES), SUMMARY),
    Brick("dense", "Dense layer", "connector", "Mixes a summary's numbers together.", _in(SUMMARY), SUMMARY, (
        Param("size", "Output size", "int", 128, "How many numbers come out.", 1, 8192),
    )),
    Brick("dropout", "Dropout", "connector", "Randomly mutes numbers during training to reduce overfitting.", _in(SUMMARY), SUMMARY, (
        Param("percent", "Muted (%)", "int", 20, "Share of numbers muted at each step.", 0, 90),
    )),
    Brick("combine", "Combine", "connector", "Joins two summaries into one.", (Port("a", (SUMMARY,), "A"), Port("b", (SUMMARY,), "B")), SUMMARY),
    # Task heads
    Brick("classify", "Classify", "head", "Picks one category for each input.", _in(SUMMARY), None, (CLASSES,)),
    Brick("predict_number", "Predict a number", "head", "Outputs one number for each input.", _in(SUMMARY)),
    Brick("forecast", "Forecast next steps", "head", "Predicts the values that come next.", _in(SUMMARY, SERIES), None, (HORIZON,)),
    Brick("generate", "Generate text", "head", "Predicts the next token at every position.", _in(SERIES)),
    Brick("detect_anomalies", "Detect anomalies", "head", "Learns to rebuild normal input, then flags what it cannot rebuild.", _in(IMAGE, SERIES)),
]

BRICKS: dict[str, Brick] = {b.id: b for b in _BRICKS}

TRAINING_DIAL = {
    "quick": {"name": "Quick", "epochs": 5, "batch_size": 32, "learning_rate": 1e-3},
    "balanced": {"name": "Balanced", "epochs": 20, "batch_size": 32, "learning_rate": 1e-3},
    "thorough": {"name": "Thorough", "epochs": 60, "batch_size": 32, "learning_rate": 5e-4},
}


def _node(id: str, type: str, x: int, y: int, **params: Any) -> dict[str, Any]:
    return {"id": id, "type": type, "x": x, "y": y, "params": params}


def _edge(source: str, target: str, handle: str = "in") -> dict[str, str]:
    return {"source": source, "target": target, "targetHandle": handle}


KITS = [
    {
        "id": "defect_detector",
        "domain": "Computer vision",
        "name": "Defect detector",
        "family": "CNN",
        "audience": "Image classification for surface defects on manufactured parts.",
        "projectName": "Defect detector",
        "training": "balanced",
        "nodes": [
            _node("socket", "image_folder", 0, 0, image_size=128),
            _node("backbone", "backbone", 0, 150),
            _node("residual", "residual", 0, 300),
        ],
        "edges": [_edge("socket", "backbone"), _edge("backbone", "residual")],
        "steps": [
            {"text": "Add Global pooling and connect it to the Residual block output.", "check": {"brick": "global_pool"}},
            {"text": "Add Classify and connect it to the Global pooling output.", "check": {"brick": "classify"}},
            {"text": "Select Classify and enter your categories, such as OK, Scratch, Dent.", "check": {"paramChanged": "classify"}},
            {"text": "Run the test.", "check": {"testPassed": True}},
        ],
    },
    {
        "id": "energy_forecaster",
        "domain": "Time series",
        "name": "Energy forecaster",
        "family": "LSTM",
        "audience": "Multi-step forecasting of energy consumption from historical readings.",
        "projectName": "Energy forecaster",
        "training": "thorough",
        "nodes": [
            _node("socket", "time_series", 0, 0, columns="energy_kwh", sequence_length=96),
            _node("lstm1", "lstm", 0, 150),
            _node("lstm2", "lstm", 0, 300),
        ],
        "edges": [_edge("socket", "lstm1"), _edge("lstm1", "lstm2")],
        "steps": [
            {"text": "Add Last step and connect it to the second LSTM layer output.", "check": {"brick": "last_step"}},
            {"text": "Add Forecast next steps and connect it to the Last step output.", "check": {"brick": "forecast"}},
            {"text": "Run the test.", "check": {"testPassed": True}},
        ],
    },
    {
        "id": "sensor_watchdog",
        "domain": "Time series",
        "name": "Sensor watchdog",
        "family": "LSTM autoencoder",
        "audience": "Reconstruction-based anomaly detection for multivariate sensor data.",
        "projectName": "Sensor watchdog",
        "training": "balanced",
        "nodes": [
            _node("socket", "time_series", 0, 0, columns="temperature, vibration", sequence_length=32),
            _node("encoder", "lstm_encoder", 0, 150),
            _node("decoder", "lstm_decoder", 0, 300),
        ],
        "edges": [_edge("socket", "encoder"), _edge("encoder", "decoder")],
        "steps": [
            {"text": "Add Detect anomalies and connect it to the LSTM decoder output.", "check": {"brick": "detect_anomalies"}},
            {"text": "Run the test.", "check": {"testPassed": True}},
        ],
    },
    {
        "id": "ticket_sorter",
        "domain": "Language",
        "name": "Support ticket sorter",
        "family": "Pretrained transformer",
        "audience": "Text classification with DistilBERT.",
        "projectName": "Ticket sorter",
        "training": "quick",
        "nodes": [
            _node("socket", "text_input", 0, 0, vocab_size=30522, max_length=64),
            _node("bert", "pretrained_language", 0, 150),
        ],
        "edges": [_edge("socket", "bert")],
        "steps": [
            {"text": "Add Mean pooling and connect it to the Pretrained language model output.", "check": {"brick": "mean_pool"}},
            {"text": "Add Classify and connect it to the Mean pooling output.", "check": {"brick": "classify"}},
            {"text": "Select Classify and enter your categories, such as Billing, Bug, Question.", "check": {"paramChanged": "classify"}},
            {"text": "Run the test.", "check": {"testPassed": True}},
        ],
    },
    {
        "id": "text_generator",
        "domain": "Language",
        "name": "Text generator",
        "family": "Decoder-only transformer",
        "audience": "GPT-style next-word model trained from scratch on your own text.",
        "projectName": "Text generator",
        "training": "balanced",
        "nodes": [
            _node("socket", "text_input", 0, 0),
            _node("embed", "embedding", 0, 150),
            _node("position", "positional", 0, 300),
            _node("decoder", "transformer_decoder", 0, 450),
        ],
        "edges": [_edge("socket", "embed"), _edge("embed", "position"), _edge("position", "decoder")],
        "steps": [
            {"text": "Add Generate text and connect it to the Transformer decoder output.", "check": {"brick": "generate"}},
            {"text": "Run the test.", "check": {"testPassed": True}},
        ],
    },
    {
        "id": "translator",
        "domain": "Language",
        "name": "Translator",
        "family": "Encoder-decoder transformer",
        "audience": "T5-style sequence-to-sequence model for translation or summarisation.",
        "projectName": "Translator",
        "training": "balanced",
        "nodes": [
            _node("source", "text_input", 0, 0),
            _node("source_embed", "embedding", 0, 150),
            _node("source_position", "positional", 0, 300),
            _node("encoder", "transformer_encoder", 0, 450),
            _node("target", "target_text", 320, 0),
            _node("target_embed", "embedding", 320, 150),
            _node("target_position", "positional", 320, 300),
            _node("decoder", "transformer_decoder", 320, 600),
        ],
        "edges": [
            _edge("source", "source_embed"), _edge("source_embed", "source_position"), _edge("source_position", "encoder"),
            _edge("target", "target_embed"), _edge("target_embed", "target_position"), _edge("target_position", "decoder"),
            _edge("encoder", "decoder", "memory"),
        ],
        "steps": [
            {"text": "Add Generate text and connect it to the Transformer decoder output.", "check": {"brick": "generate"}},
            {"text": "Run the test.", "check": {"testPassed": True}},
        ],
    },
    {
        "id": "vit_classifier",
        "domain": "Computer vision",
        "name": "Vision transformer classifier",
        "family": "Pretrained transformer",
        "audience": "Image classification with ViT-B/16.",
        "projectName": "Vision classifier",
        "training": "quick",
        "nodes": [
            _node("socket", "image_folder", 0, 0, image_size=224),
            _node("vit", "pretrained_vision", 0, 150),
        ],
        "edges": [_edge("socket", "vit")],
        "steps": [
            {"text": "Add Classify and connect it to the Pretrained vision transformer output.", "check": {"brick": "classify"}},
            {"text": "Select Classify and enter your categories.", "check": {"paramChanged": "classify"}},
            {"text": "Run the test.", "check": {"testPassed": True}},
        ],
    },
]


# Beginner mode shows these components, and the settings most people need.
WHEN_TO_USE = {
    "image_folder": "Choose this when your data is a folder of pictures.",
    "time_series": "Choose this when your data is measurements over time, such as hourly energy use.",
    "text_input": "Choose this when your data is written text, such as emails or support tickets.",
    "cnn_encoder": "A ready-made way to understand images. Add a task head after it.",
    "lstm_encoder": "A ready-made way to understand measurements over time. Add a task head after it.",
    "image_recogniser": "The quickest route: a complete model that sorts images into your categories.",
    "forecaster": "The quickest route: a complete model that predicts future values.",
    "transformer_text_encoder": "A ready-made way to understand text. Add a task head after it.",
    "text_classifier": "The quickest route: a complete model that sorts text into your categories.",
    "conv": "Spots small patterns such as edges and textures. Stack a few to spot bigger shapes.",
    "pool": "Shrinks the image so the model runs faster and sees the bigger picture.",
    "backbone": "Reuses a model that has already learned from millions of photos. A good choice when you have few images.",
    "cnn_decoder": "Rebuilds an image, for spotting images that look unusual.",
    "lstm": "Remembers what happened earlier in a sequence.",
    "lstm_decoder": "Rebuilds or extends a sequence, for forecasting or spotting unusual readings.",
    "pretrained_language": "Reuses a model that already understands English. A good choice when you have little text.",
    "transformer_decoder": "Writes text one word at a time.",
    "embedding": "Turns words into numbers the model can work with.",
    "positional": "Tells the model what order the words came in.",
    "global_pool": "Connects image layers to a task head. Use it just before the task head.",
    "last_step": "Connects time-series layers to a task head. Use it just before the task head.",
    "mean_pool": "Connects text layers to a task head. Use it just before the task head.",
    "classify": "Use this when each input belongs to one category, such as OK or Damaged.",
    "predict_number": "Use this when the answer is a number, such as a price.",
    "forecast": "Use this to predict the next values in a time series.",
    "detect_anomalies": "Use this to flag anything that looks different from normal.",
    "generate": "Use this to write new text in the style of your training text.",
}
BASIC_PARAMS = {"image_size", "channels", "augment", "vary_flip", "vary_crop", "vary_rotate", "vary_light", "vary_auto", "columns", "sequence_length", "classes", "horizon", "max_length", "steps"}
GOALS = {
    "defect_detector": "Sort photos into categories",
    "energy_forecaster": "Predict future values from past measurements",
    "sensor_watchdog": "Spot unusual sensor readings",
    "ticket_sorter": "Sort written text into categories",
    "text_generator": "Write new text in the style of my text",
    "translator": "Translate or summarise text",
    "vit_classifier": "Sort photos using a powerful pretrained model",
}


def _brick_json(brick: Brick) -> dict[str, Any]:
    data = asdict(brick)
    data["beginner"] = brick.id in WHEN_TO_USE
    data["why"] = WHEN_TO_USE.get(brick.id, "")
    for param in data["params"]:
        param["basic"] = param["key"] in BASIC_PARAMS
    return data


def catalogue_json() -> dict[str, Any]:
    return {
        "portTypes": PORT_TYPES,
        "groups": GROUPS,
        "bricks": [_brick_json(b) for b in _BRICKS],
        "training": TRAINING_DIAL,
        "kits": [kit | {"goal": GOALS[kit["id"]]} for kit in KITS],
    }
