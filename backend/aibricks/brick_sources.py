"""Readable PyTorch source for each brick. Only the bricks a build uses are written to its bricks.py."""
from .catalogue import WRITERS

_WRITER_TABLE = "\n".join(
    f"        {name!r}: {{'repo': {spec['repo']!r}, 'hidden': {spec['hidden']}, 'heads': {spec['heads']}, "
    f"'kv_heads': {spec['kv_heads']}, 'ffn': {spec['ffn']}, 'vocab': {spec['vocab']}}},"
    for name, spec in WRITERS.items()
)

_RECURRENT = '''class @@CLASS@@(nn.Module):
    """@@DOC@@"""

    def __init__(self, in_features, hidden, layers=1, bidirectional=False):
        super().__init__()
        self.layer = nn.@@TORCH@@(
            in_features, hidden, num_layers=layers,
            bidirectional=bidirectional, batch_first=True,
        )

    def forward(self, x):
        output, _ = self.layer(x)
        return output
'''

SOURCES = {
    "conv": ("ConvBlock", '''class ConvBlock(nn.Module):
    """Convolution: scans the image for small visual patterns."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)
'''),
    "pool": ("Pool", '''class Pool(nn.Module):
    """Pooling: halves the image size and keeps the strongest (or average) signals."""

    def __init__(self, mode="max"):
        super().__init__()
        self.pool = nn.MaxPool2d(2) if mode == "max" else nn.AvgPool2d(2)

    def forward(self, x):
        return self.pool(x)
'''),
    "residual": ("ResidualBlock", '''class ResidualBlock(nn.Module):
    """Residual block: two convolutions plus a shortcut, which helps deep models train."""

    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.layers(x))
'''),
    "upsample": ("Upsample", '''class Upsample(nn.Module):
    """Upsampling: doubles the image size."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)
'''),
    "backbone": ("ResNet18Backbone", '''class ResNet18Backbone(nn.Module):
    """Pretrained image backbone: ResNet-18 from torchvision, without its final layers."""

    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        try:
            model = resnet18(weights=weights)
        except URLError as error:
            raise RuntimeError(
                "Could not download the pretrained ResNet-18 weights. Check your internet connection. "
                "If the error mentions a certificate and you use the python.org installer on a Mac, "
                'run "Install Certificates.command" in your Python folder, then try again.'
            ) from error
        # Drop the average pool and the 1000-class output: your own bricks replace them.
        self.features = nn.Sequential(*list(model.children())[:-2])
        self.freeze = freeze
        if freeze:
            for parameter in self.features.parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.features.eval()  # keep frozen batch-norm statistics unchanged
        return self

    def forward(self, x):
        return self.features(x)
'''),
    "cnn_decoder": ("CNNDecoder", '''class CNNDecoder(nn.Module):
    """CNN decoder: turns a summary back into an image of the requested size."""

    def __init__(self, in_features, channels, height, width):
        super().__init__()
        self.channels = channels
        self.output_size = (height, width)
        self.project = nn.Linear(in_features, channels * 4 * 4)
        layers = []
        size = 4
        while size < max(height, width):
            layers += [
                nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            ]
            size *= 2
        self.grow = nn.Sequential(*layers)

    def forward(self, x):
        x = self.project(x).view(-1, self.channels, 4, 4)
        x = self.grow(x)
        return nn.functional.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)
'''),
    "lstm": ("LSTMLayer", _RECURRENT.replace("@@CLASS@@", "LSTMLayer").replace("@@TORCH@@", "LSTM").replace("@@DOC@@", "LSTM layer: reads a sequence and remembers long-range patterns.")),
    "gru": ("GRULayer", _RECURRENT.replace("@@CLASS@@", "GRULayer").replace("@@TORCH@@", "GRU").replace("@@DOC@@", "GRU layer: a lighter LSTM that often trains faster.")),
    "rnn": ("RNNLayer", _RECURRENT.replace("@@CLASS@@", "RNNLayer").replace("@@TORCH@@", "RNN").replace("@@DOC@@", "RNN layer: the simplest sequence reader, best for short patterns.")),
    "lstm_decoder": ("LSTMDecoder", '''class LSTMDecoder(nn.Module):
    """LSTM decoder: unrolls a summary into a sequence of the requested length."""

    def __init__(self, in_features, hidden, steps):
        super().__init__()
        self.steps = steps
        self.layer = nn.LSTM(in_features, hidden, batch_first=True)

    def forward(self, x):
        repeated = x.unsqueeze(1).repeat(1, self.steps, 1)
        output, _ = self.layer(repeated)
        return output
'''),
    "global_pool": ("GlobalPool", '''class GlobalPool(nn.Module):
    """Global pooling: averages each channel of an image into a single number."""

    def forward(self, x):
        return x.mean(dim=(2, 3))
'''),
    "flatten": ("Flatten", '''class Flatten(nn.Module):
    """Flatten: unrolls every pixel into one long list of numbers."""

    def forward(self, x):
        return x.flatten(start_dim=1)
'''),
    "last_step": ("LastStep", '''class LastStep(nn.Module):
    """Last step: keeps only the final real step of a sequence, skipping any padding."""

    def forward(self, x, padding_mask=None):
        if padding_mask is None:
            return x[:, -1, :]
        last = (~padding_mask).sum(dim=1).clamp_min(1) - 1
        return x[torch.arange(len(x), device=x.device), last]
'''),
    "mean_pool": ("MeanPool", '''class MeanPool(nn.Module):
    """Mean pooling: averages every real step of a sequence, skipping any padding."""

    def forward(self, x, padding_mask=None):
        if padding_mask is None:
            return x.mean(dim=1)
        keep = (~padding_mask).unsqueeze(-1).to(x.dtype)
        return (x * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1)
'''),
    "embedding": ("TokenEmbedding", '''class TokenEmbedding(nn.Module):
    """Embedding: turns each token number into a learned vector. Token 0 is padding."""

    def __init__(self, vocab_size, size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, size, padding_idx=0)

    def forward(self, x):
        return self.embedding(x)
'''),
    "positional": ("PositionalEncoding", '''class PositionalEncoding(nn.Module):
    """Positional encoding: adds a fixed wave pattern so attention knows the token order."""

    def __init__(self, size, max_length):
        super().__init__()
        position = torch.arange(max_length).unsqueeze(1)
        frequency = torch.exp(torch.arange(0, size, 2) * (-math.log(10000.0) / size))
        pattern = torch.zeros(max_length, size)
        pattern[:, 0::2] = torch.sin(position * frequency)
        pattern[:, 1::2] = torch.cos(position * frequency)[:, : size // 2]
        self.register_buffer("pattern", pattern.unsqueeze(0))

    def forward(self, x):
        return x + self.pattern[:, : x.size(1)]
'''),
    "attention": ("AttentionBlock", '''class AttentionBlock(nn.Module):
    """Attention block: each position looks at every other position."""

    def __init__(self, size, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(size, heads, batch_first=True)
        self.norm = nn.LayerNorm(size)

    def forward(self, x, padding_mask=None):
        attended, _ = self.attention(x, x, x, key_padding_mask=padding_mask, need_weights=False)
        return self.norm(x + attended)
'''),
    "transformer_encoder": ("TransformerEncoder", '''class TransformerEncoder(nn.Module):
    """Transformer encoder: stacked self-attention layers that read the whole sequence."""

    def __init__(self, size, heads, layers):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            size, heads, dim_feedforward=size * 4, activation="gelu", batch_first=True, norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, x, padding_mask=None):
        return self.layers(x, src_key_padding_mask=padding_mask)
'''),
    "decoder_only": ("TransformerDecoderOnly", '''class TransformerDecoderOnly(nn.Module):
    """Transformer decoder (GPT-style): each position sees only the tokens before it."""

    def __init__(self, size, heads, layers):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            size, heads, dim_feedforward=size * 4, activation="gelu", batch_first=True, norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, x, padding_mask=None):
        steps = x.size(1)
        future = torch.triu(torch.ones(steps, steps, dtype=torch.bool, device=x.device), diagonal=1)
        return self.layers(x, mask=future, src_key_padding_mask=padding_mask)
'''),
    "decoder_memory": ("TransformerDecoder", '''class TransformerDecoder(nn.Module):
    """Transformer decoder (T5-style): reads the encoder and sees only earlier output tokens."""

    def __init__(self, size, heads, layers, memory_size):
        super().__init__()
        # If the encoder's size differs, this adapter converts it.
        self.adapt_memory = nn.Identity() if memory_size == size else nn.Linear(memory_size, size)
        layer = nn.TransformerDecoderLayer(
            size, heads, dim_feedforward=size * 4, activation="gelu", batch_first=True, norm_first=True,
        )
        self.layers = nn.TransformerDecoder(layer, layers)

    def forward(self, x, memory, padding_mask=None, memory_padding_mask=None):
        steps = x.size(1)
        future = torch.triu(torch.ones(steps, steps, dtype=torch.bool, device=x.device), diagonal=1)
        return self.layers(
            x, self.adapt_memory(memory), tgt_mask=future,
            tgt_key_padding_mask=padding_mask, memory_key_padding_mask=memory_padding_mask,
        )
'''),
    "pretrained_language": ("PretrainedLanguageModel", '''class PretrainedLanguageModel(nn.Module):
    """Pretrained language model: DistilBERT from Hugging Face. Token 0 is padding."""

    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        from transformers import DistilBertConfig, DistilBertModel

        if pretrained:
            try:
                self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            except OSError as error:
                raise RuntimeError(
                    "Could not download DistilBERT from Hugging Face. Check your internet connection and try again."
                ) from error
        else:
            self.model = DistilBertModel(DistilBertConfig())
        self.freeze = freeze
        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.model.eval()  # keep frozen layers free of dropout
        return self

    def forward(self, x):
        return self.model(input_ids=x, attention_mask=(x != 0).long()).last_hidden_state
'''),
    "pretrained_writer": ("PretrainedWriter", '''class PretrainedWriter(nn.Module):
    """Pretrained writer: a small open model from Hugging Face that already writes text.

    Only its last few layers change while training, so it keeps what it already knows.
    """

    # Where each model downloads from, and its sizes.
    MODELS = {
@@WRITERS@@
    }

    def __init__(self, name, adjust_layers=2, pretrained=True):
        super().__init__()
        from transformers import AutoConfig, AutoModelForCausalLM

        spec = self.MODELS[name]
        self.repo = spec["repo"]
        if pretrained:
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.repo, dtype=torch.float32)
            except OSError as error:
                raise RuntimeError(
                    f"Could not download {name} from Hugging Face. Check your internet connection and try again. "
                    "After the first download it works offline."
                ) from error
        else:
            # A stand-in with the same sizes but only 2 layers, so the test run needs no download.
            config = AutoConfig.for_model(
                "llama", hidden_size=spec["hidden"], num_attention_heads=spec["heads"], num_key_value_heads=spec["kv_heads"],
                intermediate_size=spec["ffn"], vocab_size=spec["vocab"], num_hidden_layers=2, tie_word_embeddings=True,
            )
            self.model = AutoModelForCausalLM.from_config(config)
        # Some models, such as Granite, divide their word scores by a fixed number.
        self.logits_scaling = float(getattr(self.model.config, "logits_scaling", None) or 1.0)
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        body = self.model.base_model
        for part in [*body.layers[-adjust_layers:], body.norm]:
            for parameter in part.parameters():
                parameter.requires_grad = True

    def forward(self, x):
        # What the model makes of each position. Generate text turns this into a score for every word.
        return self.model.base_model(input_ids=x).last_hidden_state
'''.replace("@@WRITERS@@", _WRITER_TABLE)),
    "pretrained_vision": ("PretrainedVisionTransformer", '''class PretrainedVisionTransformer(nn.Module):
    """Pretrained vision transformer: ViT-B/16 from torchvision, without its final layer."""

    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        try:
            self.model = vit_b_16(weights=weights)
        except URLError as error:
            raise RuntimeError(
                "Could not download the pretrained ViT-B/16 weights. Check your internet connection. "
                "If the error mentions a certificate and you use the python.org installer on a Mac, "
                'run "Install Certificates.command" in your Python folder, then try again.'
            ) from error
        self.model.heads = nn.Identity()
        self.freeze = freeze
        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.freeze:
            self.model.eval()  # keep frozen layers free of dropout
        return self

    def forward(self, x):
        return self.model(x)
'''),
    "generate": ("GenerateHead", '''class GenerateHead(nn.Module):
    """Generate text: gives a score for every word in the vocabulary at each position."""

    def __init__(self, in_features, vocab_size):
        super().__init__()
        self.output = nn.Linear(in_features, vocab_size)

    def forward(self, x):
        return self.output(x)
'''),
    "generate_writer": ("PretrainedWriterHead", '''class PretrainedWriterHead(nn.Module):
    """Generate text with a pretrained writer: reuses the writer's own word scores, so it can write from the start."""

    def __init__(self, writer):
        super().__init__()
        self.output = writer.model.get_output_embeddings()  # shared with the writer, not a copy
        self.scaling = writer.logits_scaling

    def forward(self, x):
        return self.output(x) / self.scaling
'''),
    "dense": ("DenseLayer", '''class DenseLayer(nn.Module):
    """Dense layer: mixes a summary's numbers together."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(in_features, out_features), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layers(x)
'''),
    "dropout": ("Dropout", '''class Dropout(nn.Module):
    """Dropout: randomly mutes numbers during training to reduce overfitting."""

    def __init__(self, probability):
        super().__init__()
        self.dropout = nn.Dropout(probability)

    def forward(self, x):
        return self.dropout(x)
'''),
    "combine": ("Combine", '''class Combine(nn.Module):
    """Combine: joins two summaries into one longer summary."""

    def forward(self, a, b):
        return torch.cat([a, b], dim=1)
'''),
    "classify": ("ClassifyHead", '''class ClassifyHead(nn.Module):
    """Classify: outputs one score per category. The highest score is the prediction."""

    def __init__(self, in_features, num_classes):
        super().__init__()
        self.output = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.output(x)
'''),
    "predict_number": ("PredictNumberHead", '''class PredictNumberHead(nn.Module):
    """Predict a number: outputs a single value for each input."""

    def __init__(self, in_features):
        super().__init__()
        self.output = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.output(x)
'''),
    "forecast_summary": ("ForecastHead", '''class ForecastHead(nn.Module):
    """Forecast next steps: predicts all future values at once from a summary."""

    def __init__(self, in_features, horizon, num_features):
        super().__init__()
        self.horizon = horizon
        self.num_features = num_features
        self.output = nn.Linear(in_features, horizon * num_features)

    def forward(self, x):
        return self.output(x).view(-1, self.horizon, self.num_features)
'''),
    "forecast_series": ("SequenceForecastHead", '''class SequenceForecastHead(nn.Module):
    """Forecast next steps: turns each decoded step into that step's predicted values."""

    def __init__(self, in_features, num_features):
        super().__init__()
        self.output = nn.Linear(in_features, num_features)

    def forward(self, x):
        return self.output(x)
'''),
    "rebuild_image": ("RebuildImageHead", '''class RebuildImageHead(nn.Module):
    """Detect anomalies: rebuilds the input image. Poorly rebuilt images are unusual."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.output = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.output(x)
'''),
    "rebuild_series": ("RebuildSeriesHead", '''class RebuildSeriesHead(nn.Module):
    """Detect anomalies: rebuilds the input sequence. Poorly rebuilt sequences are unusual."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.output = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.output(x)
'''),
}
