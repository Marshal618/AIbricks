# AI Bricks

Design a neural network from typed, composable bricks. AI Bricks validates the build, tests it
on sample data and generates a readable PyTorch project to train on your own data.

This covers Phases 1 and 2 of the roadmap: CNN, LSTM and transformer bricks, image, time-series
and text data sockets, pretrained language and vision models,
code generation, the test run and the project download.

## Beginner and Advanced modes

Beginner mode shows goal-based templates, a smaller component list, the settings most people
need and plain explanations. Advanced mode shows every component and setting, custom training
values and an editor for `model.py`.

**Convert to blocks** rebuilds the canvas from `model.py`. The code is parsed with Python's `ast`
module and never run. It must use AI Bricks components, so custom Python belongs in the exported
project.

## Run it

```
./run.sh
```

Then open http://127.0.0.1:5173. The first run installs PyTorch and the editor's packages,
which takes a few minutes.

## Test it

```
cd backend
PYTHONPATH=. .venv/bin/pytest tests -q
```

## How it fits together

- `frontend/` is the editor, built with React and React Flow.
- `backend/aibricks/catalogue.py` defines every brick, connector type and starter kit.
- `backend/aibricks/graph.py` validates a build and works out every shape.
- `backend/aibricks/codegen.py` and `brick_sources.py` write the PyTorch project.
- `backend/aibricks/code_parser.py` turns `model.py` back into blocks without running it.
- `backend/aibricks/runner.py` runs the test in a separate, time-limited process.

## Before hosting it publicly

The API binds to localhost. Test runs execute only generated code, and user text enters that
code only as validated string literals. A public deployment still needs container isolation,
no network access for test runs, memory limits, rate limiting and shared storage for the
record of passed builds, which is currently held in memory.
