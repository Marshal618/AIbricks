# AI Bricks

Bricked up AI

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

## Train on your own images

For models that sort images into categories, the Train page takes your images, trains on this
computer and lets you try the result on a new image. It runs the same generated `train.py` and
`predict.py` that the download contains. Other models train from the downloaded project.

Images and text labels are stored in `backend/.workspace/`, which git ignores. Nothing is sent anywhere, though a
pretrained model downloads its weights the first time. Only the latest trained model is kept.
Set `AI_BRICKS_WORKSPACE` to store these files somewhere else.

The Train page also shows which held-back images the model got wrong. Downloads of these models
include `review.py` for the same list and `serve.py`, a small FastAPI web API for the trained model.

## Find open data

The Label page can search public labelled datasets on Hugging Face, show each one's licence, match
its categories to your model's and copy a sample to this computer. Your search words go to Hugging
Face, but your own data doesn't. Only public, ungated datasets work, and they need a category column
that the dataset viewer can read. Trained model downloads include `DATA_SOURCES.md`, which credits
what you imported. Check each dataset's licence yourself before relying on it.

## Label your data

The Label page sorts unlabelled images and text with one key press each: number keys choose a
category, S skips, X removes and Z undoes. Labelled images go straight into the categories that
training reads. A trained model can suggest labels for the remaining images, starting with the
ones it's least sure about. Text labels download as `texts.csv`, which goes at `data/texts.csv` in
a text project. Mistakes on the Train page can also be moved to the right category or removed.

## Saved projects

Builds save automatically to `backend/.workspace/projects/`, and the Projects dialog opens them
again. A project can also be saved to a file and opened on another computer. Files contain the
build only, not images or trained models.

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
- `backend/aibricks/labelling.py` stores unsorted images and text, and makes label suggestions.
- `backend/aibricks/opendata.py` searches and imports open datasets from Hugging Face.
- `backend/aibricks/projects.py` stores saved projects as JSON files.
- `backend/aibricks/runner.py` runs the test in a separate, time-limited process.
- `backend/aibricks/training.py` stores your images and runs training and predictions.

## Before hosting it publicly

The API binds to localhost. Test runs execute only generated code, and user text enters that
code only as validated string literals. A public deployment still needs container isolation,
no network access for test runs, per-user image storage, memory limits, rate limiting and shared storage for the
record of passed builds, which is currently held in memory.
