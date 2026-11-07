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

Datasets are stored in `backend/.workspace/`, which git ignores. Nothing is sent anywhere, though a
pretrained model downloads its weights the first time. The 10 newest trained models are kept for each workspace, with their scores listed on the dataset they came from.
Set `AI_BRICKS_WORKSPACE` to store these files somewhere else.

The Train page also shows which held-back images the model got wrong. Downloads of these models
include `review.py` for the same list and `serve.py`, a small FastAPI web API for the trained model.

## The three screens

- **Home** is where you start, with your recent models and datasets.
- **Data labeller** holds your datasets. Each one has its own categories, images or text, and credits.
- **Model builder** is the canvas and the Design, Code, Test, Train and Export steps.

You can start with either one. To train, choose a dataset on the Train step.

## Predict from a spreadsheet

Create a dataset with the type **Spreadsheet (CSV)** and upload a file on its **Data** tab. On the **Models** tab, choose the column to predict, then train. It predicts a category or a number using gradient-boosted trees from scikit-learn.

Columns likely to make a score misleading are left out by default, with the reason shown: row numbers, IDs, free text, dates and columns that nearly give away the answer. You can turn most of them back on. Exact repeated rows are removed before the split, so the same row can't appear on both sides of it. Check any very high score yourself, as these checks can't catch everything. If your rows are ordered in time, a random hold-back can also make a forecast look better than it will be.

Uploading a new file replaces the table, and any unsaved data is saved as a version first. Files can be up to 50 MB and 200,000 rows.

## Dataset pages, versions and models

Each dataset has four tabs:

- **Overview** has its description, licence, data sources and a preview of each category.
- **Label** is where you add categories and examples.
- **Versions** saves your data and labels so you can compare or restore them. Restoring saves any unsaved work as a new version first.
- **Models** lists what was trained on this data, with the data version and score of each, and downloads for the newest 10.

Training saves a version automatically when the data has changed, then trains on a frozen copy of it.

## Train a text sorter

Open a text dataset, go to **Models** and press **Start training**. It needs at least 4 labelled examples per category. You can then try the model, fix labels on its mistakes and download it as a project that runs on its own.

It follows the [SetFit](https://arxiv.org/abs/2209.11055) method, written directly on PyTorch and Transformers rather than using the `setfit` package: it fine-tunes the small `sentence-transformers/all-MiniLM-L6-v2` model, then trains a classifier on top. The first run downloads the base model, about 90 MB, from Hugging Face. Your text isn't sent anywhere. This model is separate from the block designs on the canvas, which still train text from the export.

Versions share files with the dataset, so they take little extra disk space. Examples removed from a dataset stay in older versions until you delete those versions, which erases them from the app. A trained model can still reflect what it learned from them.

## Create a dataset

A dataset can be filled in two ways, and they can be mixed:

- **Find open data** searches public labelled datasets on Hugging Face, shows each one's licence, and
  copies a sample to this computer. Your search words go to Hugging Face, but your own data doesn't.
  Only public, ungated datasets with a readable category column work. Trained model downloads include
  `DATA_SOURCES.md`, which credits what the dataset imported. Check each licence yourself.
- **Labelling by keyboard** sorts your own images and text: number keys choose a category, S skips,
  X removes and Z undoes. A model trained on the dataset can suggest labels for the remaining images,
  starting with the ones it's least sure about.

Text labels download as `texts.csv`, which goes at `data/texts.csv` in a text project. Mistakes shown on
the Train step can be moved to the right category or removed.

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
- `backend/aibricks/datasets.py` stores datasets and their categories.
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
