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

- **Home** shows your workspace: totals, the newest training runs from every dataset with their status and scores, your builds and datasets, and what this computer trains on.
- **Data labeller** holds your datasets. Each one has its own categories, images or text, and credits.
- **Model builder** is the canvas and the Design, Code, Test, Train and Export steps.

You can start with either one. To train, choose a dataset on the Train step.

## Predict from a spreadsheet

Create a dataset with the type **Spreadsheet (CSV)** and upload a file on its **Data** tab. On the **Models** tab, choose the column to predict, then train. It predicts a category or a number using gradient-boosted trees from scikit-learn.

Columns likely to make a score misleading are left out by default, with the reason shown: row numbers, IDs, free text, dates and columns that nearly give away the answer. You can turn most of them back on. Exact repeated rows are removed before the split, so the same row can't appear on both sides of it. Check any very high score yourself, as these checks can't catch everything. If your rows are ordered in time, a random hold-back can also make a forecast look better than it will be.

Uploading a new file replaces the table, and any unsaved data is saved as a version first. Files can be up to 50 MB and 200,000 rows.

## Data health and levels

The **Health** tab gives image and text datasets a level:

1. **Started**: at least 2 categories, with enough examples to train.
2. **Solid**: 20 or more examples in each category, no category more than 10 times the size of another, and no example filed under two categories.
3. **Checked**: the label check has run on the current data, and no more than 5% of labels are still flagged.
4. **Reviewed**: every flagged label and near-duplicate pair with different labels has been reviewed.

The label check scores each example with a model that never saw it, using a pretrained image or text model with a simple classifier trained on the rest of the data. Following the confident learning method, it flags examples the model is confidently sure belong to a different category. You can move, keep or remove each one, and nothing changes without your say. It also finds near-duplicates.

The percentage shown is the share of labels that passed the check, not a measure of how many are truly right. In tests where we deliberately mislabelled real data, it caught 65–80% of the mistakes in news headlines and 74–82% in cat and dog photos. About 55–62% of its flags on headlines were real mistakes, and 93–100% on the photos. Published research on public datasets found about half of such flags were real errors. Spreadsheet datasets show table statistics instead, such as repeated rows and blank cells.

## Dataset pages, versions and models

Each dataset has four tabs:

- **Overview** has its description, licence, data sources and a preview of each category.
- **Label** is where you add categories and examples.
- **Versions** saves your data and labels so you can compare or restore them. Restoring saves any unsaved work as a new version first.
- **Models** lists what was trained on this data, with the data version and score of each, and downloads for the newest 10.

Training saves a version automatically when the data has changed, then trains on a frozen copy of it.

## Use your webcam

On a photo dataset's **Label** tab, **Use webcam** opens the camera. Hold a category's button, or focus it and hold Space, to take 4 photos a second straight into that category. The camera only runs while the panel is open, and photos are saved on this computer like any others, so versions and the label check include them.

**Try it with your webcam** appears under the newest photo model on the Models tab and after training in the Model builder. It shows the model's guesses live. The model stays loaded in one background process while you use it, so each frame is answered in milliseconds rather than the second or so it takes to start Python, and it closes after 2 minutes unused.

## See what a model looked at

**Where did it look?** appears on a photo you try, on each held-back photo the model got wrong, and as **Show where it looks** on the webcam. It shades the photo so the parts the model used glow and the rest is dimmed. Use it to catch a model that learned the background instead of the object, such as every mug photo being taken on the same table.

It uses Grad-CAM (Selvaraju et al., 2017) on the last layer that still keeps the photo's layout. Models with no such layer, such as the vision transformer, use pixel gradients instead, which are coarser. The original Grad-CAM study found these maps helped people with no machine-learning training tell a stronger model from a weaker one.

For text, **Which words mattered?** removes each word in turn and measures how far the top answer's chance falls without it. Words that pushed toward the answer are shaded blue and words that pushed against it orange. Only the first 80 words are checked.

Both show where the model looked, not proof of why it answered. A model can look in the right place and still be wrong.

## Compare models and read their cards

Every run holds back about 20% of the data to check the model on. The held-back examples are picked by a fingerprint of each photo, text or row rather than at random, so models trained on the same dataset are checked on mostly the same examples, even after you add data.

**Compare two models** on the Models tab checks them only on the examples both held back, which neither learned from. It counts the examples one got right and the other missed, and uses McNemar's exact test (Dietterich, 1998) to say whether the gap is likely to be luck. Models that predict numbers are compared by which was closer on each row. Models trained before this feature was added need training again to be compared.

**Card** shows a model card for any finished model, with sections from Mitchell et al. (2019): details, intended use, training data, how it was checked, results with a likely range for the accuracy, and limitations worked out from the results, such as weak categories, uneven data or a small check. You write the intended use, and downloads include the card as `MODEL_CARD.md`.

## Sort sounds

Create a dataset with the type **Sound**. It starts with a **Background noise** category, because without one the model has to pick one of your sounds even when nothing is happening.

On the **Label** tab, **Use microphone** and hold a category's button while making the sound. Every second you hold becomes one 1-second clip. Background categories also get **Record 20 s** for one long, steady recording of the room. You can add WAV files too, and listen to every clip to remove bad ones. Recordings are saved as 16 kHz mono WAV, and only WAV files can be added for now.

On the **Models** tab, **Start training** needs at least 4 clips per category, and 20 or more gives steadier scores. Each clip becomes a spectrogram, a picture of which pitches are loud at each moment. The pretrained [Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778), which learned from about two million AudioSet clips, reads that picture and stays frozen. A small classifier learns your categories from what it reports, using each clip and two time-shifted, louder or quieter copies. The transformer was made for 10-second clips. For 1-second clips its table of positions is cut to the first 1.28 seconds, as the AST paper does for shorter inputs. The first run downloads it, about 350 MB, from Hugging Face.

**Listen with the microphone** then shows the model's guess for the last second, about twice a second, with the spectrogram it heard. **Show where it listens** highlights the moments and pitches that mattered most. Mistakes can be played back and moved, and downloads include `predict.py` for WAV files. The label check isn't available for sound yet, so sound datasets reach level 2 on the Health tab at most.

The model hears the room as well as the sound. Record in the places you'll use it, and expect it to do worse with a different microphone.

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
- `backend/aibricks/sounds.py` stores sound clips and trains the sound sorter.
- `backend/aibricks/explain.py` makes the heatmaps and word weights that show where a model looked.
- `backend/aibricks/system.py` reports the training device and workspace size for the home screen.

## Before hosting it publicly

The API binds to localhost. Test runs execute only generated code, and user text enters that
code only as validated string literals. A public deployment still needs container isolation,
no network access for test runs, per-user image storage, memory limits, rate limiting and shared storage for the
record of passed builds, which is currently held in memory.
