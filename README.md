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

Each dataset page shows its next step at the top, as one sentence and one button. The tabs are numbered in the order you use them:

1. **Add data** is where you name your categories and add and sort examples.
2. **Check** gives the data a level and runs the label check.
3. **Train** trains models and lists them, with the data version and score of each, and downloads for the newest 10.

Two more tabs hold reference information: **About** has the description, licence, data sources and a preview of each category, and **History** saves versions of your data and labels so you can compare or restore them. Restoring saves any unsaved work as a new version first.

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

## Find a better model

On a photo dataset's **Train** tab, **Find a better model** tries several models within the time you choose and keeps the best. Round 1 trains a simple starting point and four versions that each change one thing: varying the training photos, a wider model, starting from a pretrained ResNet-18, or learning more slowly. They all get the same short training and are checked on the same held-back photos, so it can show what each change did, using McNemar's test to separate real differences from luck. Round 2 trains the best two, and a mix of the changes that helped, for three times as long. This is successive halving, the idea behind ASHA (Li et al., 2020).

Only the winner is kept, as a normal model. **Open its blocks** opens it in the Model builder, so you can see exactly what it is and change it. With fewer than 10 held-back photos it can't tell real differences from luck, and it says so.

## Vary training photos

**Vary training photos** on the Image folder block changes each training photo a little every time the model sees it, so it learns the object rather than one exact picture. Once it's on, you choose:

- **Mirror left and right.** Leave it off if left and right matter, as with arrows or writing.
- **Crop and zoom**, up to a percentage.
- **Turn**, up to a number of degrees.
- **Light and colour**, up to a percentage.
- **Automatic mix**, which adds TrivialAugment (Müller and Hutter, 2021): one random change of a random strength per photo, with no tuning needed.

**Preview with my photos** shows one of your photos and eight versions varied as training will vary them. Photos are varied fresh during training, and only the training photos: the held-back ones never change. AI Bricks doesn't save varied copies into datasets, because copies of a training photo landing among the held-back ones is a known way to make a score look far better than it is. Downloads include the same code.

## Spot what's unusual

When creating a photo dataset, choose **Spot what's unusual** to make a model that learns from normal photos only. It has two categories. **Normal** photos are what it learns from: aim for 30, in all the conditions it will see. **Unusual** is optional: a few photos with problems, used only to check it.

It follows PatchCore (Roth et al., 2022), which reached 99% on the MVTec AD factory-defect benchmark. A pretrained ResNet-18 describes every small patch of each normal photo, and an evenly spread sample of those descriptions is kept as a memory. A new photo is unusual when part of it is far from everything in that memory. The heatmap shows which parts.

The line between normal and unusual comes from the training photos only: each half is scored against a memory of the other half, to see how far unseen normal photos usually are. That keeps the held-back photos for an honest check. **How strict should it be?** moves the line and shows the false alarms and catches on the held-back photos at each position. The webcam, photo checks and downloads follow the line you choose.

## Find things in photos

When creating a photo dataset, choose **Find and count things in photos** to make a model that draws a box around each thing it finds and counts them. Name the things to find, add photos (from files or the webcam), then drag a box around every one on **Add data**. Save with Enter; keys 1 to 9 pick the thing. A photo with nothing in it can be marked **Nothing to find here**, which teaches it what not to find. Once a model exists, **Let the model draw first** fills in its guesses as dashed boxes for you to fix.

It fine-tunes a Faster R-CNN detector (Ren et al., 2015) with a MobileNetV3 backbone, pretrained on the COCO photos by torchvision. The first run downloads it, about 75 MB. Training needs 10 done photos and 5 boxes of each thing; aim for 30 boxes of each. A fifth of the photos are held back and checked: how many things it found, how many it missed, its false alarms and the standard mAP@50 score, with the photos it got wrong shown with both your boxes and its own. **Try it** draws live boxes and counts on the webcam, and a slider sets how sure it must be before it draws a box. Versions keep the boxes, so any model can be traced to the exact boxes it learned from.

## Sort poses and hand signs

When creating a photo dataset, choose **Sort body poses or hand signs**, then whether it should look at the whole body or at hands. Add categories, such as Arms up and Squat, or Thumbs up and Peace, and take photos of each, most easily with the webcam. Add a category for doing nothing in particular, so it has an answer when you aren't making any of the others.

MediaPipe (Google, Apache 2.0) finds 33 points on a body, or 21 on each hand, on this computer. The points are measured from the shoulders or the wrist, in units of the body's or hand's size, so where someone stands and how far away they are don't matter. A small network then learns which category the points make. Because it learns from points, not pixels, clothes, light and background don't get in the way, and a few dozen photos are often enough. Training takes seconds. The first run downloads MediaPipe's point-finding model, about 6 MB for bodies or 8 MB for hands, and every trained model keeps its own copy, so the downloaded project works offline.

**Try it** draws the points it finds on a photo or the live webcam, with its answers. When no body or hand is in view, it says so instead of guessing. Photos where nothing could be seen are left out of learning and reported. The held-back photos it got wrong can be moved to another category or removed. The label check and **Find a better model** don't apply to poses, as both look at whole photos.

## Where it fails

Under the newest photo model on the **Train** tab, **Where it fails** looks for kinds of photos the model gets wrong much more often than others, and names them in everyday words, such as “dark photo” or “close-up”. Each group shows how often it's wrong compared with overall, the mix-up it makes most, example photos and what to do, with buttons to add photos or change training.

It follows the idea of Domino (Eyuboglu et al., ICLR 2022). CLIP, an open model that understands photos and short phrases together, describes every held-back photo. Similar photos are grouped, with mistakes counted a little so that groups of mistakes stand out. A group is named by the phrase that fits it best compared with all the other photos. The names are a good guess, so check them against the examples: Domino named groups exactly right about a third of the time. It needs at least 6 held-back mistakes, and it runs on this computer. The first time downloads CLIP, about 600 MB.

## Training check

After a photo model trains, the **Training check** reads its loss chart and names what it sees in plain words: memorising the training photos, too simple to learn, unstable steps, still improving when it stopped, or healthy. Each finding offers fixes. On the Train step of the Model builder, **Vary training photos**, **Learn more slowly** and **Train for longer** change your design and take you to the test.

## Turn a photo model into an app

**Open as an app** and **Download app** appear under the newest photo model on the Models tab and on the Train step. They make a single web page that runs the model by itself in any browser, with the camera or a chosen photo. Photos are checked on the device and never uploaded.

The page can follow simple rules: when it's at least a chosen percentage sure of a category for half a second, it can play a beep, show a message, count, or send the result to a web address you type, such as an [ntfy](https://ntfy.sh) topic. Rules are saved in that browser.

The model is exported to [ONNX](https://onnx.ai), an open format for trained models, and run by ONNX Runtime Web. That runtime loads from a pinned CDN version, checked against a fixed hash, so the page needs internet the first time it opens. When it opens, it runs a fixed test pattern and compares the answer with PyTorch's, and it warns you if they differ. Phones only allow the camera on `https://` pages, so put the file on any website, such as GitHub Pages, to use it on a phone. Models over 120 MB, like the vision transformer, are too big for a page. Text and sound models can't become apps yet.

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
- `backend/aibricks/anomaly.py` learns what normal looks like, and `augment.py` varies training photos.
- `backend/aibricks/slices.py` finds and names groups of photos a model gets wrong.
- `backend/aibricks/experiments.py` runs Find a better model, and `checkup.py` makes the training check.
- `backend/aibricks/sounds.py` stores sound clips and trains the sound sorter.
- `backend/aibricks/explain.py` makes the heatmaps and word weights that show where a model looked.
- `backend/aibricks/system.py` reports the training device and workspace size for the home screen.

## Before hosting it publicly

The API binds to localhost. Test runs execute only generated code, and user text enters that
code only as validated string literals. A public deployment still needs container isolation,
no network access for test runs, per-user image storage, memory limits, rate limiting and shared storage for the
record of passed builds, which is currently held in memory.
