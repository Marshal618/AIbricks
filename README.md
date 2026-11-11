# AI Bricks

**Make your own AI models on your own computer, without writing code.**

AI Bricks is for people who are curious and a bit technical but aren't machine-learning experts. You
collect and label examples, train a model with one button, try it on new photos, sounds or text,
and take it away as a web app or a Python project. If you want to go further, you can design your
own neural network by connecting blocks, and see the PyTorch code it turns into.

Everything runs on your computer. Your photos, sounds, text and spreadsheets are never uploaded,
and there are no paid AI services behind it. The only downloads are open, pretrained starting
models, fetched once the first time a feature needs them.

## What you can make

| You want to… | Example | Choose when creating a dataset |
|---|---|---|
| Sort photos into categories | Ripe or unripe fruit, cat or dog | **Images**, then **Sort photos into categories** |
| Find and count things in photos | Count the screws on a tray | **Images**, then **Find and count things in photos** |
| Spot when something looks wrong | Scratches on a part, a blocked drain | **Images**, then **Spot what's unusual** |
| Recognise body poses or hand signs | Exercise form, thumbs up, a peace sign | **Images**, then **Sort body poses or hand signs** |
| Recognise sounds | A clap, a doorbell, a dog barking | **Sound** |
| Sort text | Support emails into Billing, Bugs and Other | **Text** |
| Predict a number or category from a table | House prices, whether a customer leaves | **Spreadsheet (CSV)** |

Every model gets an honest score on examples it never learned from, shows the mistakes it made,
and can be tried straight away on new examples, including live from your webcam or microphone.

## Get started

You need **Python 3** and **Node.js** (with npm), on macOS or Linux. Then run:

```
./run.sh
```

Open http://127.0.0.1:5173. The first run sets up its own Python environment and installs PyTorch,
MediaPipe and the editor's packages, which takes a few minutes. After that it starts in seconds.

A graphics card isn't needed. Training uses an NVIDIA GPU or Apple silicon when there is one, and
otherwise the processor.

### Your first model in about ten minutes

1. Open **Data labeller** and create a dataset. Give it a name, choose **Images** and **Sort photos
   into categories**, and type a few categories, such as `Cup, Pen, Nothing`.
2. On **Add data**, choose **Use webcam**. Hold a category's button while you move the object
   around: it takes 4 photos a second straight into that category. Aim for 20 or more each, in
   different light and positions. You can also add photo files or folders.
3. On **Check**, see your data's level and fix anything it flags.
4. On **Train**, choose **Find a better model**, or open the **Model builder** to design one
   yourself. When it finishes, you'll see its score on held-back photos and the ones it got wrong.
5. Under **Your newest model**, choose **Try it with your webcam**. Then **Download app** to get a
   web page that runs the model by itself, on any device.

The page for each dataset always shows the next step at the top, as one sentence and one button.

## How it's organised

There are three screens:

- **Home** shows everything at a glance: your newest models from every dataset with their scores,
  your designs and datasets, and what this computer trains on.
- **Data labeller** holds your datasets. Each has numbered tabs in the order you use them:
  1. **Add data**: name your categories and add and sort examples.
  2. **Check**: your data's level, and a check for mislabelled examples.
  3. **Train**: train models, try them, see their mistakes, compare them and download them.

  **About** holds the description, licence and data sources, and **History** keeps versions of your
  data and labels.
- **Model builder** is a canvas where you connect blocks into a neural network, then move through
  the **Design**, **Code**, **Test**, **Train** and **Export** steps.

**Beginner** mode, the default, shows goal-based templates, fewer settings and plain explanations.
**Advanced** mode shows every block and setting, custom training values and an editor for
`model.py`. Switch between them at the top right.

## Adding and labelling data

- **Your own files.** Add photos, a folder of photos, WAV files, or text and CSV files.
- **Webcam.** On a photo dataset, **Use webcam** and hold a category's button, or focus it and
  hold Space, to take 4 photos a second into that category. The camera only runs while the panel is
  open.
- **Microphone.** On a sound dataset, **Use microphone** and hold a category's button while making
  the sound. Every second you hold becomes one clip.
- **Open data.** **Find open data** searches public labelled datasets on Hugging Face, shows each
  one's licence, and copies a sample to this computer. Your search words go to Hugging Face, but your
  own data doesn't. Only public, ungated datasets with a readable category column work. Model
  downloads include `DATA_SOURCES.md`, which credits what was imported. Check each licence yourself.
- **Labelling by keyboard.** Unsorted photos and text can be sorted quickly: number keys choose a
  category, S skips, X removes and Z undoes. Once a model is trained, **Suggest labels** has the
  dataset's newest model guess the rest, starting with the ones it's least sure about, so you only
  confirm or correct with Enter. It works for photo, pose and text datasets. Sound clips are filed as
  they're recorded, so there's nothing left to suggest for.

### Data levels and the label check

The **Check** tab gives photo and text datasets a level:

1. **Started**: at least 2 categories, with enough examples to train.
2. **Solid**: 20 or more examples in each category, no category more than 10 times the size of
   another, and no example filed under two categories.
3. **Checked**: the label check has run on the current data, and no more than 5% of labels are
   still flagged.
4. **Reviewed**: every flagged label, and every near-duplicate pair with different labels, has been
   reviewed.

The label check scores each example with a model that never saw it: a pretrained image or text
model with a simple classifier trained on the rest of the data. Following the confident learning
method, it flags examples the model is confidently sure belong to a different category. You can
move, keep or remove each one, and nothing changes without your say. It also finds near-duplicates.

The percentage shown is the share of labels that passed the check, not proof of how many are right.
In tests where we deliberately mislabelled real data, it caught 65–80% of the mistakes in news
headlines and 74–82% in cat and dog photos. About 55–62% of its flags on headlines were real
mistakes, and 93–100% on the photos. Published research on public datasets found about half of such
flags were real errors. Spreadsheet datasets show table statistics instead, such as repeated rows
and blank cells. The label check isn't available for sound or poses yet, so those datasets reach
level 2 at most.

### Versions

**History** saves versions of your data and labels, so you can compare or restore them. Training
saves a version automatically when the data has changed, then trains on a frozen copy, so every
model can be traced to the exact data it learned from. Restoring saves any unsaved work as a new
version first.

Versions share files with the dataset, so they take little extra disk space. Examples removed from a
dataset stay in older versions until you delete those versions, which erases them from the app. A
trained model can still reflect what it learned from them.

## Kinds of model

### Sort photos

Train from the **Train** tab with **Find a better model**, or design your own in the Model builder.
Training runs the same `train.py` and `predict.py` that the download contains.

**Find a better model** tries several models within the time you choose and keeps the best. Round 1
trains a simple starting point and four versions that each change one thing: varying the training
photos, a wider model, starting from a pretrained ResNet-18, or learning more slowly. They all get the
same short training and are checked on the same held-back photos, so it can show what each change
did, using McNemar's test to separate real differences from luck. Round 2 trains the best two, and a
mix of the changes that helped, for three times as long. This is successive halving, the idea behind
ASHA (Li et al., 2020). Only the winner is kept. **Open its blocks** shows it in the Model builder.
With fewer than 10 held-back photos it can't tell real differences from luck, and it says so.

**Vary training photos**, on the Image folder block, changes each training photo a little every
time the model sees it, so it learns the object rather than one exact picture. Once it's on, you
choose:

- **Mirror left and right.** Leave it off if left and right matter, as with arrows or writing.
- **Crop and zoom**, up to a percentage.
- **Turn**, up to a number of degrees.
- **Light and colour**, up to a percentage.
- **Automatic mix**, which adds TrivialAugment (Müller and Hutter, 2021): one random change of a
  random strength per photo, with no tuning needed.

**Preview with my photos** shows one of your photos and eight varied versions. Only training photos
are varied, fresh each time: the held-back ones never change. AI Bricks doesn't save varied copies
into datasets, because copies of a training photo landing among the held-back ones is a known way to
make a score look far better than it is.

### Find and count things in photos

Name the things to find, add photos, then drag a box around every one on **Add data**. Save with
Enter; keys 1 to 9 pick the thing. Mark a photo with nothing in it as **Nothing to find here**, which
teaches it what not to find. Once a model exists, **Let the model draw first** fills in its guesses
as dashed boxes for you to fix.

It fine-tunes a Faster R-CNN detector (Ren et al., 2015) with a MobileNetV3 backbone, pretrained on
the COCO photos by torchvision. The first run downloads it, about 75 MB. Training needs 10 done
photos and 5 boxes of each thing; aim for 30 boxes of each. The results show how many things it
found and missed on held-back photos, its false alarms and the standard mAP@50 score, with the photos
it got wrong shown with your boxes and its own. **Try it** draws live boxes and counts on the webcam,
and a slider sets how sure it must be before it draws a box. Versions keep the boxes.

### Spot what's unusual

This learns from normal photos only, so you don't need examples of every possible problem. There
are two categories. **Normal** photos are what it learns from: aim for 30, in all the conditions it
will see. **Unusual** is optional: a few photos with problems, used only to check it.

It follows PatchCore (Roth et al., 2022), which reached 99% on the MVTec AD factory-defect benchmark.
A pretrained ResNet-18 describes every small patch of each normal photo, and an evenly spread sample
of those descriptions is kept as a memory. A new photo is unusual when part of it is far from
everything in that memory, and a heatmap shows which parts.

The line between normal and unusual comes from the training photos only: each half is scored against
a memory of the other half, to see how far unseen normal photos usually are. **How strict should it
be?** moves the line and shows the false alarms and catches on the held-back photos at each position.

### Sort body poses and hand signs

After choosing this, pick whether it should look at the whole body or at hands. Add categories, such
as Arms up and Squat, or Thumbs up and Peace, and take photos of each, most easily with the webcam.
Add a category for doing nothing in particular, so it has an answer when you aren't making any of
the others.

MediaPipe (Google, Apache 2.0) finds 33 points on a body, or 21 on each hand. The points are measured
from the shoulders or the wrist, in units of the body's or hand's size, so where someone stands and
how far away they are don't matter. A small network then learns which category the points make.
Because it learns from points, not pixels, clothes, light and background don't get in the way, and a
few dozen photos are often enough. Training takes seconds. The first run downloads MediaPipe's
point-finding model, about 6 MB for bodies or 8 MB for hands, and every trained model keeps its own
copy.

**Try it** draws the points it finds on a photo or the live webcam, with its answers. When no body or
hand is in view, it says so instead of guessing. Photos where nothing could be seen are left out of
learning and reported.

### Sort sounds

A sound dataset starts with a **Background noise** category, because without one the model has to
pick one of your sounds even when nothing is happening. Background categories also get
**Record 20 s** for one long, steady recording of the room. You can add WAV files, and listen to
every clip to remove bad ones. Recordings are saved as 16 kHz mono WAV.

**Start training** needs at least 4 clips per category; 20 or more gives steadier scores. Each clip
becomes a spectrogram, a picture of which pitches are loud at each moment. The pretrained
[Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778), which learned from about two
million AudioSet clips, reads that picture and stays frozen. A small classifier learns your
categories from what it reports, using each clip and two time-shifted, louder or quieter copies. The
first run downloads it, about 350 MB, from Hugging Face.

**Listen with the microphone** shows the model's guess for the last second, about twice a second.
The model hears the room as well as the sound, so record in the places you'll use it, and expect it to
do worse with a different microphone.

### Sort text

**Start training** needs at least 4 labelled examples per category. It follows the
[SetFit](https://arxiv.org/abs/2209.11055) method: it fine-tunes the small
`sentence-transformers/all-MiniLM-L6-v2` model, then trains a classifier on top. The first run
downloads the base model, about 90 MB, from Hugging Face. Text labels download as `texts.csv`.

### Predict from a spreadsheet

Upload a CSV file on **Add data**, then on **Train** choose the column to predict. It predicts a
category or a number using gradient-boosted trees from scikit-learn. Files can be up to 50 MB and
200,000 rows. Uploading a new file replaces the table, and any unsaved data is saved as a version
first.

Columns likely to make a score misleading are left out by default, with the reason shown: row
numbers, IDs, free text, dates and columns that nearly give away the answer. You can turn most of
them back on. Exact repeated rows are removed before the split, so the same row can't appear on both
sides of it. Check any very high score yourself, as these checks can't catch everything. If your rows
are ordered in time, a random hold-back can make a forecast look better than it will be.

### Design your own in the Model builder

Connect blocks on the canvas: image, text and time-series inputs, convolution, LSTM and transformer
layers, and pretrained language and vision models. AI Bricks checks that the blocks fit together and
works out every shape, then **Test** runs the design on sample data, and **Train** trains it on one
of your photo datasets. **Code** shows the PyTorch it turns into, and **Export** downloads it as a
readable project. Designs for other kinds of data train from the downloaded project.

**Convert to blocks**, in Advanced mode, rebuilds the canvas from an edited `model.py`. The code is
read with Python's `ast` module and never run. It must use AI Bricks blocks, so custom Python belongs
in the exported project.

Designs save automatically, and the Projects dialog opens them again. A design can also be saved to a
file and opened on another computer. Files contain the design only, not data or trained models.

## Understanding your model

- **Held-back examples.** Every run holds back about 20% of the data to check the model on. They're
  picked by a fingerprint of each example rather than at random, so models trained on the same
  dataset are checked on mostly the same examples, even after you add data.
- **Mistakes.** After training, you see the held-back examples the model got wrong, most confident
  first: on the Model builder's **Train** step for photo models, and on the dataset's **Train** tab
  for the others. Move any that are filed in the wrong category, or remove them.
- **Where did it look?** On a photo you try, on each mistake, and as **Show where it looks** on the
  webcam, it shades the photo so the parts the model used glow. Use it to catch a model that learned
  the background instead of the object. It uses Grad-CAM (Selvaraju et al., 2017), or pixel gradients
  for models like the vision transformer. For sound, **Show where it listens** highlights the moments
  and pitches that mattered.
- **Which words mattered?** For text, it removes each word in turn and measures how far the top
  answer's chance falls. Words that pushed toward the answer are shaded blue, and against it orange.
  Only the first 80 words are checked. These views show where a model looked, not proof of why it
  answered.
- **Where it fails.** Under the newest photo model, it looks for kinds of photos the model gets wrong
  much more often than others, and names them in everyday words, such as “dark photo” or “close-up”,
  with examples and what to do. It follows the idea of Domino (Eyuboglu et al., ICLR 2022), using
  CLIP to describe the photos. The names are a good guess, so check them against the examples. It
  needs at least 6 held-back mistakes. The first time downloads CLIP, about 600 MB.
- **Training check.** After a photo model trains, it reads the loss chart and names what it sees:
  memorising the training photos, too simple to learn, unstable steps, still improving when it
  stopped, or healthy, with fixes for each.
- **Compare two models.** It checks them only on the examples both held back, counts the examples one
  got right and the other missed, and uses McNemar's exact test (Dietterich, 1998) to say whether the
  gap is likely to be luck. Scores from small datasets are rough, so this tells you whether a better
  score is real.
- **Model card.** **Card** shows a model card for any finished model, following Mitchell et al.
  (2019): what it is, its training data, how it was checked, results with a likely range for the
  accuracy, and limitations worked out from the results. You write the intended use. Downloads
  include it as `MODEL_CARD.md`.

## Using your model elsewhere

### As a web app

**Open as an app** and **Download app** appear under the newest photo or pose model. They make a
single web page that runs the model by itself in any browser, with the camera or a chosen photo.
Nothing is uploaded.

The page can follow simple rules: when it's at least a chosen percentage sure of a category for half
a second, it can play a beep, show a message, count, or send the result to a web address you type,
such as an [ntfy](https://ntfy.sh) topic for phone notifications. Rules are saved in that browser.

The model is exported to [ONNX](https://onnx.ai), an open format for trained models, and run by ONNX
Runtime Web. A pose model's page also carries MediaPipe's point-finding model and runs it with
MediaPipe's web library. Both libraries load from pinned CDN versions checked against fixed hashes,
so the page needs the internet the first time it opens. When it opens, it checks its answers against
the ones PyTorch gave when the page was made, and warns you if they differ.

Phones only allow the camera on `https://` pages, so to use an app on a phone, put the file on any
website, such as GitHub Pages. Models over 120 MB, like the vision transformer, are too big for a
page. Text, sound, unusual-photo and find-things models can't become apps yet.

### As a Python project

**Download** on any model in the **Train** tab gives a zip with the trained model and readable
`train.py`, `predict.py` and `README.md` files, so it runs and retrains without AI Bricks. Your data
isn't included. Photo models built from blocks also include `review.py`, which lists the held-back
mistakes, and `serve.py`, a small FastAPI web API for the model.

## Privacy and storage

- Everything you add is stored in `backend/.workspace/`, which git ignores. Set
  `AI_BRICKS_WORKSPACE` to store it somewhere else.
- Nothing you add is sent anywhere. The only outgoing connections are searches in **Find open data**,
  downloads of pretrained models the first time a feature needs them, and rules you set up in a web
  app to send results to an address you typed.
- Pretrained models are cached in `backend/.workspace/cache/`. Set `AI_BRICKS_CACHE` to move them.
- The 10 newest trained models are kept for each workspace. Older model files are removed to save
  space, but their scores stay listed.

## For developers

### Run the tests

```
cd backend
PYTHONPATH=. .venv/bin/pytest tests -q
```

The full suite trains real models, so it takes several minutes, and the first run downloads
pretrained models into `backend/.test-cache/`.

### How it fits together

- `frontend/` is the editor and labeller, built with React, TypeScript, Vite and React Flow.
- `backend/` is a FastAPI server. `aibricks/main.py` holds the API routes.
- Model builder: `catalogue.py` defines every block, connector type and starter kit; `graph.py`
  validates a design and works out every shape; `codegen.py` and `brick_sources.py` write the PyTorch
  project; `code_parser.py` turns `model.py` back into blocks without running it; `runner.py` runs the
  test in a separate, time-limited process; `projects.py` stores saved designs.
- Data: `datasets.py` stores datasets and categories; `labelling.py` stores unsorted photos and text
  and makes label suggestions; `versions.py` keeps versions; `opendata.py` imports open datasets;
  `health.py` and `health_check.py` give levels and run the label check.
- Training: `training.py` runs training and predictions in separate processes; `experiments.py` runs
  Find a better model; `augment.py` varies training photos; `checkup.py` makes the training check.
- Kinds of model: `finder.py` finds things in photos; `anomaly.py` spots what's unusual; `poses.py`
  sorts poses and hand signs; `sounds.py` sorts sounds; `textmodels.py` sorts text; `tables.py`
  predicts from spreadsheets.
- Understanding models: `explain.py` makes heatmaps and word weights; `slices.py` finds where a model
  fails; `compare.py` compares models; `cards.py` writes model cards.
- `webapp.py` turns models into web apps, and `system.py` reports the training device and storage.

Each kind of model writes its own readable `train.py` and `predict.py` and runs them in a separate
process, so a model trained in the app behaves exactly like one trained from its download.

### Before hosting it publicly

AI Bricks is made to run on your own computer, and the API only listens on localhost. Test runs
execute only generated code, and user text enters that code only as validated string literals. A
public deployment would still need sign-in and per-user storage, container isolation, no network
access for test runs, memory limits, rate limiting and shared storage for the record of passed
designs, which is currently held in memory.
