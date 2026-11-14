"""The AI Bricks API."""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from . import anomaly, augment, finder, poses, cards, catalogue, explain as explaining, code_parser, codegen, compare as comparing, datasets, graph as graphs, labelling, opendata, health, projects, runner, slices, sounds, experiments, system, tables, textmodels, training, versions, webapp, pi, outline, forecast, writer, namefinder


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    experiments._stop.set()  # an experiment ends as stopped, not failed
    training.shutdown()


app = FastAPI(title="AI Bricks", lifespan=lifespan)
# Only pages served from this computer can reach the API, which now stores private images.
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

# Builds that have passed a test run, by the hash of their generated code.
_passed: set[str] = set()
_PASSED_LIMIT = 10_000


def _issues_json(analysis: graphs.Analysis) -> list[dict]:
    return [{"message": i.message, "node": i.node} for i in analysis.issues]


@app.get("/api/catalogue")
def get_catalogue() -> dict:
    return catalogue.catalogue_json() | {"codeReference": code_parser.reference(), "codeStarter": code_parser.STARTER}


@app.post("/api/analyse")
def analyse(graph: graphs.GraphIn) -> dict:
    """Checks a build and, when it is complete, returns its code."""
    analysis = graphs.analyse(graph)
    response = {
        "ok": analysis.ok, "issues": _issues_json(analysis), "shapes": analysis.shapes, "files": {}, "dataNeeds": [], "tested": False,
        "buildHash": None, "categories": None, "labelKind": None, "labelCategories": None,
    }
    if analysis.ok:
        files = codegen.generate(graph, analysis)
        response["files"] = files
        response["dataNeeds"] = codegen.data_description(analysis)
        response["buildHash"] = codegen.build_hash(files)
        response["tested"] = response["buildHash"] in _passed
        response["categories"] = codegen.trainable_categories(analysis)
        response["labelKind"], response["labelCategories"] = codegen.labelling_info(analysis)
    return response


class ParseRequest(BaseModel):
    code: str = Field(max_length=200_000)
    graph: graphs.GraphIn


@app.post("/api/parse-model")
def parse_model(request: ParseRequest) -> dict:
    """Reads model.py and returns it as blocks. The code is parsed, never run."""
    current = [{"id": n.id, "type": n.type, "params": n.params} for n in request.graph.nodes]
    result = code_parser.parse_model(request.code, current)
    if result.errors:
        return {"ok": False, "errors": result.errors, "nodes": [], "edges": [], "notes": []}
    notes: list[dict] = []
    candidate = request.graph.model_copy(update={
        "nodes": [graphs.NodeIn(**node) for node in result.nodes],
        "edges": [graphs.EdgeIn(**edge) for edge in result.edges],
    })
    analysis = graphs.analyse(candidate)
    if analysis.ok:
        notes = code_parser.correction_notes(result, codegen.generate(candidate, analysis)["model.py"])
    return {"ok": True, "errors": [], "nodes": result.nodes, "edges": result.edges, "notes": notes}


@app.post("/api/test-run")
async def test_run(graph: graphs.GraphIn) -> StreamingResponse:
    """Streams a test run as newline-delimited JSON."""
    analysis = graphs.analyse(graph)
    if not analysis.ok:
        raise HTTPException(422, {"message": "Fix the build's issues first.", "issues": _issues_json(analysis)})
    files = codegen.generate(graph, analysis)

    async def stream():
        async for event in runner.run_smoke_test(files):
            if event.get("event") == "result" and event.get("passed"):
                if len(_passed) >= _PASSED_LIMIT:
                    _passed.clear()
                _passed.add(codegen.build_hash(files))
            yield json.dumps(event) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/export")
def export(graph: graphs.GraphIn) -> Response:
    """Returns the project as a zip file, but only if this exact code passed a test run."""
    analysis = graphs.analyse(graph)
    if not analysis.ok:
        raise HTTPException(422, {"message": "Fix the build's issues first.", "issues": _issues_json(analysis)})
    files = codegen.generate(graph, analysis)
    if codegen.build_hash(files) not in _passed:
        raise HTTPException(409, {"message": "Run the test on this exact build before downloading it."})
    slug = codegen.slug(graph.projectName)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, contents in files.items():
            archive.writestr(f"{slug}/{name}", contents)
    return Response(
        buffer.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{slug}.zip"'},
    )


# ----------------------------------------------------------------- in-app training

def _needs_json(request: Request) -> None:
    """Actions with no body still need a header that other websites can't send without permission."""
    if request.headers.get("content-type") != "application/json":
        raise HTTPException(415, {"message": "Send this request as application/json."})


def _raise(error: training.TrainingError) -> HTTPException:
    return HTTPException(error.status, {"message": error.message})


def _scope(dataset: str = Query(default=training.DEFAULT_DATASET, max_length=40)) -> str:
    """The dataset a request applies to."""
    try:
        datasets.require(dataset)
    except training.TrainingError as error:
        raise _raise(error) from None
    return dataset


Scope = Annotated[str, Depends(_scope)]


async def _read_upload(request: Request, limit: int = training.MAX_IMAGE_BYTES) -> bytes:
    """Reads one raw image. Requiring this content type stops other websites posting files here."""
    if request.headers.get("content-type") != "application/octet-stream":
        raise HTTPException(415, {"message": "Send the image as application/octet-stream."})
    data = bytearray()
    async for chunk in request.stream():
        data += chunk
        if len(data) > limit:
            raise HTTPException(413, {"message": f"This file is larger than {limit // (1024 * 1024)} MB."})
    return bytes(data)


@app.get("/api/dataset")
def get_dataset(dataset: Scope, category: list[str] = Query(default=[], max_length=100)) -> dict:
    return {
        "counts": training.counts(category, dataset),
        "minPerCategory": training.MIN_PER_CATEGORY,
        "minTotal": training.MIN_TOTAL,
        "suggestedPerCategory": training.SUGGESTED_PER_CATEGORY,
        "problems": training.readiness_problems(category, dataset),
    }


@app.post("/api/dataset/images")
async def add_dataset_image(request: Request, dataset: Scope, category: str = Query(min_length=1, max_length=200)) -> dict:
    data = await _read_upload(request)
    try:
        return await run_in_threadpool(training.add_image, category, data, dataset)
    except training.TrainingError as error:
        raise _raise(error) from None


@app.delete("/api/dataset/images")
def clear_dataset_images(dataset: Scope, category: str = Query(min_length=1, max_length=200)) -> dict:
    try:
        training.clear_category(category, dataset)
    except training.TrainingError as error:
        raise _raise(error) from None
    return {"ok": True}


@app.get("/api/train/current")
def get_training(kind: Literal["image", "text", "table", "sound", "anomaly", "detect", "pose", "outline", "forecast", "writer"] | None = None) -> dict:
    return {"job": training.public(training.latest(kind) if kind else training.current())}


@app.post("/api/train")
def start_training(graph: graphs.GraphIn, dataset: Scope) -> dict:
    analysis = graphs.analyse(graph)
    if not analysis.ok:
        raise HTTPException(422, {"message": "Fix the build's issues first.", "issues": _issues_json(analysis)})
    names = codegen.trainable_categories(analysis)
    if names is None:
        raise HTTPException(422, {"message": "In-app training covers models that sort images into categories. Export this model to train it."})
    files = codegen.generate(graph, analysis)
    build_hash = codegen.build_hash(files)
    if build_hash not in _passed:
        raise HTTPException(409, {"message": "Run the test on this exact build before training it."})
    if labelling.is_suggesting():
        raise HTTPException(409, {"message": "Wait for the model to finish making suggestions."})
    epochs = int(codegen.training_settings(graph)["epochs"])
    try:
        job = training.start(
            graph.projectName, codegen.slug(graph.projectName), build_hash, files, names, epochs, dataset,
            None if dataset == training.DEFAULT_DATASET else versions.freeze,
        )
    except training.TrainingError as error:
        raise _raise(error) from None
    return {"job": training.public(job)}


@app.delete("/api/train/{job_id}")
def stop_training(job_id: str) -> dict:
    try:
        return {"job": training.public(training.stop(job_id))}
    except training.TrainingError as error:
        raise _raise(error) from None


@app.post("/api/train/{job_id}/predict")
async def predict(job_id: str, request: Request) -> dict:
    data = await _read_upload(request)
    try:
        return {"answers": await run_in_threadpool(training.predict, job_id, data)}
    except training.TrainingError as error:
        raise _raise(error) from None


@app.post("/api/train/{job_id}/predict-live")
async def predict_live(job_id: str, request: Request, explain: bool = False) -> dict:
    """For webcam frames: keeps the model loaded between requests so answers come back quickly.

    With explain=true, the reply also has a heatmap of where in the photo the model looked.
    """
    data = await _read_upload(request)
    try:
        reply = await run_in_threadpool(training.predict_live, job_id, data, explain)
    except training.TrainingError as error:
        raise _raise(error) from None
    if "boxes" in reply or "points" in reply or "mask" in reply:
        return reply  # a detector's boxes and counts, or the body and hand points a pose model found
    return reply if explain else {"answers": reply["answers"]}


@app.post("/api/train/{job_id}/explain")
async def explain_image(job_id: str, request: Request) -> dict:
    """The model's answers for a photo, and a heatmap of where it looked."""
    data = await _read_upload(request)
    try:
        return await run_in_threadpool(training.predict_live, job_id, data, True)
    except training.TrainingError as error:
        raise _raise(error) from None


_APP_POLICY = (
    "default-src 'none'; script-src 'unsafe-inline' 'wasm-unsafe-eval' https://cdn.jsdelivr.net; "
    "connect-src data: https: http:; style-src 'unsafe-inline'; img-src blob: data:; media-src blob: mediastream:; "
    "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"
)


@app.get("/api/train/{job_id}/app")
def model_app(job_id: str, download: bool = False) -> Response:
    """A web page that runs this photo model by itself, in any browser, with its own rules."""
    try:
        text, name = webapp.page(job_id)
    except training.TrainingError as error:
        raise _raise(error) from None
    headers = {"Content-Security-Policy": _APP_POLICY, "Cache-Control": "no-store"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="{name}"'
    return Response(text, media_type="text/html; charset=utf-8", headers=headers)


@app.get("/api/train/{job_id}/pi")
def model_pi_kit(job_id: str) -> Response:
    """A zip that runs this model on a Raspberry Pi's camera, with rules that switch its pins."""
    try:
        data, name = pi.kit(job_id)
    except training.TrainingError as error:
        raise _raise(error) from None
    return Response(data, media_type="application/zip", headers={"Content-Disposition": f'attachment; filename="{name}"', "Cache-Control": "no-store"})


@app.get("/api/train/{job_id}/mistakes/{index}")
def mistake_image(job_id: str, index: int) -> FileResponse:
    try:
        path = training.mistake_image(job_id, index)
    except training.TrainingError as error:
        raise _raise(error) from None
    return FileResponse(path, headers={"Cross-Origin-Resource-Policy": "same-origin", "X-Content-Type-Options": "nosniff"})


class MistakeFix(BaseModel):
    action: Literal["move", "remove", "undo"]
    category: str | None = Field(default=None, max_length=200)


@app.post("/api/train/{job_id}/mistakes/{index}/fix")
def fix_mistake(job_id: str, index: int, fix: MistakeFix) -> dict:
    try:
        return {"job": training.public(training.fix_mistake(job_id, index, fix.action, fix.category))}
    except training.TrainingError as error:
        raise _raise(error) from None


@app.get("/api/train/{job_id}/download")
def download_trained(job_id: str) -> FileResponse:
    """Returns the project with its trained model. The user's images are not included."""
    handle, path = tempfile.mkstemp(suffix=".zip")
    os.close(handle)
    try:
        stored = training.stored(job_id)
        job = training.bundle(job_id, Path(path), versions.credits(stored), cards.markdown(stored))
    except training.TrainingError as error:
        os.unlink(path)
        raise _raise(error) from None
    return FileResponse(
        path, media_type="application/zip", filename=f"{job.slug}_trained.zip",
        background=BackgroundTask(os.unlink, path),
    )


# ---------------------------------------------------------------- saved projects

@app.get("/api/projects")
def list_projects() -> dict:
    return {"projects": projects.listing()}


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict:
    try:
        return projects.load(project_id)
    except KeyError:
        raise HTTPException(404, {"message": "That project no longer exists."}) from None


@app.put("/api/projects/{project_id}")
def save_project(project_id: str, project: projects.ProjectIn) -> dict:
    try:
        return projects.save(project_id, project)
    except KeyError:
        raise HTTPException(422, {"message": "That project ID isn't valid."}) from None


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str) -> dict:
    try:
        projects.delete(project_id)
    except KeyError:
        raise HTTPException(404, {"message": "That project no longer exists."}) from None
    return {"ok": True}


# --------------------------------------------------------------------- labelling

class LabelChoice(BaseModel):
    category: str = Field(min_length=1, max_length=200)


class LabelUndo(BaseModel):
    category: str | None = Field(default=None, max_length=200)


def _labelled(action, *arguments):
    """Runs a labelling action and returns {"ok": true}, or the problem as an HTTP error."""
    try:
        action(*arguments)
    except training.TrainingError as error:
        raise _raise(error) from None
    return {"ok": True}


@app.post("/api/label/images")
async def add_unsorted_image(request: Request, dataset: Scope) -> dict:
    data = await _read_upload(request)
    try:
        return await run_in_threadpool(labelling.add_image, data, dataset)
    except training.TrainingError as error:
        raise _raise(error) from None


@app.get("/api/label/images")
def get_image_queue(dataset: Scope) -> dict:
    return labelling.queue(dataset)


@app.delete("/api/label/images")
def clear_unsorted_images(dataset: Scope) -> dict:
    return _labelled(labelling.clear_images, dataset)


@app.post("/api/label/suggest", dependencies=[Depends(_needs_json)])
def suggest_labels(dataset: Scope) -> dict:
    return _labelled(labelling.start_suggestions, dataset)


@app.get("/api/label/images/{image_id}/file")
def get_unsorted_image(image_id: str, dataset: Scope) -> FileResponse:
    try:
        path = labelling.image_file(image_id, dataset)
    except training.TrainingError as error:
        raise _raise(error) from None
    return FileResponse(path, headers={"Cross-Origin-Resource-Policy": "same-origin", "X-Content-Type-Options": "nosniff"})


@app.post("/api/label/images/{image_id}/assign")
def assign_image(image_id: str, choice: LabelChoice, dataset: Scope) -> dict:
    return _labelled(labelling.assign_image, image_id, choice.category, dataset)


@app.post("/api/label/images/{image_id}/unassign")
def unassign_image(image_id: str, undo: LabelUndo, dataset: Scope) -> dict:
    return _labelled(labelling.unassign_image, image_id, undo.category, dataset)


@app.post("/api/label/images/{image_id}/remove", dependencies=[Depends(_needs_json)])
def remove_image(image_id: str, dataset: Scope) -> dict:
    return _labelled(labelling.remove_image, image_id, dataset)


@app.post("/api/label/text")
async def add_unsorted_text(request: Request, dataset: Scope, kind: Literal["csv", "lines"] = "lines") -> dict:
    data = await _read_upload(request, labelling.MAX_TEXT_FILE_BYTES)
    try:
        return await run_in_threadpool(labelling.add_text, data, kind, dataset)
    except training.TrainingError as error:
        raise _raise(error) from None


@app.get("/api/label/text")
def get_text_queue(dataset: Scope, category: list[str] = Query(default=[], max_length=100)) -> dict:
    return labelling.text_queue(category, dataset)


@app.delete("/api/label/text")
def clear_unsorted_text(dataset: Scope) -> dict:
    return _labelled(labelling.clear_text, dataset)


@app.get("/api/label/text/export")
def export_text_labels(dataset: Scope, category: list[str] = Query(default=[], max_length=100)) -> Response:
    return Response(
        labelling.export_text(category, dataset), media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="texts.csv"'},
    )


@app.post("/api/label/text/{item_id}/assign")
def assign_text(item_id: str, choice: LabelChoice, dataset: Scope) -> dict:
    return _labelled(labelling.assign_text, item_id, choice.category, dataset)


@app.post("/api/label/text/{item_id}/unassign", dependencies=[Depends(_needs_json)])
def unassign_text(item_id: str, dataset: Scope) -> dict:
    return _labelled(labelling.unassign_text, item_id, dataset)


@app.post("/api/label/text/{item_id}/remove", dependencies=[Depends(_needs_json)])
def remove_text(item_id: str, dataset: Scope) -> dict:
    return _labelled(labelling.remove_text, item_id, dataset)


# ------------------------------------------------------------------- open datasets

class OpenDataImport(BaseModel):
    dataset: str = Field(max_length=200)
    kind: Literal["image", "text"]
    mapping: dict[str, str] = Field(max_length=2000)
    perCategory: int


@app.get("/api/open-data/search")
def search_open_data(kind: Literal["image", "text"], q: str = Query(default="", max_length=100)) -> dict:
    try:
        return {"results": opendata.search(q, kind)}
    except training.TrainingError as error:
        raise _raise(error) from None


@app.get("/api/open-data/inspect")
def inspect_open_data(kind: Literal["image", "text"], dataset: str = Query(max_length=200)) -> dict:
    try:
        return opendata.inspect(dataset, kind)
    except training.TrainingError as error:
        raise _raise(error) from None


@app.post("/api/open-data/import")
def start_open_data_import(request: OpenDataImport, dataset: Scope) -> dict:
    try:
        return {"job": opendata.start_import(request.dataset, request.kind, request.mapping, request.perCategory, dataset)}
    except training.TrainingError as error:
        raise _raise(error) from None


@app.get("/api/open-data/import")
def get_open_data_import(dataset: Scope) -> dict:
    return opendata.status(dataset)


@app.delete("/api/open-data/import")
def cancel_open_data_import() -> dict:
    opendata.cancel()
    return {"ok": True}


# ---------------------------------------------------------------- saved datasets

class DatasetRename(BaseModel):
    name: str = Field(min_length=1, max_length=60)


class CategoryNames(BaseModel):
    names: list[str] = Field(max_length=datasets.MAX_CATEGORIES)


class CategoryRename(BaseModel):
    old: str = Field(max_length=200)
    new: str = Field(max_length=200)


def _dataset_reply(action, *arguments) -> dict:
    try:
        return action(*arguments)
    except training.TrainingError as error:
        raise _raise(error) from None


@app.get("/api/datasets")
def list_datasets() -> dict:
    return {"datasets": datasets.listing()}


@app.post("/api/datasets")
def create_dataset(request: datasets.DatasetIn) -> dict:
    return _dataset_reply(datasets.create, request)


@app.get("/api/datasets/{dataset_id}")
def get_saved_dataset(dataset_id: str) -> dict:
    return _dataset_reply(datasets.get, dataset_id)


@app.put("/api/datasets/{dataset_id}/name")
def rename_dataset(dataset_id: str, request: DatasetRename) -> dict:
    return _dataset_reply(datasets.rename, dataset_id, request.name)


@app.post("/api/datasets/{dataset_id}/categories")
def add_dataset_categories(dataset_id: str, request: CategoryNames) -> dict:
    return _dataset_reply(datasets.add_categories, dataset_id, request.names)


@app.put("/api/datasets/{dataset_id}/categories")
def rename_dataset_category(dataset_id: str, request: CategoryRename) -> dict:
    return _dataset_reply(datasets.rename_category, dataset_id, request.old, request.new)


@app.delete("/api/datasets/{dataset_id}/categories")
def remove_dataset_category(dataset_id: str, name: str = Query(max_length=200)) -> dict:
    return _dataset_reply(datasets.remove_category, dataset_id, name)


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: str) -> dict:
    return _labelled(datasets.delete, dataset_id)


@app.put("/api/datasets/{dataset_id}/details")
def update_dataset_details(dataset_id: str, request: datasets.DatasetDetails) -> dict:
    return _dataset_reply(datasets.update_details, dataset_id, request)


@app.get("/api/datasets/{dataset_id}/preview")
def preview_dataset(dataset_id: str) -> dict:
    return _dataset_reply(datasets.preview, dataset_id)


@app.get("/api/datasets/{dataset_id}/preview/{image_id}")
def preview_dataset_image(dataset_id: str, image_id: str, category: str = Query(max_length=200)) -> FileResponse:
    try:
        path = datasets.preview_image(dataset_id, category, image_id)
    except training.TrainingError as error:
        raise _raise(error) from None
    return FileResponse(path, headers={"Cross-Origin-Resource-Policy": "same-origin", "X-Content-Type-Options": "nosniff"})


# ---------------------------------------------------------------- dataset versions

class VersionIn(BaseModel):
    message: str = Field(default="", max_length=versions.MAX_MESSAGE)


@app.get("/api/datasets/{dataset_id}/versions")
def list_versions(dataset_id: str) -> dict:
    return _dataset_reply(versions.overview, dataset_id)


@app.post("/api/datasets/{dataset_id}/versions")
def save_version(dataset_id: str, request: VersionIn) -> dict:
    return _dataset_reply(versions.save, dataset_id, request.message)


@app.get("/api/datasets/{dataset_id}/versions/{number}/compare")
def compare_version(dataset_id: str, number: int) -> dict:
    return _dataset_reply(versions.compare, dataset_id, number)


@app.post("/api/datasets/{dataset_id}/versions/{number}/restore", dependencies=[Depends(_needs_json)])
def restore_version(dataset_id: str, number: int) -> dict:
    return _dataset_reply(versions.restore, dataset_id, number)


@app.delete("/api/datasets/{dataset_id}/versions/{number}")
def delete_version(dataset_id: str, number: int) -> dict:
    return _labelled(versions.delete, dataset_id, number)


# ------------------------------------------------------------- models from a dataset

@app.get("/api/models")
def list_all_models(limit: int = Query(default=20, ge=1, le=50)) -> dict:
    """The newest runs from every dataset, each with its dataset's name."""
    names = {row["id"]: row["name"] for row in datasets.listing()}
    rows = training.all_models(limit)
    for row in rows:
        row["datasetName"] = names.get(row["dataset"])
    return {"models": rows}


@app.get("/api/system")
def system_info() -> dict:
    return system.info()


@app.get("/api/datasets/{dataset_id}/models")
def list_dataset_models(dataset_id: str) -> dict:
    try:
        datasets._read(dataset_id)
        return {"models": training.saved_models(dataset_id)}
    except training.TrainingError as error:
        raise _raise(error) from None


class AugmentPreview(BaseModel):
    dataset: str = Field(min_length=1, max_length=40)
    size: int = Field(ge=16, le=512)
    channels: Literal[1, 3]
    params: dict[str, bool | int] = Field(max_length=10)


@app.post("/api/augment/preview")
def augment_preview(request: AugmentPreview) -> dict:
    """One of the dataset's photos, and several versions of it varied as training will vary them."""
    try:
        return augment.preview(request.dataset, request.size, request.channels, request.params)
    except training.TrainingError as error:
        raise _raise(error) from None
    except (TypeError, ValueError):
        raise HTTPException(422, {"message": "Those settings aren't valid."}) from None


@app.post("/api/datasets/{dataset_id}/train-unusual", dependencies=[Depends(_needs_json)])
def train_unusual(dataset_id: str) -> dict:
    """Learns what normal looks like from a “spot what's unusual” dataset's normal photos."""
    try:
        return {"job": training.public(anomaly.start(dataset_id))}
    except training.TrainingError as error:
        raise _raise(error) from None


@app.post("/api/datasets/{dataset_id}/train-pose", dependencies=[Depends(_needs_json)])
def train_pose(dataset_id: str) -> dict:
    """Learns to sort the body poses or hand signs in a “sort poses” dataset's photos."""
    try:
        return {"job": training.public(poses.start(dataset_id))}
    except training.TrainingError as error:
        raise _raise(error) from None


@app.get("/api/models/{job_id}/pose-mistakes")
def pose_mistakes(job_id: str) -> dict:
    return _dataset_reply(poses.mistakes, job_id)


class UnusualLine(BaseModel):
    factor: float


@app.get("/api/models/{job_id}/unusual")
def unusual_settings(job_id: str) -> dict:
    return _dataset_reply(anomaly.settings, job_id)


@app.put("/api/models/{job_id}/unusual")
def move_unusual_line(job_id: str, request: UnusualLine) -> dict:
    return _dataset_reply(anomaly.set_factor, job_id, request.factor)


class BoxesIn(BaseModel):
    boxes: list[dict[str, str | float | list[list[float]]]] = Field(max_length=finder.MAX_BOXES)


class SureIn(BaseModel):
    sure: float


@app.get("/api/datasets/{dataset_id}/boxes")
def list_boxes(dataset_id: str) -> dict:
    """Every photo in a “find things” dataset, with the boxes drawn on those that are done."""
    return _dataset_reply(finder.listing, dataset_id)


@app.put("/api/datasets/{dataset_id}/boxes/{image_id}")
def save_boxes(dataset_id: str, image_id: str, request: BoxesIn) -> dict:
    """Saves one photo's boxes. An empty list means there's nothing to find in it."""
    return _dataset_reply(finder.save_boxes, dataset_id, image_id, request.boxes)


@app.delete("/api/datasets/{dataset_id}/boxes/{image_id}")
def clear_boxes(dataset_id: str, image_id: str) -> dict:
    _dataset_reply(finder.clear_boxes, dataset_id, image_id)
    return {"ok": True}


class WordsIn(BaseModel):
    words: dict[str, str] = Field(max_length=100)


@app.get("/api/datasets/{dataset_id}/find-by-name")
def find_by_name_words(dataset_id: str) -> dict:
    """What the pretrained finder looks for for each thing, and whether it's downloaded."""
    return _dataset_reply(namefinder.words, dataset_id)


@app.put("/api/datasets/{dataset_id}/find-by-name")
def set_find_by_name_words(dataset_id: str, request: WordsIn) -> dict:
    return _dataset_reply(namefinder.set_words, dataset_id, request.words)


@app.post("/api/find-by-name/download", dependencies=[Depends(_needs_json)])
def download_name_finder() -> dict:
    """Starts downloading OWLv2, the pretrained model that finds things from their names, once."""
    return namefinder.start_download()


@app.get("/api/find-by-name")
def name_finder_status() -> dict:
    return namefinder.status()


@app.post("/api/datasets/{dataset_id}/find-by-name/{image_id}", dependencies=[Depends(_needs_json)])
def find_by_name(dataset_id: str, image_id: str) -> dict:
    """Suggested boxes for one photo, from each thing's words, found by OWLv2 on this computer."""
    return _dataset_reply(namefinder.find, dataset_id, image_id)


class ClicksIn(BaseModel):
    clicks: list[dict[str, float | bool]] = Field(min_length=1, max_length=outline.MAX_CLICKS)


@app.post("/api/datasets/{dataset_id}/outline/{image_id}", dependencies=[Depends(_needs_json)])
def click_to_outline(dataset_id: str, image_id: str, request: ClicksIn) -> dict:
    """The outline of the thing around the places clicked, found by MobileSAM on this computer."""
    return _dataset_reply(outline.click, dataset_id, image_id, request.clicks)


@app.post("/api/datasets/{dataset_id}/train-outline", dependencies=[Depends(_needs_json)])
def train_outline(dataset_id: str) -> dict:
    """Fine-tunes a pretrained outlining model on an “outline things” dataset's outlined photos."""
    try:
        return {"job": training.public(outline.start(dataset_id))}
    except training.TrainingError as error:
        raise _raise(error) from None


@app.get("/api/models/{job_id}/outline")
def outline_settings(job_id: str) -> dict:
    return _dataset_reply(outline.details, job_id)


@app.get("/api/models/{job_id}/outline-review/{number}")
def outline_review_image(job_id: str, number: int) -> FileResponse:
    try:
        path = outline.review_image(job_id, number)
    except training.TrainingError as error:
        raise _raise(error) from None
    return FileResponse(path, media_type="image/png", headers={"Cross-Origin-Resource-Policy": "same-origin", "X-Content-Type-Options": "nosniff"})


@app.post("/api/datasets/{dataset_id}/train-find", dependencies=[Depends(_needs_json)])
def train_find(dataset_id: str, detector: Literal["accurate", "fast"] = "accurate") -> dict:
    """Fine-tunes a pretrained detector on a “find things” dataset's boxed photos: the accurate one, or the
    fast one that suits phones, web pages and small computers."""
    try:
        return {"job": training.public(finder.start(dataset_id, detector))}
    except training.TrainingError as error:
        raise _raise(error) from None


@app.get("/api/models/{job_id}/find")
def find_settings(job_id: str) -> dict:
    return _dataset_reply(finder.details, job_id)


@app.put("/api/models/{job_id}/find")
def set_find_sureness(job_id: str, request: SureIn) -> dict:
    return _dataset_reply(finder.set_sure, job_id, request.sure)


@app.post("/api/models/{job_id}/slices", dependencies=[Depends(_needs_json)])
def find_where_it_fails(job_id: str) -> dict:
    """Looks for groups of held-back photos the model gets wrong far more often, and names them."""
    return _dataset_reply(slices.start, job_id)


@app.get("/api/models/{job_id}/slices")
def where_it_fails(job_id: str) -> dict:
    return _dataset_reply(slices.status, job_id)


class ExperimentRequest(BaseModel):
    minutes: int = Field(ge=1, le=60)


@app.post("/api/datasets/{dataset_id}/experiments")
def start_experiment(dataset_id: str, request: ExperimentRequest) -> dict:
    """Find a better model: tries several photo models on this dataset within a time limit."""
    return {"experiment": _dataset_reply(experiments.start, dataset_id, request.minutes)}


@app.get("/api/datasets/{dataset_id}/experiments")
def latest_experiment(dataset_id: str) -> dict:
    _dataset_reply(datasets._read, dataset_id)
    return {"experiment": experiments.latest(dataset_id), "budgets": list(experiments.BUDGETS),
            "changes": {key: {"label": label, "description": text} for key, (label, text) in experiments.CHANGES.items()}}


@app.delete("/api/datasets/{dataset_id}/experiments")
def stop_experiment(dataset_id: str) -> dict:
    return {"experiment": _dataset_reply(experiments.stop, dataset_id)}


@app.get("/api/datasets/{dataset_id}/compare")
def compare_models(
    dataset_id: str, first: str = Query(min_length=1, max_length=40), second: str = Query(min_length=1, max_length=40),
) -> dict:
    """Compares two models on the held-back examples they share."""
    try:
        datasets._read(dataset_id)
    except training.TrainingError as error:
        raise _raise(error) from None
    return _dataset_reply(comparing.compare, dataset_id, first, second)


@app.get("/api/models/{job_id}/card")
def model_card(job_id: str) -> dict:
    return _dataset_reply(cards.card, job_id)


@app.put("/api/models/{job_id}/card")
def update_model_card(job_id: str, notes: cards.CardNotes) -> dict:
    return _dataset_reply(cards.save_notes, job_id, notes)


@app.delete("/api/models/{job_id}")
def delete_saved_model(job_id: str) -> dict:
    return _labelled(training.delete_model, job_id)


# ------------------------------------------------------------------- text models

class TextToTry(BaseModel):
    text: str = Field(min_length=1, max_length=labelling.MAX_TEXT_LENGTH)


@app.post("/api/datasets/{dataset_id}/train-text", dependencies=[Depends(_needs_json)])
def train_text_model(dataset_id: str) -> dict:
    try:
        return {"job": training.public(textmodels.start(dataset_id))}
    except training.TrainingError as error:
        raise _raise(error) from None


@app.post("/api/models/{job_id}/try")
def try_text_model(job_id: str, request: TextToTry) -> dict:
    try:
        return {"answers": textmodels.try_text(job_id, request.text)}
    except training.TrainingError as error:
        raise _raise(error) from None


@app.post("/api/models/{job_id}/explain")
def explain_text_model(job_id: str, request: TextToTry) -> dict:
    """The model's answers, and how much each word pushed it toward the top one."""
    try:
        return explaining.explain_text(job_id, request.text)
    except training.TrainingError as error:
        raise _raise(error) from None


@app.get("/api/models/{job_id}/mistakes")
def text_model_mistakes(job_id: str) -> dict:
    return _dataset_reply(textmodels.mistakes, job_id)


# --------------------------------------------------------------------- sound

@app.post("/api/datasets/{dataset_id}/sounds")
async def add_sound(dataset_id: str, request: Request, category: str = Query(min_length=1, max_length=200)) -> dict:
    """Adds a WAV recording to a category, cut into 1-second clips."""
    data = await _read_upload(request, sounds.MAX_SOUND_BYTES)
    try:
        return await run_in_threadpool(sounds.add_recording, category, data, dataset_id)
    except training.TrainingError as error:
        raise _raise(error) from None


@app.get("/api/datasets/{dataset_id}/sounds")
def list_sounds(dataset_id: str, category: str = Query(min_length=1, max_length=200)) -> dict:
    return _dataset_reply(sounds.clips, dataset_id, category)


@app.get("/api/datasets/{dataset_id}/sounds/{clip_id}")
def sound_file(dataset_id: str, clip_id: str, category: str = Query(min_length=1, max_length=200)) -> FileResponse:
    path = _dataset_reply(sounds.clip_file, dataset_id, category, clip_id)
    return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "private, max-age=86400"})


@app.delete("/api/datasets/{dataset_id}/sounds/{clip_id}")
def remove_sound(dataset_id: str, clip_id: str, category: str = Query(min_length=1, max_length=200)) -> dict:
    _dataset_reply(sounds.remove_clip, dataset_id, category, clip_id)
    return {"ok": True}


@app.post("/api/datasets/{dataset_id}/train-sound", dependencies=[Depends(_needs_json)])
def train_sound_model(dataset_id: str) -> dict:
    try:
        return {"job": training.public(sounds.start(dataset_id))}
    except training.TrainingError as error:
        raise _raise(error) from None


@app.get("/api/models/{job_id}/sound-mistakes")
def sound_mistakes(job_id: str) -> dict:
    return _dataset_reply(sounds.mistakes, job_id)


@app.post("/api/models/{job_id}/listen")
async def listen(job_id: str, request: Request, explain: bool = False) -> dict:
    """A sound model's answers for a WAV recording, its spectrogram, and with explain=true where it listened."""
    data = await _read_upload(request, sounds.MAX_SOUND_BYTES)
    try:
        return await run_in_threadpool(sounds.listen, job_id, data, explain)
    except training.TrainingError as error:
        raise _raise(error) from None


# -------------------------------------------------------------- spreadsheet models

class TableTraining(BaseModel):
    target: str = Field(min_length=1, max_length=500)
    task: Literal["category", "number"]
    columns: list[str] = Field(max_length=tables.MAX_COLUMNS)


class RowToTry(BaseModel):
    values: dict[str, str] = Field(max_length=tables.MAX_COLUMNS)


@app.post("/api/datasets/{dataset_id}/table")
async def upload_table(dataset_id: str, request: Request) -> dict:
    data = await _read_upload(request, tables.MAX_FILE_BYTES)
    try:
        return await run_in_threadpool(tables.upload, dataset_id, data)
    except training.TrainingError as error:
        raise _raise(error) from None


@app.get("/api/datasets/{dataset_id}/table/analysis")
def analyse_table(
    dataset_id: str,
    target: str | None = Query(default=None, max_length=500),
    task: Literal["category", "number"] | None = None,
) -> dict:
    return _dataset_reply(tables.analyse, dataset_id, target, task)


@app.post("/api/datasets/{dataset_id}/train-table")
def train_table_model(dataset_id: str, request: TableTraining) -> dict:
    try:
        return {"job": training.public(tables.start(dataset_id, request.target, request.task, request.columns))}
    except training.TrainingError as error:
        raise _raise(error) from None


@app.get("/api/models/{job_id}/schema")
def table_model_schema(job_id: str) -> dict:
    return _dataset_reply(tables.schema, job_id)


@app.post("/api/models/{job_id}/try-row")
def try_table_model(job_id: str, request: RowToTry) -> dict:
    return _dataset_reply(tables.try_row, job_id, request.values)


# ------------------------------------------------------------------- forecasting

class ForecastTraining(BaseModel):
    when: str = Field(max_length=500)
    value: str = Field(min_length=1, max_length=500)
    horizon: int = Field(ge=1, le=forecast.MAX_HORIZON)
    combine: Literal["mean", "sum"] = "mean"


class ForecastFrom(BaseModel):
    values: list[float] = Field(min_length=1, max_length=forecast.MAX_VALUES_SENT)


@app.get("/api/datasets/{dataset_id}/forecast/analysis")
def analyse_forecast(
    dataset_id: str,
    when: str | None = Query(default=None, max_length=500),
    value: str | None = Query(default=None, max_length=500),
    horizon: int | None = Query(default=None, ge=1, le=forecast.MAX_HORIZON),
    combine: Literal["mean", "sum"] | None = None,
) -> dict:
    return _dataset_reply(forecast.analyse, dataset_id, when, value, horizon, combine)


@app.post("/api/datasets/{dataset_id}/train-forecast")
def train_forecast_model(dataset_id: str, request: ForecastTraining) -> dict:
    try:
        return {"job": training.public(forecast.start(dataset_id, request.when, request.value, request.horizon, request.combine))}
    except training.TrainingError as error:
        raise _raise(error) from None


@app.get("/api/models/{job_id}/forecast")
def forecast_details(job_id: str) -> dict:
    return _dataset_reply(forecast.details, job_id)


@app.post("/api/models/{job_id}/forecast-from")
def forecast_from_values(job_id: str, request: ForecastFrom) -> dict:
    return _dataset_reply(forecast.forecast_from, job_id, request.values)


# ------------------------------------------------------------------- writing

class WriterTraining(BaseModel):
    model: Literal[tuple(catalogue.WRITERS)] = writer.DEFAULT_MODEL  # type: ignore[valid-type]
    strength: Literal["light", "more"] = "light"
    length: Literal["quick", "balanced", "thorough"] = "balanced"


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=writer.MAX_PROMPT * 2)


class WriterAsk(BaseModel):
    prompt: str = Field(min_length=1, max_length=writer.MAX_PROMPT)
    chat: bool = True
    history: list[Turn] = Field(default_factory=list, max_length=40)
    compare: bool = False
    seed: int = Field(default=0, ge=0, le=1_000_000)


@app.get("/api/datasets/{dataset_id}/writer/analysis")
def analyse_writing(dataset_id: str) -> dict:
    return _dataset_reply(writer.analyse, dataset_id)


@app.post("/api/datasets/{dataset_id}/train-writer", dependencies=[Depends(_needs_json)])
def train_writer(dataset_id: str, request: WriterTraining) -> dict:
    try:
        return {"job": training.public(writer.start(dataset_id, request.model, request.strength, request.length))}
    except training.TrainingError as error:
        raise _raise(error) from None


@app.get("/api/models/{job_id}/writer")
def writer_details(job_id: str) -> dict:
    return _dataset_reply(writer.details, job_id)


@app.post("/api/models/{job_id}/write")
async def write_with(job_id: str, request: WriterAsk) -> dict:
    turns = [turn.model_dump() for turn in request.history]
    try:
        return await run_in_threadpool(writer.ask, job_id, request.prompt, request.chat, turns, request.compare, request.seed)
    except training.TrainingError as error:
        raise _raise(error) from None


# ------------------------------------------------------------------- data health

class HealthReview(BaseModel):
    action: Literal["keep", "move", "remove", "drop", "keep-pair"]
    id: str | None = Field(default=None, max_length=80)
    category: str | None = Field(default=None, max_length=200)
    pair: list[str] | None = Field(default=None, max_length=2)


@app.get("/api/datasets/{dataset_id}/health")
def dataset_health(dataset_id: str) -> dict:
    return _dataset_reply(health.overview, dataset_id)


@app.post("/api/datasets/{dataset_id}/health/check", dependencies=[Depends(_needs_json)])
def start_health_check(dataset_id: str) -> dict:
    return {"job": _dataset_reply(health.start, dataset_id)}


@app.post("/api/datasets/{dataset_id}/health/review")
def review_health(dataset_id: str, request: HealthReview) -> dict:
    return _dataset_reply(health.review, dataset_id, request.action, request.id, request.category, request.pair)
