"""The AI Bricks API."""
from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from . import catalogue, code_parser, codegen, graph as graphs, labelling, projects, runner, training


@contextlib.asynccontextmanager
async def lifespan(_: FastAPI):
    yield
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
def get_dataset(category: list[str] = Query(default=[], max_length=100)) -> dict:
    return {
        "counts": training.counts(category),
        "minPerCategory": training.MIN_PER_CATEGORY,
        "minTotal": training.MIN_TOTAL,
        "suggestedPerCategory": training.SUGGESTED_PER_CATEGORY,
        "problems": training.readiness_problems(category),
    }


@app.post("/api/dataset/images")
async def add_dataset_image(request: Request, category: str = Query(min_length=1, max_length=200)) -> dict:
    data = await _read_upload(request)
    try:
        return await run_in_threadpool(training.add_image, category, data)
    except training.TrainingError as error:
        raise _raise(error) from None


@app.delete("/api/dataset/images")
def clear_dataset_images(category: str = Query(min_length=1, max_length=200)) -> dict:
    try:
        training.clear_category(category)
    except training.TrainingError as error:
        raise _raise(error) from None
    return {"ok": True}


@app.get("/api/train/current")
def get_training() -> dict:
    return {"job": training.public(training.current())}


@app.post("/api/train")
def start_training(graph: graphs.GraphIn) -> dict:
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
        job = training.start(graph.projectName, codegen.slug(graph.projectName), build_hash, files, names, epochs)
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
        job = training.bundle(job_id, Path(path))
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
async def add_unsorted_image(request: Request) -> dict:
    data = await _read_upload(request)
    try:
        return await run_in_threadpool(labelling.add_image, data)
    except training.TrainingError as error:
        raise _raise(error) from None


@app.get("/api/label/images")
def get_image_queue() -> dict:
    return labelling.queue()


@app.delete("/api/label/images")
def clear_unsorted_images() -> dict:
    return _labelled(labelling.clear_images)


@app.post("/api/label/suggest", dependencies=[Depends(_needs_json)])
def suggest_labels() -> dict:
    return _labelled(labelling.start_suggestions)


@app.get("/api/label/images/{image_id}/file")
def get_unsorted_image(image_id: str) -> FileResponse:
    try:
        path = labelling.image_file(image_id)
    except training.TrainingError as error:
        raise _raise(error) from None
    return FileResponse(path, headers={"Cross-Origin-Resource-Policy": "same-origin", "X-Content-Type-Options": "nosniff"})


@app.post("/api/label/images/{image_id}/assign")
def assign_image(image_id: str, choice: LabelChoice) -> dict:
    return _labelled(labelling.assign_image, image_id, choice.category)


@app.post("/api/label/images/{image_id}/unassign")
def unassign_image(image_id: str, undo: LabelUndo) -> dict:
    return _labelled(labelling.unassign_image, image_id, undo.category)


@app.post("/api/label/images/{image_id}/remove", dependencies=[Depends(_needs_json)])
def remove_image(image_id: str) -> dict:
    return _labelled(labelling.remove_image, image_id)


@app.post("/api/label/text")
async def add_unsorted_text(request: Request, kind: Literal["csv", "lines"] = "lines") -> dict:
    data = await _read_upload(request, labelling.MAX_TEXT_FILE_BYTES)
    try:
        return await run_in_threadpool(labelling.add_text, data, kind)
    except training.TrainingError as error:
        raise _raise(error) from None


@app.get("/api/label/text")
def get_text_queue(category: list[str] = Query(default=[], max_length=100)) -> dict:
    return labelling.text_queue(category)


@app.delete("/api/label/text")
def clear_unsorted_text() -> dict:
    return _labelled(labelling.clear_text)


@app.get("/api/label/text/export")
def export_text_labels(category: list[str] = Query(default=[], max_length=100)) -> Response:
    return Response(
        labelling.export_text(category), media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="texts.csv"'},
    )


@app.post("/api/label/text/{item_id}/assign")
def assign_text(item_id: str, choice: LabelChoice) -> dict:
    return _labelled(labelling.assign_text, item_id, choice.category)


@app.post("/api/label/text/{item_id}/unassign", dependencies=[Depends(_needs_json)])
def unassign_text(item_id: str) -> dict:
    return _labelled(labelling.unassign_text, item_id)


@app.post("/api/label/text/{item_id}/remove", dependencies=[Depends(_needs_json)])
def remove_text(item_id: str) -> dict:
    return _labelled(labelling.remove_text, item_id)
