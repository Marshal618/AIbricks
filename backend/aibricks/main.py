"""The AI Bricks API."""
from __future__ import annotations

import io
import json
import zipfile

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse

from pydantic import BaseModel, Field

from . import catalogue, code_parser, codegen, graph as graphs, runner

app = FastAPI(title="AI Bricks")

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
    response = {"ok": analysis.ok, "issues": _issues_json(analysis), "shapes": analysis.shapes, "files": {}, "dataNeeds": [], "tested": False}
    if analysis.ok:
        files = codegen.generate(graph, analysis)
        response["files"] = files
        response["dataNeeds"] = codegen.data_description(analysis)
        response["tested"] = codegen.build_hash(files) in _passed
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
