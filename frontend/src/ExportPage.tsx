import { useEditor } from "./context";

interface Props {
  projectFolder: string;
  onDownload: () => void;
  onTest: () => void;
  onCode: () => void;
}

export function ExportPage({ projectFolder, onDownload, onTest, onCode }: Props) {
  const { analysis, catalogue, mode } = useEditor();
  const beginner = mode === "beginner";
  const tested = analysis?.tested ?? false;
  const files = Object.keys(analysis?.files ?? {});
  const hasGenerate = files.includes("generate.py");
  return (
    <div className="page-scroll">
      <div className="page-inner page-narrow">
        <header className="page-head">
          <div>
            <h1>Export</h1>
            <p className="muted">{beginner ? "Download your model as code, then train it on your own computer with your own data." : "A standalone PyTorch project. You run the full training on your own hardware."}</p>
          </div>
          <button className="button button-primary" disabled={!tested} onClick={onDownload}>Download .zip</button>
        </header>

        {!tested && (
          <div className="notice">
            <span>Export unlocks after this exact model passes a test run.</span>
            <button className="button button-small" onClick={onTest}>Go to Test</button>
          </div>
        )}

        {files.length > 0 && (
          <>
            <section className="card">
              <h2>{beginner ? "Step 1: Prepare your data" : "Data requirements"}</h2>
              <ul className="plain-list">{analysis!.dataNeeds.map((need) => <li key={need}>{need}</li>)}</ul>
              <p className="muted small">Set the file paths in the section of <code>data.py</code> marked PUT YOUR DATA HERE.</p>
            </section>
            <section className="card">
              <h2>{beginner ? "Step 2: Train your model" : "Train the model"}</h2>
              {beginner && <p className="muted small field-note-top">Unzip the download, open a terminal in that folder's parent, and run these lines. You need Python installed.</p>}
              <pre className="command">{`cd ${projectFolder}
pip install -r requirements.txt
python train.py${hasGenerate ? '\npython generate.py "your text"' : ""}`}</pre>
            </section>
            <section className="card">
              <div className="card-head">
                <h2>Included files</h2>
                <button className="button button-small" onClick={onCode}>View code</button>
              </div>
              <p className="mono small">{files.join("   ")}</p>
            </section>
          </>
        )}
        {files.length === 0 && catalogue && <p className="muted">Files are listed here once the model has no open problems.</p>}
      </div>
    </div>
  );
}
