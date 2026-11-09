import * as api from "./api";

/** Links to the page that runs a photo model by itself, with a line on what it is. */
export function ModelApp({ jobId, beginner }: { jobId: string; beginner: boolean }) {
  return (
    <div className="model-app field-note">
      <div className="label-buttons" style={{ justifyContent: "flex-start", flexWrap: "wrap" }}>
        <a className="button button-small" href={api.modelAppUrl(jobId)} target="_blank" rel="noopener">Open as an app</a>
        <a className="button button-small" href={api.modelAppUrl(jobId, true)} download>Download app</a>
      </div>
      <p className="muted small">
        {beginner
          ? "One web page that runs this model by itself, in any browser, with the camera. Add rules like “beep when it sees a cup”. Photos never leave the device. To use it on a phone, put the file on any website with an https:// address, such as GitHub Pages."
          : "Single HTML file: the model as ONNX, run by ONNX Runtime Web from a pinned CDN build. It self-checks against PyTorch output on load. Rules can beep, show a message, count or POST to a URL."}
      </p>
    </div>
  );
}
