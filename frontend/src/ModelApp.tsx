import * as api from "./api";
import { Why } from "./Ui";

/** Links to the page that runs a photo model by itself, with a line on what it is. */
export function ModelApp({ jobId, beginner }: { jobId: string; beginner: boolean }) {
  return (
    <div className="model-app field-note">
      <div className="label-buttons" style={{ justifyContent: "flex-start", flexWrap: "wrap" }}>
        <a className="button button-small" href={api.modelAppUrl(jobId)} target="_blank" rel="noopener">Open as an app</a>
        <a className="button button-small" href={api.modelAppUrl(jobId, true)} download>Download app</a>
      </div>
      <p className="muted small">A web page that runs this model on its own, with rules like “beep when it sees a cup”.</p>
      <Why>
        {beginner
          ? "It works in any browser and photos never leave the device. To use it on a phone, put the file on any website with an https:// address, such as GitHub Pages, because phones only allow the camera there."
          : "Single HTML file: the model as ONNX, run by ONNX Runtime Web from a pinned CDN build. It self-checks against PyTorch output on load. Rules can beep, show a message, count or POST to a URL."}
      </Why>
    </div>
  );
}
