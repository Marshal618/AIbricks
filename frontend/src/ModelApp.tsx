import * as api from "./api";
import { Why } from "./Ui";

/** Links to the page that runs a photo, pose or fast find-things model by itself, with a line on what it is. */
export function ModelApp({ jobId, beginner, example = "beep when it sees a cup", pose = false }: { jobId: string; beginner: boolean; example?: string; pose?: boolean }) {
  return (
    <div className="model-app field-note">
      <div className="label-buttons" style={{ justifyContent: "flex-start", flexWrap: "wrap" }}>
        <a className="button button-small" href={api.modelAppUrl(jobId)} target="_blank" rel="noopener">Open as an app</a>
        <a className="button button-small" href={api.modelAppUrl(jobId, true)} download>Download app</a>
        <a className="button button-small" href={api.piKitUrl(jobId)} download>Raspberry Pi kit</a>
      </div>
      <p className="muted small">A web page that runs this model on its own, with rules like “{example}”.</p>
      <Why>
        {beginner
          ? "It works in any browser and photos never leave the device. To use it on a phone, put the file on any website with an https:// address, such as GitHub Pages, because phones only allow the camera there."
          : pose
            ? "Single HTML file: the classifier as ONNX (ONNX Runtime Web) and MediaPipe's .task model, run by @mediapipe/tasks-vision, both pinned CDN builds. On load it checks the classifier and its point measurements against PyTorch. Rules can beep, show a message, count or POST to a URL."
            : "Single HTML file: the model as ONNX, run by ONNX Runtime Web from a pinned CDN build. It self-checks against PyTorch output on load. Rules can beep, show a message, count or POST to a URL."}
      </Why>
      <p className="muted small">The Raspberry Pi kit runs it on a Pi's camera and can switch a pin on while it sees something, for a light, buzzer or motor.</p>
      <Why>
        {beginner
          ? "Unzip it on a Raspberry Pi with 64-bit Raspberry Pi OS, run “sh setup.sh”, then “.venv/bin/python run.py”. Its README shows how to wire an LED and start it when the Pi turns on. A USB webcam works too, on any computer."
          : `Zip with model.onnx${pose ? ", MediaPipe's .task file" : ""}, settings.json and run.py: ONNX Runtime on CPU, Picamera2 or OpenCV for frames, gpiozero for pins (--when NAME --pin 17), --command to run something, --json for other programs. No PyTorch needed. setup.sh makes a venv and runs a self-check against PyTorch's outputs.`}
      </Why>
    </div>
  );
}
