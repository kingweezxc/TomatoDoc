# TomatoDoc UI + prediction backend

This package connects the improved TomatoDoc frontend to your existing Keras model through a local Flask API.

## Expected project structure

Copy `frontend/` and `backend/` into your existing `~/projects/TomatoDoc/` folder so it looks like this:

```text
TomatoDoc/
├── ai/
│   ├── activate.sh
│   └── models/
│       ├── tomatodoc_best.keras
│       └── class_names.json
├── backend/
│   ├── app.py
│   └── requirements.txt
└── frontend/
    └── index.html
```

Do not replace or copy your `.venv` folder. Keep using the AI environment you already configured.
[text](backend)
## Run in WSL (recommended)

Open the VS Code **WSL terminal** and run:

```bash
cd ~/projects/TomatoDoc
source ai/activate.sh
python -m pip install -r backend/requirements.txt
python backend/app.py
```

When Flask starts, open this address in your Windows browser:

http://127.0.0.1:5000

Leave the terminal running while using the web app. The first prediction may take longer while the model loads. The API loads `ai/models/tomatodoc_best.keras` and `ai/models/class_names.json` from your existing project.

## Check the backend

Open http://127.0.0.1:5000/api/health. A healthy response resembles:

```json
{"status":"ready","classes":10,"model":"tomatodoc_best.keras"}
```

If it says the model or labels are missing, check that both files exist under `ai/models/`.

## Important model detail

The prediction API resizes uploaded images to 128 × 128 RGB and sends pixel values to the model without dividing by 255, matching the prediction approach shown in your existing `predict_leaf.py` (which notes that the model's rescaling layer handles normalization). If your saved model does not contain that rescaling layer, the preprocessing must be adjusted to match training.

## Notes

- Supported images: JPG, PNG, WEBP; maximum upload size 12 MB.
- The frontend keeps up to 10 recent scan previews in browser `localStorage`; use **Clear history** to remove them.
- This is a local development server. It is not production-hardened and should not be exposed directly to the public internet.
- Model predictions are estimates, not a definitive agricultural diagnosis.
