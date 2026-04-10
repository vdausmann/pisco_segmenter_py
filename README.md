# pisco_segmenter_py

Standalone, pip-installable segmentation package used by `process_pisco_profiles`.

## Install

```bash
# after publishing on PyPI
pip install pisco_segmenter_py

# local development install
pip install -e .
```

## Usage

```python
from pisco_segmenter import run_segmenter

run_segmenter("/path/to/images", "/path/to/output", deconvolution=True)
```

## Deconvolution model path

By default, deconvolution expects a model under:

`pisco_segmenter/models/lucyd-edof-plankton_231204.pth`

You can override this with:

`PISCO_SEGMENTER_MODEL_PATH=/abs/path/to/model.pth`
