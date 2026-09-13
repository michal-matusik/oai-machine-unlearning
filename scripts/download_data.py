"""Download the officially-provided pretrained LeNet checkpoint.

Fashion-MNIST itself is fetched automatically by `src.dataset.setup_data`
via `torchvision.datasets.FashionMNIST(download=True)` and does not need a
separate download step.
"""

from pathlib import Path

import gdown

CHECKPOINT_FILE_ID = "1YSn8EFjbYDcDCVdlByA_kDKCg4VZLwH8"

if __name__ == "__main__":
    output = Path("data")
    output.mkdir(exist_ok=True)
    gdown.download(id=CHECKPOINT_FILE_ID, output=str(output / "lenet_base_final.pt"), quiet=False)
