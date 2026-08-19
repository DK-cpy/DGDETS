"""Dataset download helpers with paths anchored to this package."""

import os
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm


DEFAULT_DATA_ROOT = Path(__file__).resolve().parent / "data"


def _data_root(raw_data_path=None):
    if raw_data_path is None:
        return DEFAULT_DATA_ROOT
    path = Path(raw_data_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def download_file(url, filename):
    """Stream ``url`` to ``filename`` and return the local path."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("Content-Length", 0)) or None
    with filename.open("wb") as handle:
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                progress.update(len(chunk))
                handle.write(chunk)
        progress.close()
    return str(filename)


def download_pets(raw_data_path=None):
    root = _data_root(raw_data_path)
    pets = root / "pets"
    pets.mkdir(parents=True, exist_ok=True)
    files = {
        "test15.pth": "https://www.dropbox.com/s/kzmrwyyk5iaugv0/test15.pth?dl=1",
        "train85.pth": "https://www.dropbox.com/s/w7mikpztkamnw9s/train85.pth?dl=1",
    }
    for name, url in files.items():
        target = pets / name
        if target.is_file():
            print("%s has already been downloaded." % target)
        else:
            print("Downloading %s" % target)
            download_file(url, target)


def download_aircraft(raw_data_path=None):
    root = _data_root(raw_data_path)
    root.mkdir(parents=True, exist_ok=True)
    target = root / "fgvc-aircraft-2013b"
    train_marker = target / "data" / "images_manufacturer_trainval.txt"
    test_marker = target / "data" / "images_manufacturer_test.txt"
    images = target / "data" / "images"
    if train_marker.is_file() and test_marker.is_file() and images.is_dir():
        print("FGVC-Aircraft has already been prepared at %s" % target)
        return

    archive = root / "fgvc-aircraft-2013b.tar.gz"
    if not archive.is_file():
        print("Downloading %s" % archive)
        download_file(
            "http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/"
            "fgvc-aircraft-2013b.tar.gz",
            archive,
        )
    print("Extracting %s to %s" % (archive, root))
    with tarfile.open(str(archive)) as package:
        package.extractall(str(root))
    if archive.is_file():
        os.remove(str(archive))


if __name__ == "__main__":
    download_pets()
    download_aircraft()
