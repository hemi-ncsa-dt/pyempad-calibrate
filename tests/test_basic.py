import os
import shutil

import numpy as np
import pytest
from click.testing import CliRunner
from pyempad_calibrate.command import main


@pytest.fixture
def test_files_path(tmp_path):
    src_path = os.path.join(os.path.dirname(__file__), "data")
    data_path = tmp_path / "data"
    data_path.mkdir()
    background_path = data_path / "background.bin"
    shutil.copy(os.path.join(src_path, "bkgd", "n1000.bkgd"), background_path)

    raw_path = data_path / "raw.bin"
    shutil.copy(os.path.join(src_path, "raw", "n1000.raw"), raw_path)

    calib_path = tmp_path / "calib"
    calib_path.mkdir()
    for file in os.listdir(src_path):
        if file.endswith(".r32"):
            shutil.copy(os.path.join(src_path, file), calib_path)

    return tmp_path


@pytest.fixture
def reference_data():
    return np.load(
        os.path.join(os.path.dirname(__file__), "data", "bg_subtracted_n1000.raw")
    )


def test_main(test_files_path, reference_data):
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=test_files_path) as td:
        args = [
            "--calib_path",
            os.path.join(td, "..", "calib"),
            "--shape",
            "128",
            "128",
            "--output_path",
            ".",
            os.path.join(td, "..", "data/background.bin"),
            os.path.join(td, "..", "data/raw.bin"),
        ]

        result = runner.invoke(main, args)
        assert result.exit_code == 0

        assert os.path.exists(os.path.join(td, "bg_subtracted_raw"))
        output_data = np.fromfile(
            os.path.join(td, "bg_subtracted_raw"), dtype=np.float32
        ).reshape((128, 128, 1000), order="C")
        assert np.allclose(output_data, reference_data, atol=1e-5)  # float32 precision
