import os
import time

import click
import numpy as np

from pyempad_calibrate.utils import combine_chunk, debounce, combine_direct


@click.command()
@click.option("--calib_path", default="./", help="Path to calibration files")
@click.option("--shape", type=(int, int), default=(128, 128), help="Shape of images")
@click.option("--output_path", default="./", help="Path to save results")
@click.option("--direct", is_flag=True, help="Use direct mode for reading data", default=False)
@click.argument("background_file", type=click.Path(exists=True))
@click.argument("raw_file", type=click.Path(exists=True))
def main(calib_path, shape, output_path, direct, background_file, raw_file):
    print("Reading calibration data")
    g1 = np.concatenate(
        (
            np.fromfile(os.path.join(calib_path, "G1A_prelim.r32"), dtype=np.float32),
            np.fromfile(os.path.join(calib_path, "G1B_prelim.r32"), dtype=np.float32),
        )
    )
    g2 = np.concatenate(
        (
            np.fromfile(os.path.join(calib_path, "G2A_prelim.r32"), dtype=np.float32),
            np.fromfile(os.path.join(calib_path, "G2B_prelim.r32"), dtype=np.float32),
        )
    )
    off = np.concatenate(
        (
            np.fromfile(os.path.join(calib_path, "B2A_prelim.r32"), dtype=np.float32),
            np.fromfile(os.path.join(calib_path, "B2B_prelim.r32"), dtype=np.float32),
        )
    )
    print("Reading background data")
    nframes = os.stat(background_file).st_size // (128 * 128 * 4)
    print(f"Detected {nframes=}")
    start_time = time.time()
    if direct:
        frames = combine_direct(g1, g2, off, background_file, nframes)
    else:
        values = np.fromfile(background_file, dtype=np.uint32)
        frames = combine_chunk(values, g1, g2, off)
    print(f"-- Took {time.time() - start_time:.2f} seconds")

    frames = frames.reshape((shape[0], shape[1], -1), order="F")
    print("Background shape:", frames.shape)
    bkgodata = np.mean(frames[:, :, 0::2], axis=2)
    bkgedata = np.mean(frames[:, :, 1::2], axis=2)
    del frames

    print("Reading raw data")
    nframes = os.stat(raw_file).st_size // (128 * 128 * 4)
    start_time = time.time()
    if direct:
        frames = combine_direct(g1, g2, off, raw_file, nframes)
    else:
        values = np.fromfile(raw_file, dtype=np.uint32)
        frames = combine_chunk(values, g1, g2, off)
    print(f"-- Took {time.time() - start_time:.2f} seconds")

    frames = frames.reshape((shape[0], shape[1], -1), order="F")
    frames[:, :, 0::2] -= bkgodata[:, :, None]
    frames[:, :, 1::2] -= bkgedata[:, :, None]
    print("Raw data shape:", frames.shape)
    print("Debouncing")
    debounce(frames, 10, 3)
    print("Multiplying by flat fields")
    flatfA = np.fromfile(
        os.path.join(calib_path, "FFA_prelim.r32"), dtype=np.float32
    ).reshape(shape, order="F")
    flatfB = np.fromfile(
        os.path.join(calib_path, "FFB_prelim.r32"), dtype=np.float32
    ).reshape(shape, order="F")
    frames[:, :, 0::2] *= flatfA[:, :, None]
    frames[:, :, 1::2] *= flatfB[:, :, None]
    print("Saving results")
    output_fname = os.path.join(output_path, f"bg_subtracted_{os.path.basename(raw_file)}"[:-4])
    np.save(output_fname, frames)
