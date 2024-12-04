import os
import time

import click
import numpy as np

from pyempad_calibrate.utils import combine_direct, debounce


@click.command()
@click.option("--calib_path", default="./", help="Path to calibration files")
@click.option("--shape", type=(int, int), default=(128, 128), help="Shape of images")
@click.option("--output_path", default="./", help="Path to save results")
@click.argument("background_file", type=click.Path(exists=True))
@click.argument("raw_file", type=click.Path(exists=True))
def main(calib_path, shape, output_path, background_file, raw_file):
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
    flatfA = np.fromfile(
        os.path.join(calib_path, "FFA_prelim.r32"), dtype=np.float32
    ).reshape(shape, order="F")
    flatfB = np.fromfile(
        os.path.join(calib_path, "FFB_prelim.r32"), dtype=np.float32
    ).reshape(shape, order="F")
    print("Reading background data")
    # Read background data in chunks
    tot_frames = os.stat(background_file).st_size // (128 * 128 * 4)
    print(f"Detected {tot_frames=}")
    offset = 0

    start_time = time.time()
    bkgodata = np.zeros((shape[0], shape[1]), dtype=np.float32)
    bkgedata = np.zeros((shape[0], shape[1]), dtype=np.float32)
    niter = 0
    while tot_frames > 0:
        nframes = min(tot_frames, 16384)
        print(f"Reading {nframes=} from {offset=}")
        frames = combine_direct(g1, g2, off, background_file, nframes, offset=offset)
        frames = frames.reshape((shape[0], shape[1], -1), order="F")
        bkgodata += np.mean(frames[:, :, 0::2], axis=2)
        bkgedata += np.mean(frames[:, :, 1::2], axis=2)
        offset += nframes * 128 * 128 * 4
        tot_frames -= nframes
        niter += 1
        del frames
    bkgodata /= niter
    bkgedata /= niter
    print(f"-- Took {time.time() - start_time:.2f} seconds")

    print("Reading raw data")
    tot_frames = os.stat(raw_file).st_size // (128 * 128 * 4)
    print(f"Detected {tot_frames=}")

    output_fname = os.path.join(
        output_path, f"bg_subtracted_{os.path.basename(raw_file)}"[:-4]
    )
    start_time = time.time()
    niter = 0
    offset = 0
    with open(output_fname, "wb") as f:
        while tot_frames > 0:
            nframes = min(tot_frames, 16384)
            print(f"Reading {nframes=} from {offset=}")
            frames = combine_direct(g1, g2, off, raw_file, nframes, offset=offset)
            frames = frames.reshape((shape[0], shape[1], -1), order="F")
            frames[:, :, 0::2] -= bkgodata[:, :, None]
            frames[:, :, 1::2] -= bkgedata[:, :, None]
            print("Debouncing")
            debounce(frames, 10, 3)
            frames[:, :, 0::2] *= flatfA[:, :, None]
            frames[:, :, 1::2] *= flatfB[:, :, None]
            ## frames.tofile(f)   # This changes order to C and is slow...
            f.write(frames.tobytes())
            offset += nframes * 128 * 128 * 4
            tot_frames -= nframes
            niter += 1
            del frames


if __name__ == "__main__":
    main()
