import argparse
import json
import os
import re
import cv2
import h5py
import numpy as np
import numpy.ma as ma
import tifffile
from ScanImageTiffReader import ScanImageTiffReader
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.ndimage import gaussian_filter, median_filter, shift, uniform_filter1d
from scipy.signal import convolve
from sklearn.linear_model import HuberRegressor
from tqdm import tqdm


def extract_pixel_resolution(metadata_str, default_resolution):
    print("Extracting pixel resolution...")
    # Extract resolution of recordfile from raw tiffmetadata
    pixel_resolution = None
    for line in metadata_str.split("\n"):
        if "pixelResolutionXY" in line:
            match = re.search(r'pixelResolutionXY": \[(.*?)\]', line)
            if match:
                pixel_resolution = match.group(1)
                break

    if pixel_resolution:
        pixel_resolution = [int(x) for x in pixel_resolution.split(",")]
        print("pixelResolutionXY found in metadata.")
    else:
        print("pixelResolutionXY not found in metadata. Using default resolution.")
        pixel_resolution = default_resolution

    return pixel_resolution


def run(params, data_dir, output_path, seed=0):
    np.random.seed(seed)

    # Create output directory
    if not os.path.exists(output_path):
        print("Creating output directory...")
        os.makedirs(output_path)
    print("Output directory created at", output_path)

    for ds_factor in (1, 2, 8):
        ds_str = f"DOWNSAMPLED-{ds_factor}x" if ds_factor > 1 else "RAW"
        fns = sorted(
            [
                fn
                for fn in os.listdir(data_dir)
                # if fn.endswith(f"REGISTERED_{ds_str}.tif") and "DENOISED" not in fn # Before the use of Z stacks
                if fn.endswith("_Ch2.ome.tif") and "DENOISED" not in fn
            ]
        )
        if fns:
            break

    # Intialize kernel
    kernel = np.exp(-np.arange(0, 8, params["frametime"] / params["tau"]))
    sw = np.ceil(3 * params["sigma"]).astype(int)
    skernel = np.zeros((2 * sw + 1, 2 * sw + 1, sw + 1))
    skernel[sw, sw, int(sw / 2)] = 1
    skernel = gaussian_filter(
        skernel, [params["sigma"], params["sigma"], params["sigma"] / 2]
    )
    skernel *= (skernel >= skernel[sw].min()) / np.max(skernel)

    if not fns:
        print("No TIFF files found.")
    else:
        print("tiff files found:", fns, "\n")

        print("Loading registered and downsampled data...")
        for fn in fns:
            if "REGISTERED" not in fn:
                print(f"Loading {fn}...")
                metadata_str = ScanImageTiffReader(
                    os.path.join(data_dir, fn)
                ).metadata()
                params["IMsz"] = extract_pixel_resolution(metadata_str, [125, 45])[::-1]

            print(f"Loading {fn}...")

            GT = {}  # Groundtruth dictionary
            print("[DEBUG] os.path.join(data_dir, fn):", os.path.join(data_dir, fn))
            mov = tifffile.imread(
                os.path.join(data_dir, fn)
            )  # TOOD: Read the Z stack. No longer a XYT rather XYZ.
            IMVol_Avg = mov[
                3:-3, :, :
            ]  # IMVol_Avg take mov which is 3d average accross time from ZXYT.
            BG = np.percentile(IMVol_Avg[~np.isnan(IMVol_Avg)], 30)
            IMVol_Avg = np.maximum(np.nan_to_num(IMVol_Avg) - BG, 0)
            IMVol_Avg /= np.percentile(IMVol_Avg, 99)
            selR = np.arange(
                (IMVol_Avg.shape[1] - params["IMsz"][0]) // 2,
                (IMVol_Avg.shape[1] + params["IMsz"][0]) // 2,
            )
            selC = np.arange(
                (IMVol_Avg.shape[2] - params["IMsz"][1]) // 2,
                (IMVol_Avg.shape[2] + params["IMsz"][1]) // 2,
            )
            selZ = np.arange(
                5, IMVol_Avg.shape[0] - 5
            )  # selZ select middle 5 frames. Remove top 5 and bottom 5 frames.

            tmp = median_filter(
                IMVol_Avg, size=(3, 3, 2)
            )  # Change IMavg to IMVol_Avg size with  size=(3, 3, 1) or size=(3, 3, 2) Not sure if filtering in Z would help.
            tmp = tmp > min(np.percentile(tmp, 97), 4 * np.mean(tmp))

            # Set certain regions to False
            mD = max(int(params["minDistance"]), 0)

            # Set slices for the depth (selZ)
            tmp[: selZ[0] + mD, :, :] = False
            tmp[selZ[-1] + 1 - mD :, :, :] = False

            # Set slices for the rows (selR)
            tmp[:, : selR[0] + mD, :] = False
            tmp[:, selR[-1] + 1 - mD :, :] = False

            # Set slices for the columns (selC)
            tmp[:, :, : selC[0] + mD] = False
            tmp[:, :, selC[-1] + 1 - mD :] = False

            tmp = np.transpose(np.where(tmp))

            if params["nsites"] > 0:
                if params["minDistance"] <= 0:
                    releaseSites = tmp[
                        np.random.choice(tmp.shape[0], params["nsites"], replace=False)
                    ]
                    zz, rr, cc = np.unravel_index(releaseSites, mov.shape)
                    zz, dr, dc = np.random.rand(3, len(rr)) - 0.5
                else:
                    releaseSites = np.empty((params["nsites"], 3))
                    for j in range(params["nsites"]):
                        trial = 0
                        while True:
                            i = np.random.choice(len(tmp))
                            candidate = tmp[i] + np.random.rand(3) - 0.5
                            if j == 0 or np.all(
                                np.linalg.norm(releaseSites[:j] - candidate, axis=1)
                                >= params["minDistance"]
                            ):
                                releaseSites[j] = candidate
                                tmp = np.delete(tmp, i, 0)
                                break
                            trial += 1
                            if trial > 10000:
                                raise ValueError(
                                    f"Failed to place site {j+1}/{params['nsites']} at a minimum "
                                    f"distance of {params['minDistance']} within 10000 trials."
                                )
                    zz, rr, cc = zip(*releaseSites)
                    dz, dr, dc = releaseSites.T - np.array([zz, rr, cc])
                # Save Coordinates
                # + 1  Adjust index for Python's 0-based indexing
                GT["R"] = rr + dr - selR[0]
                # + 1  Adjust index for Python's 0-based indexing
                GT["C"] = cc + dc - selC[0]

                GT["Z"] = zz + dz - selZ[0]

            else:
                GT["R"], GT["C"], GT["Z"] = [], [], []

            # Simulate synapses
            for trialIx in tqdm(
                range(1, params["numTrials"] + 1), desc="Simulation Progress"
            ):
                fnstem = (
                    f'SIMULATION_{fn[:11]}{params["SimDescription"]}_Trial{trialIx}'
                )

                B = params["brightness"] * np.exp(
                    -np.arange(params["T"]) * params["frametime"] / params["bleachTau"]
                )

                activity = np.zeros((params["nsites"], params["T"]))
                if params["nsites"] > 0:
                    # Generate random data
                    random_data = np.random.rand(*activity.shape)

                    # Apply threshold
                    thresholded_data = random_data < params["activityThresh"]

                    # Smooth the data with a moving mean
                    smoothed_data = uniform_filter1d(
                        thresholded_data.astype(float), size=40, axis=1
                    )

                    # Generate spikes
                    spikes = np.random.rand(*activity.shape) < smoothed_data**2

                    activity[spikes] = np.minimum(
                        params["maxspike"],
                        np.maximum(
                            params["minspike"],
                            params["spikeAmp"]
                            * np.random.randn(*activity[spikes].shape),
                        ),
                    )

                    activity = convolve(
                        activity, kernel.reshape(1, -1), mode="same", method="direct"
                    )

                # Initialize movie and idealFilts
                movie = np.tile(
                    IMVol_Avg[:, :, :, None], (1, 1, 1, params["T"])
                )  # TODO: Ask MX if using 11 Z stacks is enough for movie?
                idealFilts = np.zeros((*IMVol_Avg.shape, params["nsites"]))
                # Iterate over sites
                for siteN in range(params["nsites"]):
                    # Extract subarray S
                    S = IMVol_Avg[
                        np.maximum((int(zz[siteN]) - int(np.ceil(sw / 2))), 0) : int(
                            zz[siteN]
                        )
                        + int(np.ceil(sw / 2))
                        + 1,
                        np.maximum((int(rr[siteN]) - sw), 0) : int(rr[siteN]) + sw + 1,
                        np.maximum((int(cc[siteN]) - sw), 0) : int(cc[siteN]) + sw + 1,
                    ]  # [Warning] Might use memory when zz is introduced.

                    shiftedKernel = shift(skernel.T, [dz[siteN], dr[siteN], dc[siteN]])
                    zIdxs = np.arange(
                        int(zz[siteN]) - int(np.ceil(sw / 2)),
                        int(zz[siteN]) + int(np.ceil(sw / 2)) + 1,
                    )
                    rIdxs = np.arange(int(rr[siteN]) - sw, int(rr[siteN]) + sw + 1)
                    cIdxs = np.arange(int(cc[siteN]) - sw, int(cc[siteN]) + sw + 1)

                    z_valid = (zIdxs >= 0) & (zIdxs < IMVol_Avg.shape[0])
                    r_valid = (rIdxs >= 0) & (rIdxs < IMVol_Avg.shape[1])
                    c_valid = (cIdxs >= 0) & (cIdxs < IMVol_Avg.shape[2])

                    # Apply valid_mask to shiftedKernel and S
                    filtered_kernel = shiftedKernel[z_valid][:, r_valid][:, :, c_valid]

                    # Apply translation to skernel and multiply by S
                    sFilt = np.multiply(S, filtered_kernel)

                    # Multiply sFilt by activity and reshape
                    temp = np.multiply(
                        sFilt[:, :, :, None], activity[siteN, :].reshape(1, 1, 1, -1)
                    )

                    # Add temp to corresponding subarray in movie
                    movie[
                        np.maximum((int(zz[siteN]) - int(np.ceil(sw / 2))), 0) : int(
                            zz[siteN]
                        )
                        + int(np.ceil(sw / 2))
                        + 1,
                        np.maximum((int(rr[siteN]) - sw), 0) : int(rr[siteN]) + sw + 1,
                        np.maximum((int(cc[siteN]) - sw), 0) : int(cc[siteN]) + sw + 1,
                        :,
                    ] += temp

                    # Store sFilt in idealFilts
                    idealFilts[
                        np.maximum((int(zz[siteN]) - int(np.ceil(sw / 2))), 0) : int(
                            zz[siteN]
                        )
                        + int(np.ceil(sw / 2))
                        + 1,
                        np.maximum((int(rr[siteN]) - sw), 0) : int(rr[siteN]) + sw + 1,
                        np.maximum((int(cc[siteN]) - sw), 0) : int(cc[siteN]) + sw + 1,
                        siteN,
                    ] = sFilt

                # Simulate motion and noise
                envelope = np.square(
                    np.sin(np.cumsum(np.random.randn(params["T"]) / 20))
                )
                motionPC1 = np.convolve(
                    np.multiply(
                        envelope,
                        np.sin(
                            np.convolve(
                                np.random.randn(params["T"]) ** 3,
                                np.ones(40) / 40,
                                mode="same",
                            )
                            / 10
                        ),
                    )
                    * params["motionAmp"],
                    np.ones(5) / 5,
                    mode="same",
                )
                motionPC2 = np.convolve(
                    np.multiply(
                        envelope,
                        np.sin(
                            np.convolve(
                                np.random.randn(params["T"]) ** 3,
                                np.ones(40) / 40,
                                mode="same",
                            )
                            / 10
                        ),
                    )
                    * params["motionAmp"],
                    np.ones(5) / 5,
                    mode="same",
                )
                motionPC3 = np.convolve(
                    np.multiply(
                        envelope,
                        np.sin(
                            np.convolve(
                                np.random.randn(params["T"]) ** 3,
                                np.ones(40) / 40,
                                mode="same",
                            )
                            / 10
                        ),
                    )
                    * params["motionAmp"],
                    np.ones(5) / 5,
                    mode="same",
                )

                psi = np.pi * np.random.rand(1)
                psi = psi[0]
                theta = np.pi * np.random.rand(1)
                theta = theta[0]
                phi = np.pi * np.random.rand(1)
                phi = phi[0]

                R = (
                    np.array(
                        [
                            [np.cos(psi), -np.sin(psi), 0],
                            [np.sin(psi), np.cos(psi), 0],
                            [0, 0, 1],
                        ]
                    )
                    @ np.array(
                        [
                            [np.cos(theta), 0, np.sin(theta)],
                            [0, 1, 0],
                            [-np.sin(theta), 0, np.cos(theta)],
                        ]
                    )
                    @ np.array(
                        [
                            [1, 0, 0],
                            [0, np.cos(phi), -np.sin(phi)],
                            [0, np.sin(phi), np.cos(phi)],
                        ]
                    )
                )

                motion = R @ np.array([motionPC1, motionPC2, motionPC3])

                A1 = 1
                A2 = 0.25
                A3 = 0.15  # TODO: May need to fine tune for Z motion

                GT["motionR"] = A1 * motion[0]
                GT["motionC"] = A2 * motion[1]
                GT["motionZ"] = A3 * motion[2]

                Ad = np.zeros(
                    (len(selR), len(selC), params["numChannels"], params["T"]),
                    dtype=np.float32,
                )
                selR_grid, selC_grid = np.meshgrid(selR, selC, indexing="ij")
                selZ_grid, _, _ = np.meshgrid(selZ, selR, selC, indexing="ij")
                zs, rows, cols = movie.shape[:3]
                excessNoise_file_path = "../code/utils/excess_noise_est.npy"

                if os.path.exists(excessNoise_file_path):
                    excessNoise = np.load(excessNoise_file_path)
                else:
                    print(f"excessNoise File not found: {excessNoise_file_path}")

                excessNoise = np.clip(
                    excessNoise, 0.5, 2
                )  # Hardcoded for now. MX will ask JF about this!

                for frameIx in range(params["T"]):
                    # Create the 2x3 transformation matrix for this frame
                    M = np.float32(
                        [
                            [
                                1,
                                0,
                                GT["motionC"][frameIx],
                            ],  # Translation in columns (x-axis)
                            [0, 1, GT["motionR"][frameIx]],
                        ]  # Translation in rows (y-axis)
                    )

                    # Initialize a temporary array to store the transformed 3D volume for this frame
                    tmp_transformed = np.zeros_like(movie[:, :, :, frameIx])

                    # Iterate over each depth slice in the 3D volume
                    for depthIx in range(movie.shape[0]):
                        # Apply the transformation to each 2D slice
                        tmp_transformed[depthIx, :, :] = cv2.warpAffine(
                            movie[
                                depthIx, :, :, frameIx
                            ],  # 2D slice at depthIx and frameIx
                            M,
                            (cols, rows),  # Output size (width, height)
                            borderMode=cv2.BORDER_CONSTANT,
                            flags=cv2.INTER_CUBIC,
                        )

                    # Extract the motion value for the current frame
                    z = GT['motionZ'][frameIx]

                    # Calculate middle index of the volume
                    middle_frame = tmp_transformed.shape[0] // 2

                    # Determine the base frame index and interpolation weight
                    base_frame = middle_frame + int(np.floor(z))
                    alpha = z - np.floor(z)  # Fractional part for linear interpolation

                    # Extract the neighboring frames
                    frame_below = tmp_transformed[base_frame, selR_grid, selC_grid]
                    frame_above = tmp_transformed[base_frame + 1, selR_grid, selC_grid]

                    # Perform linear interpolation between frames
                    interpolated = (1 - alpha) * frame_below + alpha * frame_above

                    # Apply brightness scaling and add dark current
                    lam = interpolated * B[frameIx] + params["darkrate"]
                    lam = np.maximum(lam, 0)  # Ensure lam is non-negative

                    # Simulate Poisson noise and scale by photonScale and excessNoise
                    Ad[:, :, 0, frameIx] = (
                        np.random.poisson(lam)
                        * params["photonScale"]
                        * excessNoise[: params["IMsz"][0], : params["IMsz"][1]]
                    )

                # The Ad array now contains the simulated data for this trial
                sz = Ad.shape

                GT["activity"] = activity
                GT["ROIs"] = idealFilts[selZ_grid, selR_grid, selC_grid]

                # Save the raw data
                output_directory = os.path.join(output_path, params["SimDescription"])
                # Create the output directory if it doesn't exist
                os.makedirs(output_directory, exist_ok=True)

                Ad_reshaped = Ad.transpose(3, 0, 1, 2).reshape(
                    params["T"], params["IMsz"][0], params["IMsz"][1]
                )

                if params["writetiff"]:
                    fnwrite_tif = os.path.join(output_directory, f"{fnstem}.tif")
                    print(f"Writing {fnwrite_tif} as tiff...")
                    tifffile.imwrite(fnwrite_tif, Ad_reshaped)
                else:
                    fnwrite_AD = os.path.join(output_directory, f"{fnstem}.h5")
                    print(f"Writing {fnwrite_AD} as h5...")
                    # Create a new h5 file
                    with h5py.File(fnwrite_AD, "w") as f:
                        # Create a dataset and write the data
                        f.create_dataset("data", data=Ad_reshaped, compression="gzip")

                fnwrite_AD = os.path.join(output_directory, f"{fnstem}_groundtruth.h5")
                with h5py.File(fnwrite_AD, "w") as f:
                    print(f"Writing {fnwrite_AD} as h5...")
                    f.create_dataset("GT/R", data=GT["R"], compression="gzip")
                    f.create_dataset("GT/C", data=GT["C"], compression="gzip")
                    f.create_dataset(
                        "GT/motionR", data=GT["motionR"], compression="gzip"
                    )
                    f.create_dataset(
                        "GT/motionC", data=GT["motionC"], compression="gzip"
                    )
                    f.create_dataset(
                        "GT/activity", data=GT["activity"], compression="gzip"
                    )
                    f.create_dataset("GT/ROIs", data=GT["ROIs"], compression="gzip")

                with open(output_directory + "/simulation_parameters.json", "w") as f:
                    json.dump(params, f)


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input to the folder that contains tiff files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="../results/",
        help="Output folder to save the results.",
    )

    # Add optional arguments with default values
    parser.add_argument(
        "--SimDescription",
        type=str,
        default="default",
        help="String describing each simulation.",
    )
    parser.add_argument(
        "--darkrate", type=float, default=0.02, help="photon rate added to detector."
    )
    parser.add_argument(
        "--maxshift", type=int, default=30, help="Used for alignment"
    )  # Remove it for simulation
    parser.add_argument("--IMsz", type=int, nargs=2, default=[45, 125])  # Remove this
    # Remove as it for aignment
    parser.add_argument("--ds_time", type=int, default=3)
    # Remove as it for aignment
    parser.add_argument("--clipShift", type=int, default=5)
    # Remove as it for aignment
    parser.add_argument("--alpha", type=float, default=0.0005)
    parser.add_argument(
        "--frametime", type=float, default=0.0023, help="Time between frames in seconds"
    )
    parser.add_argument(
        "--brightness",
        type=float,
        default=0.2,  # JF changed, was 2
        help="Proportional factor that multiplies the sample brightness",
    )
    parser.add_argument(
        "--bleachTau",
        type=int,
        default=10000,
        help="Exponential time constant of bleaching in seconds.",
    )
    parser.add_argument(
        "--T", type=int, default=10000, help="Number of frames to simulate."
    )
    parser.add_argument(
        "--motionAmp",
        type=float,
        default=100,  # JF changed, was 50
        help="Factor that multiplies simulated sample movement",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.027,
        help="Time constant of the decay of the indicator in seconds",
    )
    parser.add_argument(
        "--activityThresh",
        type=float,
        default=0.12,
        help="Lower this threshrold to generate less spikes.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.33,
        help="Size of the spatial filter. How big a synapse is in pixels.",
    )
    parser.add_argument(
        "--photonScale",
        type=int,
        default=600,  # JF changed, was 1000
        help="Amplitude of a single photon in digitizer units.",
    )  # Won't vary in practice
    parser.add_argument(
        "--nsites",
        type=int,
        default=30,  # JF changed, was 50
        help="Number of synapses in a recording.",
    )
    parser.add_argument(
        "--minDistance",
        type=float,
        default=1.5,  # JF added
        help="Minimum distance between synapses.",
    )
    parser.add_argument(
        "--minspike",
        type=float,
        default=0.3,
        help="Minimum fractional change in a spiking event.",
    )
    parser.add_argument(
        "--maxspike",
        type=float,
        default=4,
        help="Maximum fractional change in a spiking event.",
    )
    parser.add_argument(
        "--spikeAmp",
        type=int,
        default=2,
        help="Mean fractional change in a spiking event.",
    )
    # Remove as it for aignment
    parser.add_argument("--numChannels", type=int, default=1)
    parser.add_argument(
        "--numTrials",
        type=int,
        default=5,
        help="Number of trials with same parameters but differne random seed",
    )
    parser.add_argument(
        "--writetiff",
        action="store_true",
        default=True,
        help="Whether to save the movie as tiff instead h5.",
    )

    # Parse the arguments
    args = parser.parse_args()
    # Assign the parsed arguments to params dictionary
    params = vars(args)
    params["dsFac"] = 2**args.ds_time

    run(params, args.input, args.output)
