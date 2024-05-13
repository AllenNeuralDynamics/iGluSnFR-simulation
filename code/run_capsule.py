import os
import re
import h5py
import random
import argparse
import numpy as np
import tifffile
from tqdm import tqdm
from scipy.signal import convolve
from ScanImageTiffReader import ScanImageTiffReader
from scipy.ndimage import median_filter, gaussian_filter, shift

def extract_pixel_resolution(metadata_str, default_resolution):
    print('Extracting pixel resolution...')
    # Extract resolution of recordfile from raw tiffmetadata
    pixel_resolution = None
    for line in metadata_str.split('\n'):
        if 'pixelResolutionXY' in line:
            match = re.search(r'pixelResolutionXY": \[(.*?)\]', line)
            if match:
                pixel_resolution = match.group(1)
                break

    if pixel_resolution:
        pixel_resolution = [int(x) for x in pixel_resolution.split(',')]
        print("pixelResolutionXY found in metadata.")
    else:
        print("pixelResolutionXY not found in metadata. Using default resolution.")
        pixel_resolution = default_resolution
    
    return pixel_resolution

def run(params, data_dir, output_path):
    # Create output directory
    if not os.path.exists(output_path):
        print('Creating output directory...')
        os.makedirs(output_path)
    
    print('Output directory created at', output_path)
    
    # Save the parameters
    params = params
    
    numChannels = 1
    np.random.seed(1) # Seed for reproducability 

    print(data_dir)
    fns = [fn for fn in os.listdir(data_dir) if fn.endswith('.tif') and 'REGISTERED_DOWNSAMPLED-8x' in fn]
    
    # Intialize kernel
    kernel = np.exp(-np.arange(0, int(8*params['tau'])+1) / params['tau'])
    sw = np.ceil(3*params['sigma']).astype(int)
    skernel = np.zeros((2*sw+1, 2*sw+1))
    skernel[sw, sw] = 1
    skernel = gaussian_filter(skernel, [params['sigma']-1, params['sigma']-1])
    skernel = skernel / np.max(skernel)
    # Replace corner values with zeros
    rows, cols = skernel.shape
    
    # In matlab gaussian_filter does zero padding by replacing the corner values with zero but in python it doesn't do that so that has to be done manually
    skernel[0, [0, -1]] = 0  # First row corners
    skernel[-1, [0, -1]] = 0  # Last row corners
    skernel[:, 0] = 0  # First column (excluding corners)
    skernel[:, -1] = 0  # Last column (excluding corners)

    if not fns:
        print('No TIFF files found.')
    else:
        print('tiff files found:', fns, '\n')

        print('Loading registered and downsampled data...')
        for fn in fns:
            if 'REGISTERED' not in fn:
                print(f'Loading {fn}...')
                metadata_str = ScanImageTiffReader(os.path.join(data_dir, fn)).metadata() 
                params['IMsz'] = extract_pixel_resolution(metadata_str, [125, 45])

            if 'REGISTERED_DOWNSAMPLED-8x' in fn:
                print(f'Loading {fn}...')
                
                A = ScanImageTiffReader(os.path.join(data_dir, fn))
                A = A.data()
                
                A = A.T
                # print('A shape:', A.shape)
                IMavg = np.nanmean(A, axis=2)
                # IMavg = IMavg.T
                IMavg[np.isnan(IMavg)] = 0
                BG = np.percentile(IMavg[~np.isnan(IMavg)], 30)
                IMavg = np.maximum(IMavg - BG, 0)
                IMavg /= np.percentile(IMavg, 99)
                selR = np.arange((IMavg.shape[0] - params['IMsz'][0]) // 2, (IMavg.shape[0] + params['IMsz'][0]) // 2)
                selC = np.arange((IMavg.shape[1] - params['IMsz'][1]) // 2, (IMavg.shape[1] + params['IMsz'][1]) // 2)
                
                tmp = median_filter(IMavg, size=(3, 3))
                tmp = tmp > min(np.percentile(tmp, 97), 4*np.mean(tmp))
                # Set certain regions to False
                tmp[:selR[0], :] = False
                tmp[selR[-1] + 1:, :] = False
                tmp[:, :selC[0]] = False
                tmp[:, selC[-1] + 1:] = False
                tmp = np.where(tmp)

                releaseSites = random.sample(list(zip(tmp[0], tmp[1])), params['nsites'])
                rr, cc = zip(*releaseSites)
                dr = np.random.rand(len(rr)) - 0.5
                dc = np.random.rand(len(cc)) - 0.5

                # Simulate synapses
                for trialIx in tqdm(range(1, 6), desc='Simulation Progress'):
                    fnstem = f'SIMULATION_{fn[:11]}{params["SimDescription"]}_Trial{trialIx}'
                    
                    B = params['brightness'] * np.exp(-np.arange(1, params['T'] + 1) / params['bleachTau'])
                    activity = np.zeros((params['nsites'], params['T']))
                    spikes_prob = np.random.rand(*activity.shape) < params['activityThresh']
                    spikes = np.random.rand(*activity.shape) < spikes_prob.mean(axis=1)[:, None]**2
                    
                    activity[spikes] = np.minimum(
                        params['maxspike'],
                        np.maximum(
                            params['minspike'],
                            params['spikeAmp'] * np.random.randn(*activity[spikes].shape),
                        ),
                    )
                    
                    activity = convolve(activity, kernel.reshape(1, -1), mode='same', method='direct')

                    # Initialize movie and idealFilts
                    movie = np.tile(IMavg[:, :, None], (1, 1, params['T']))
                    idealFilts = np.zeros((*IMavg.shape, params['nsites']))

                    # Iterate over sites
                    for siteN in range(params['nsites']-1, -1, -1):
                        # Extract subarray S
                        S = IMavg[rr[siteN]-sw:rr[siteN]+sw+1, cc[siteN]-sw:cc[siteN]+sw+1]

                        # Apply translation to skernel and multiply by S
                        sFilt = np.multiply(S , shift(skernel, [dr[siteN], dc[siteN]]))

                        # Multiply sFilt by activity and reshape
                        sFilt = sFilt[:, :, None]  # Add an extra dimension
                        temp = np.multiply(sFilt , activity[siteN, :].reshape(1, 1, -1))

                        # Add temp to corresponding subarray in movie
                        movie[rr[siteN]-sw:rr[siteN]+sw+1, cc[siteN]-sw:cc[siteN]+sw+1, :] += temp # TOdo: Check rnage of movie and temp. See if they differ by a huge factor. Print and see the movie before conversion to uint16 at every iteration. 

                        # Store sFilt in idealFilts
                        sFilt = np.squeeze(sFilt)  # Remove extra dimension
                        idealFilts[rr[siteN]-sw:rr[siteN]+sw+1, cc[siteN]-sw:cc[siteN]+sw+1, siteN] = sFilt   
                                
                    # movie = movie.astype(np.uint16)

                    # Simulate motion and noise
                    envelope = np.square(np.sin(np.cumsum(np.random.randn(params['T']) / 20)))
                    motionPC1 = np.convolve(np.multiply(envelope, np.sin(np.convolve(np.random.randn(params['T'])**3, np.ones(40)/40, mode='same') / 10)) * params['motionAmp'], np.ones(5)/5, mode='same')
                    motionPC2 = np.convolve(np.multiply(envelope, np.sin(np.convolve(np.random.randn(params['T'])**3, np.ones(40)/40, mode='same') / 10)) * params['motionAmp'], np.ones(5)/5, mode='same')
                    GT = {}
                    GT['motionR'] = 0.8 * motionPC1 + 0.4 * motionPC2
                    GT['motionC'] = 0.2 * motionPC1 - 0.2 * motionPC2

                    Ad = np.zeros((len(selR), len(selC), 1, params['T']), dtype=np.float32)
                    for frameIx in range(params['T'] - 1, -1, -1):
                        # tmp = np.roll(movie[:, :, frameIx], (int(GT['motionR'][frameIx]), int(GT['motionC'][frameIx])), axis=(0, 1))
                        tmp = shift(movie[:, :, frameIx], [GT['motionR'][frameIx], GT['motionC'][frameIx]])
                        excessNoise = np.maximum(0.5, np.minimum(2, 1 + np.random.randn(selR.size, selC.size) / 2))
                        selR_grid, selC_grid = np.meshgrid(selR, selC, indexing='ij')
                        lam = np.add(np.multiply(tmp[selR_grid, selC_grid], B[frameIx]), params['darkrate'])
                        lam = np.maximum(lam, 0)  # Ensure lam is non-negative
                        Ad[:, :, 0, frameIx] = np.multiply(np.multiply(np.random.poisson(lam), excessNoise), params['photonScale'])

                    # The Ad array now contains the simulated data for this trial
                    Ad = np.reshape(Ad, (Ad.shape[0], Ad.shape[1], numChannels, -1))
                    Ad = np.array(Ad, dtype=np.float32)
                    sz = Ad.shape

                    GT['activity'] = activity

                    # initR = 0
                    # initC = 0
                    nDSframes = sz[3] // params['dsFac']  # number of downsampled frames

                    # Create the output directory path by joining the output_path with the simulation directories
                    output_directory = os.path.join(output_path, 'SIMULATIONS', params['SimDescription'])
                    
                    # Create the directory if it doesn't exist
                    os.makedirs(output_directory, exist_ok=True)
                    
                    # Define the full path for the file to write
                    fnwrite = os.path.join(output_directory, f'{fnstem}.tif')
                    Ad_reshaped = Ad.transpose(3, 0, 1, 2).reshape(params['T'], params['IMsz'][0], params['IMsz'][1])
                    tifffile.imwrite(fnwrite, Ad_reshaped)

                    # initFrames = 400
                    # framesToRead = initFrames * dsFac

                    # Y = downsampleTime(Ad[:, :, :, :framesToRead], ds_time)

                    # SIMPARAMS_filename = os.path.join(output_directory, f'{fnstem}_SIMPARAMS.h5')

                    # # Create an HDF5 file
                    # with h5py.File(SIMPARAMS_filename, 'w') as f:
                    #     # Create datasets within the HDF5 file
                    #     f.create_dataset('numChannel', data=params['numChannels'])
                    #     f.create_dataset('frametime', data=params['frametime'])
                    #     f.create_dataset('motionC', data=motionC)
                    #     f.create_dataset('motionR', data=motionR)


if __name__ == "__main__": 
    # Create argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, required=True, help='Input to the folder that contains tiff files.')
    parser.add_argument('--output', type=str, required=True, help='Output folder to save the results.')

    # Add optional arguments with default values
    parser.add_argument('--SimDescription', type=str, default='Standard2')
    parser.add_argument('--darkrate', type=float, default=0.02)
    parser.add_argument('--maxshift', type=int, default=30)
    parser.add_argument('--IMsz', type=int, nargs=2, default=[125, 45])
    parser.add_argument('--ds_time', type=int, default=3)
    parser.add_argument('--clipShift', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.0005)
    parser.add_argument('--frametime', type=float, default=0.0023)
    parser.add_argument('--brightness', type=int, default=2)
    parser.add_argument('--bleachTau', type=int, default=10000)
    parser.add_argument('--T', type=int, default=10000)
    parser.add_argument('--motionAmp', type=int, default=50)
    parser.add_argument('--tau', type=float, default=8 * 1.33)
    parser.add_argument('--activityThresh', type=float, default=0.12)
    parser.add_argument('--sigma', type=float, default=1.33) # size of the spatial filter. How big a synapse is in pixels.
    parser.add_argument('--photonScale', type=int, default=1000)
    parser.add_argument('--nsites', type=int, default=50)
    parser.add_argument('--minspike', type=float, default=0.3)
    parser.add_argument('--maxspike', type=float, default=4)
    parser.add_argument('--spikeAmp', type=int, default=2)
    parser.add_argument('--numChannels', type=int, default=1)

    # Parse the arguments
    args = parser.parse_args()

    data_dir = args.input
    output_path = args.output

    # Assign the parsed arguments to params dictionary
    params = {}
    params['SimDescription'] = args.SimDescription
    params['darkrate'] = args.darkrate
    params['maxshift'] = args.maxshift
    params['IMsz'] = args.IMsz
    params['ds_time'] = args.ds_time
    params['dsFac'] = 2 ** args.ds_time
    params['clipShift'] = args.clipShift
    params['alpha'] = args.alpha
    params['frametime'] = args.frametime
    params['brightness'] = args.brightness
    params['bleachTau'] = args.bleachTau
    params['T'] = args.T
    params['motionAmp'] = args.motionAmp
    params['tau'] = args.tau
    params['activityThresh'] = args.activityThresh
    params['sigma'] = args.sigma
    params['photonScale'] = args.photonScale
    params['nsites'] = args.nsites
    params['minspike'] = args.minspike
    params['maxspike'] = args.maxspike
    params['spikeAmp'] = args.spikeAmp
    params['numChannels'] = args.numChannels

    run(params, data_dir, output_path)