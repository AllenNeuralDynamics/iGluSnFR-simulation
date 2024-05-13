import os
import re
import argparse
from ScanImageTiffReader import ScanImageTiffReader

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
    
    print('Output directory created.', output_path)
    
    # Save the parameters
    params = params
    
    numChannels = 1
    np.random.seed(1) # Seed for reproducability 

    fns = [fn for fn in os.listdir(data_dir) if fn.endswith('.tif') and 'REGISTERED_DOWNSAMPLED-8x' in fn]

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