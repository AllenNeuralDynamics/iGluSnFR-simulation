import os
import re
import h5py
import random
import scipy
import cv2
import argparse
import numpy.ma as ma
import numpy as np
import tifffile
from tqdm import tqdm
from scipy.signal import convolve
from ScanImageTiffReader import ScanImageTiffReader
from scipy.interpolate import griddata
from scipy.ndimage import median_filter, gaussian_filter, shift
from scipy.interpolate import interp1d, PchipInterpolator

def dftregistration_clipped(buf1ft, buf2ft, usfac=1, clip=None):
    if clip is None:
        clip = [0, 0]
    elif isinstance(clip, (int, float)):
        clip = [clip, clip]

    # Compute error for no pixel shift
    if usfac == 0:
        CCmax = np.sum(buf1ft * np.conj(buf2ft))
        rfzero = np.sum(np.abs(buf1ft.flatten()) ** 2)
        rgzero = np.sum(np.abs(buf2ft.flatten()) ** 2)
        error = 1.0 - CCmax * np.conj(CCmax) / (rgzero * rfzero)
        error = np.sqrt(np.abs(error))
        diffphase = np.arctan2(np.imag(CCmax), np.real(CCmax))
        output = [error, diffphase]
        return output, None

    # Whole-pixel shift - Compute crosscorrelation by an IFFT and locate the peak
    elif usfac == 1:
        m, n = buf1ft.shape
        md2 = m // 2
        nd2 = n // 2
        CC = np.fft.ifft2(buf1ft * np.conj(buf2ft))

        keep = np.ones(CC.shape, dtype=bool)
        keep[clip[0] // 2 + 1 : -clip[0] // 2, :] = False
        keep[:, clip[1] // 2 + 1 : -clip[1] // 2] = False
        CC[~keep] = 0

        max1 = np.max(np.real(CC), axis=1)
        loc1 = np.argmax(np.real(CC), axis=1)
        max2 = np.max(max1)
        loc2 = np.argmax(max1)
        rloc = loc1[loc2]
        cloc = loc2
        CCmax = CC[rloc, cloc]
        rfzero = np.sum(np.abs(buf1ft.flatten()) ** 2) / (m * n)
        rgzero = np.sum(np.abs(buf2ft.flatten()) ** 2) / (m * n)
        error = 1.0 - CCmax * np.conj(CCmax) / (rgzero * rfzero)
        error = np.sqrt(np.abs(error))
        diffphase = np.arctan2(np.imag(CCmax), np.real(CCmax))

        md2 = m // 2
        nd2 = n // 2
        if rloc > md2:
            row_shift = rloc - m
        else:
            row_shift = rloc

        if cloc > nd2:
            col_shift = cloc - n
        else:
            col_shift = cloc

        output = [error, diffphase, row_shift, col_shift]
        return output, None

    # Partial-pixel shift
    else:
        # First upsample by a factor of 2 to obtain initial estimate
        # Embed Fourier data in a 2x larger array
        m, n = buf1ft.shape
        mlarge = m * 2
        nlarge = n * 2
        CC = np.zeros((mlarge, nlarge), dtype=np.complex128)
        CC[
            m - (m // 2) : m + (m // 2) + 1,
            n - (n // 2) : n + (n // 2) + 1,
        ] = np.fft.fftshift(buf1ft) * np.conj(np.fft.fftshift(buf2ft))

        # Compute crosscorrelation and locate the peak
        CC = np.fft.ifft2(np.fft.ifftshift(CC))  # Calculate cross-correlation

        keep = np.ones(CC.shape, dtype=bool)
        keep[2 * clip[0] + 1 : -2 * clip[0], :] = False
        keep[:, 2 * clip[1] + 1 : -2 * clip[1]] = False
        CC[~keep] = 0

        max1 = np.max(np.real(CC), axis=1)
        loc1 = np.argmax(np.real(CC), axis=1)
        max2 = np.max(max1)
        loc2 = np.argmax(max1)
        max_val = np.max(np.real(CC))
        rloc, cloc = np.unravel_index(np.argmax(np.real(CC)), CC.shape)
        CCmax = CC[rloc, cloc]

        # Obtain shift in original pixel grid from the position of the
        # crosscorrelation peak
        m, n = CC.shape
        md2 = m // 2
        nd2 = n // 2
        if rloc > md2:
            row_shift = rloc - m
        else:
            row_shift = rloc
        if cloc > nd2:
            col_shift = cloc - n
        else:
            col_shift = cloc
        row_shift = row_shift / 2
        col_shift = col_shift / 2

        # If upsampling > 2, then refine estimate with matrix multiply DFT
        if usfac > 2:
            # Initial shift estimate in upsampled grid
            row_shift = round(row_shift * usfac) / usfac
            col_shift = round(col_shift * usfac) / usfac
            dftshift = np.fix(np.ceil(usfac * 1.5) / 2)  # Center of output array at dftshift+1
            # Matrix multiply DFT around the current shift estimate
            CC = np.conj(
                dftups(
                    buf2ft * np.conj(buf1ft),
                    np.ceil(usfac * 1.5),
                    np.ceil(usfac * 1.5),
                    usfac,
                    dftshift - row_shift * usfac,
                    dftshift - col_shift * usfac,
                )
            ) / (md2 * nd2 * usfac ** 2)
            # Locate maximum and map back to original pixel grid
            max1 = np.max(np.real(CC), axis=1)
            loc1 = np.argmax(np.real(CC), axis=1)
            max2 = np.max(max1)
            loc2 = np.argmax(max1)
            rloc = loc1[loc2]
            cloc = loc2
            CCmax = CC[rloc, cloc]
            rg00 = dftups(buf1ft * np.conj(buf1ft), 1, 1, usfac) / (md2 * nd2 * usfac ** 2)
            rf00 = dftups(buf2ft * np.conj(buf2ft), 1, 1, usfac) / (md2 * nd2 * usfac ** 2)
            rloc = rloc - dftshift
            cloc = cloc - dftshift
            row_shift = row_shift + rloc / usfac
            col_shift = col_shift + cloc / usfac

        # If upsampling = 2, no additional pixel shift refinement
        else:
            rg00 = np.sum(buf1ft * np.conj(buf1ft)) / m / n
            rf00 = np.sum(buf2ft * np.conj(buf2ft)) / m / n
        error = 1.0 - CCmax * np.conj(CCmax) / (rg00 * rf00)
        error = np.sqrt(np.abs(error))
        diffphase = np.arctan2(np.imag(CCmax), np.real(CCmax))
        # If its only one row or column the shift along that dimension has no
        # effect. We set to zero.
        if md2 == 1:
            row_shift = 0
        if nd2 == 1:
            col_shift = 0
        output = [error, diffphase, row_shift, col_shift]

    # Compute registered version of buf2ft
    if usfac > 0:
        nr, nc = buf2ft.shape
        Nr = np.fft.ifftshift(np.arange(-np.fix(nr / 2), np.ceil(nr / 2)))
        Nc = np.fft.ifftshift(np.arange(-np.fix(nc / 2), np.ceil(nc / 2)))
        Nc, Nr = np.meshgrid(Nc, Nr)
        Greg = buf2ft * np.exp(
            1j * 2 * np.pi * (-row_shift * Nr / nr - col_shift * Nc / nc)
        )
        Greg = Greg * np.exp(1j * diffphase)
    elif usfac == 0:
        Greg = buf2ft * np.exp(1j * diffphase)
    else:
        Greg = None

    return output, Greg


def dftups(inp, nor, noc, usfac, roff=0, coff=0):
    nr, nc = inp.shape
    # Compute kernels and obtain DFT by matrix products
    kernc = np.exp(
        (-1j * 2 * np.pi / (nc * usfac))
        * (np.fft.ifftshift(np.arange(nc)).reshape(-1, 1) - np.floor(nc / 2))
        * (np.arange(noc) - coff)
    )
    kernr = np.exp(
        (-1j * 2 * np.pi / (nr * usfac))
        * (np.arange(nor).reshape(-1, 1) - roff)
        * (np.fft.ifftshift(np.arange(nr)) - np.floor(nr / 2))
    )
    out = np.dot(np.dot(kernr, inp), kernc)
    return out

def downsampleTime(Y, ds_time):
    for _ in range(ds_time):
        Y = Y[:, :, :, :2*(Y.shape[3]//2):2] + Y[:, :, :, 1:2*(Y.shape[3]//2):2]
    return Y

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
                
                GT = {} # Groundtruth dictionary
                aData = {} # Alignment data dictionary

                A = ScanImageTiffReader(os.path.join(data_dir, fn))
                A = A.data()
                A = A.T

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
                
                # Save Coordinates
                GT['R'] = rr + dr - selR[0] + 1
                GT['C'] = cc + dc - selC[0] + 1

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
                        movie[rr[siteN]-sw:rr[siteN]+sw+1, cc[siteN]-sw:cc[siteN]+sw+1, :] += temp 

                        # Store sFilt in idealFilts
                        sFilt = np.squeeze(sFilt)  # Remove extra dimension
                        idealFilts[rr[siteN]-sw:rr[siteN]+sw+1, cc[siteN]-sw:cc[siteN]+sw+1, siteN] = sFilt   

                    # Simulate motion and noise
                    envelope = np.square(np.sin(np.cumsum(np.random.randn(params['T']) / 20)))
                    motionPC1 = np.convolve(np.multiply(envelope, np.sin(np.convolve(np.random.randn(params['T'])**3, np.ones(40)/40, mode='same') / 10)) * params['motionAmp'], np.ones(5)/5, mode='same')
                    motionPC2 = np.convolve(np.multiply(envelope, np.sin(np.convolve(np.random.randn(params['T'])**3, np.ones(40)/40, mode='same') / 10)) * params['motionAmp'], np.ones(5)/5, mode='same')
                    
                    # GT['motionR'] = 0.8 * motionPC1 + 0.4 * motionPC2
                    # GT['motionC'] = 0.2 * motionPC1 - 0.2 * motionPC2

                    psi = np.pi * np.random.rand(1)
                    A1 = 1
                    A2 = 0.25
                    GT['motionR'] = A1 * (np.cos(psi) * motionPC1 + np.sin(psi) * motionPC2)
                    GT['motionC'] = A2 * (-np.sin(psi) * motionPC1 + np.cos(psi) * motionPC2)

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

                    selR_grid, selC_grid = np.meshgrid(selR, selC, indexing='ij')
                    T0 = np.pad(IMavg[selR_grid, selC_grid], params['maxshift'], mode='constant')

                    template = T0

                    GT['activity'] = activity

                    initR = 0
                    initC = 0
                    nDSframes = sz[3] // params['dsFac']  # number of downsampled frames
                    motionDSr = np.full(nDSframes, np.nan)
                    motionDSc = np.full(nDSframes, np.nan)
                    aErrorDS = np.full(nDSframes, np.nan)
                    aRankCorr = np.full(nDSframes, np.nan)

                    # Create view matrices for interpolation
                    viewR, viewC = np.meshgrid(
                        np.arange(0, sz[0] + 2 * params['maxshift']) - params['maxshift'], #sz in matlab is 121, 45, 1, 10000
                        np.arange(0, sz[1] + 2 * params['maxshift']) - params['maxshift'],
                        indexing='ij'  # 'ij' for matrix indexing to match MATLAB's ndgrid
                    )

                    for DSframe in range(nDSframes):
                        readFrames = range(DSframe * params['dsFac'], (DSframe + 1) * params['dsFac'])
                        M = downsampleTime(Ad[:, :, :, readFrames], params['ds_time']).sum(axis=2)
                        M = np.squeeze(np.sum(M, axis=2))
                        if DSframe % 1000 == 0:
                            print(f'{DSframe} of {nDSframes}')

                        Ttmp = np.nanmean(np.stack([T0, template]), axis=0)
                        T1 = Ttmp[params['maxshift'] - initR: params['maxshift'] - initR + sz[0], params['maxshift'] - initC: params['maxshift'] - initC + sz[1]]

                        # output = dftregistration(fft2(M), fft2(T1), 4)
                        output,_ = dftregistration_clipped(np.fft.fft2(M), np.fft.fft2(T1.astype(np.float32)), 4, params['clipShift'])
                        
                        motionDSr[DSframe] = initR + output[2]
                        motionDSc[DSframe] = initC + output[3]
                        aErrorDS[DSframe] = output[0]

                        if abs(motionDSr[DSframe]) < params['maxshift'] and abs(motionDSc[DSframe]) < params['maxshift']:
                            # Create grid points
                            X, Y = np.meshgrid(np.arange(0, sz[1]), np.arange(0, sz[0]))

                            # Calculate new grid points
                            Xq = viewC + motionDSc[DSframe]  # Adjust index for Python's 0-based indexing
                            Yq = viewR + motionDSr[DSframe]  # Adjust index for Python's 0-based indexing

                            # Perform interpolation using griddata
                            # A = scipy.interpolate.griddata((X.flatten(), Y.flatten()), M.flatten(), (Xq, Yq), method='linear', fill_value=np.nan)
                            A = cv2.remap(M, Xq.astype(np.float32), Yq.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
                        
                            sel = ~np.isnan(A)

                            selCorr = ~np.isnan(A) & ~np.isnan(template)
                            A_ma = ma.array(A, mask=~selCorr)
                            template_ma = ma.array(template, mask=~selCorr)

                            aRankCorr[DSframe] = ma.corrcoef(ma.array([A_ma.compressed(), template_ma.compressed()]))[0, 1]

                            nantmp = sel & np.isnan(template)
                            template[nantmp] = A[nantmp]
                            template[sel] = (1 - params['alpha']) * template[sel] + params['alpha'] * A[sel]

                            initR = round(motionDSr[DSframe])
                            initC = round(motionDSc[DSframe])

                    tDS = np.multiply(np.arange(1, nDSframes+1), params['dsFac']) - 2**(params['ds_time']-1) + 0.5
                    
                    # Create the new time points
                    new_time_points = np.arange(0, (2**params['ds_time']) * nDSframes)

                    # Pchip Interpolator for motionC and motionR with extrapolation
                    pchip_interpolator_c = PchipInterpolator(tDS, motionDSc, extrapolate=True)
                    pchip_interpolator_r = PchipInterpolator(tDS, motionDSr, extrapolate=True)
                    motionC = pchip_interpolator_c(new_time_points)
                    motionR = pchip_interpolator_r(new_time_points)

                    # Nearest neighbor interpolation for aError with extrapolation
                    nearest_interpolator = interp1d(tDS, aErrorDS, kind='nearest', fill_value='extrapolate')
                    aError = nearest_interpolator(new_time_points)

                    params['maxshiftC'] = int(np.ceil(np.max(np.abs(motionC))))
                    params['maxshiftR'] = int(np.ceil(np.max(np.abs(motionR))))

                    viewR, viewC = np.meshgrid(
                        np.arange(0, sz[0] + 2 * params['maxshiftR']) - params['maxshiftR'],
                        np.arange(0, sz[1] + 2 * params['maxshiftC']) - params['maxshiftC'],
                        indexing='ij'  # This makes meshgrid behave like MATLAB's ndgrid
                    )

                    # Create an open meshgrid using np.ix_
                    selR_ix, selC_ix = np.ix_(selR, selC)

                    # Select the slice of the array as done in MATLAB
                    selected_slice = idealFilts[selR_ix, selC_ix, :]

                    # Define the padding widths for each dimension
                    pad_widths = [(params['maxshiftR'], params['maxshiftR']), (params['maxshiftC'], params['maxshiftC']), (0, 0)]

                    # Pad the array using np.pad
                    tt = np.pad(selected_slice, pad_widths, mode='constant', constant_values=0)

                    GT['ROIs'] = tt

                    IF = np.reshape(tt, (-1, params['nsites']))

                    # Save the raw data

                    # Create the output directory path by joining the output_path with the simulation directories
                    output_directory = os.path.join(output_path, 'SIMULATIONS', params['SimDescription'])
                    
                    # Create the directory if it doesn't exist
                    os.makedirs(output_directory, exist_ok=True)
                
                    Ad_reshaped = Ad.transpose(3, 0, 1, 2).reshape(params['T'], params['IMsz'][0], params['IMsz'][1])
                    
                    if params['writetiff']:
                        fnwrite_tif = os.path.join(output_directory, f'{fnstem}.tif')
                        print(f'Writing {fnwrite_tif} as tiff...')
                        tifffile.imwrite(fnwrite_tif, Ad_reshaped)
                    else:
                        fnwrite_AD = os.path.join(output_directory, f'{fnstem}.h5')
                        print(f'Writing {fnwrite_AD} as h5...')
                        # Create a new h5 file
                        with h5py.File(fnwrite_AD, 'w') as f:
                            # Create a dataset and write the data
                            f.create_dataset('data', data=Ad_reshaped, compression="gzip")

                    GT['ROI_activity'] = np.full((params['nsites'], params['T']), np.nan)
                    
                    for frame in range(len(motionC)):
                        for ch in range(numChannels):
                            # Create grid points
                            x, y = np.meshgrid(np.arange(0, sz[1]), np.arange(0, sz[0]))
                            
                            # Calculate new grid points
                            xi = viewC + motionC[frame]
                            yi = viewR + motionR[frame]
                            
                            # Perform interpolation using griddata
                            # B = griddata((x.flatten(), y.flatten()), Ad[:, :, ch, frame].flatten(), (xi,yi), method='linear', fill_value=np.nan)
                            B = cv2.remap(Ad[:, :, ch, frame], xi.astype(np.float32), yi.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)

                            tmp = np.reshape(B.T, -1)
                            for siteIx in range(params['nsites']):
                                support = IF[:, siteIx] > 0
                                GT['ROI_activity'][siteIx, frame] = np.dot(tmp[support], IF[support, siteIx])

                    aData['numChannels'] = 1
                    aData['frametime'] = params['frametime']
                    aData['motionR'] = motionR
                    aData['motionC'] = motionC
                    aData['aError'] = aError
                    aData['aRankCorr'] = aRankCorr
                    aData['motionDSc'] = motionDSc
                    aData['motionDSr'] = motionDSr

                    fnwrite_AD = os.path.join(output_directory, f'{fnstem}_groundtruth.h5')
                    with h5py.File(fnwrite_AD, "w") as f:
                        print(f'Writing {fnwrite_AD} as h5...')
                        f.create_dataset("GT/R", data=GT['R'], compression="gzip")
                        f.create_dataset("GT/C", data=GT['C'], compression="gzip")
                        f.create_dataset("GT/motionR", data=GT['motionR'], compression="gzip")
                        f.create_dataset("GT/motionC", data=GT['motionC'], compression="gzip")
                        f.create_dataset("GT/activity", data=GT['activity'], compression="gzip")
                        f.create_dataset("GT/ROIs", data=GT['ROIs'], compression="gzip")
                        f.create_dataset("GT/ROI_activity", data=GT['ROI_activity'], compression="gzip")

                        f.create_dataset("aData/numChannels", data=aData['numChannels'])
                        f.create_dataset("aData/frametime", data=aData['frametime'])
                        f.create_dataset("aData/motionR", data=aData['motionR'], compression="gzip")
                        f.create_dataset("aData/motionC", data=aData['motionC'], compression="gzip")
                        f.create_dataset("aData/aError", data=aData['aError'], compression="gzip")
                        f.create_dataset("aData/aRankCorr", data=aData['aRankCorr'], compression="gzip")
                        f.create_dataset("aData/motionDSc", data=aData['motionDSc'], compression="gzip")
                        f.create_dataset("aData/motionDSr", data=aData['motionDSr'], compression="gzip")


if __name__ == "__main__": 
    # Create argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, required=True, help='Input to the folder that contains tiff files.')
    parser.add_argument('--output', type=str, required=True, help='Output folder to save the results.')

    # Add optional arguments with default values
    parser.add_argument('--SimDescription', type=str, default='default', help = 'String describing each simulation.') 
    parser.add_argument('--darkrate', type=float, default=0.02, help = 'photon rate added to detector.') 
    parser.add_argument('--maxshift', type=int, default=30, help = 'Used for alignment') # Remove it for simulation
    parser.add_argument('--IMsz', type=int, nargs=2, default=[125, 45]) # Remove this
    parser.add_argument('--ds_time', type=int, default=3) # Remove as it for aignment 
    parser.add_argument('--clipShift', type=int, default=5)  # Remove as it for aignment 
    parser.add_argument('--alpha', type=float, default=0.0005)  # Remove as it for aignment 
    parser.add_argument('--frametime', type=float, default=0.0023, help = 'Time between frames in seconds') 
    parser.add_argument('--brightness', type=int, default=2, help = 'Proportional factor that multiplies the sample brightness')
    parser.add_argument('--bleachTau', type=int, default=10000, help = 'Exponential time constant of bleaching in seconds.')
    parser.add_argument('--T', type=int, default=10000, help = 'Number of frames to simulate.')
    parser.add_argument('--motionAmp', type=int, default=50, help = 'Factor that multiplies simulated sample movement')
    parser.add_argument('--tau', type=float, default = 0.027 , help = 'Time constant of the decay of the indicator in seconds')
    parser.add_argument('--activityThresh', type=float, default=0.12, help = 'Lower this threshrold to generate more spikes.')
    parser.add_argument('--sigma', type=float, default=1.33, help = 'size of the spatial filter. How big a synapse is in pixels.') # size of the spatial filter. How big a synapse is in pixels.
    parser.add_argument('--photonScale', type=int, default=1000, help = 'Amplitude of a single photon in digitizer units.') # Wont vary in practice
    parser.add_argument('--nsites', type=int, default=50 , help = 'Number of synapses in a recording.') #
    parser.add_argument('--minspike', type=float, default=0.3, help = 'Minimum fractional change in a spiking event.')
    parser.add_argument('--maxspike', type=float, default=4, help = 'Maximum fractional change in a spiking event.')
    parser.add_argument('--spikeAmp', type=int, default=2, help = 'Mean fractional change in a spiking event.')
    parser.add_argument('--numChannels', type=int, default=1)  # Remove as it for aignment 
    parser.add_argument('--writetiff', type=bool, default=False) 

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
    params['writetiff'] = args.writetiff

    run(params, data_dir, output_path)
