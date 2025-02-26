# iGluSnFR-simulation

This capsule generates synthetic dendrites simulating neural activity with realistic noise and motion artifacts closely representing 2-Photon imaging data collected from mice with fluorescent glutamate indicator iGluSnFR. 

Key features include:
- Neural activity modeling with configurable spike patterns and synaptic dynamics
- 3D motion simulation incorporating translational and rotational drift in XYZ planes
- Noise injection with Poisson-distributed photon noise and detector dark current
- Multi-format output supporting both TIFF stacks and HDF5 files for flexibility
 
# Data requirements:
This capsule requires Z-stacks to infer and apply z motion to the simulations. 

CO Data Asset: `b5118287-af2c-433e-bc57-d4156844cd5f`
```
root
└── data
    └── Bergamo-zStacks
        ├── scan_00001-REF_Ch2.ome
        │   └── scan_00001-REF_Ch2.ome.tif
        └── scan_00002-REF_Ch2.ome
            └── scan_00002-REF_Ch2.ome.tif
        .
        .
        .
        └── scan_*-REF_Ch2.ome
            └── scan_*-REF_Ch2.ome.tif
```

# Result:
```
results/  
└── parameters.csv 
└── SimulationDescription/  
    ├── SIMULATION_*.tif/.h5          # Raw time-lapse data   
    ├── *_groundtruth.h5         # Ground truth coordinates/activity  
    └── simulation_parameters.json # Configuration metadata
```
- parameters.csv contains SimDescription,motionAmp,brightness,nsites,scan which are the parameters used to generate simulation data. 
- *_groundtruth.h5 Key HDF5 datasets include `/GT/R`, `/GT/C`, `/GT/Z` for 3D coordinates, `/GT/activity` for neural spike trains, and `/GT/motionC`, `/GT/motionR`, `/GT/motionZ` for motion across XYZ.  

# Code execution:
- Go to [run](code/run) and adjust `motionAmp_values`, `brightness_values`, `nsites_values` to your liking. 
- From run change path under `subdirs` if different dataset is used, if you use a different dataset you may need to change `fn.endswith("_Ch2.ome.tif")` from [run_capsule.py](code/run_capsule.py)
> **⚠️ WARNING:** You may want to adjust `max_parallel` based on your system requirements.

# Simulation tifs:
| ![Image 1](https://github.com/user-attachments/assets/7d16465b-bb69-42fd-ad53-db81fcd46185) | ![Image 2](https://github.com/user-attachments/assets/6585e312-3e9b-4cf0-b484-5949a242b24b) |
|:--------------------------------------------------------:|:-------------------------------------------------------:|
| Before adding motion correction and noise                | After adding motion correction and noise                |





# Simulation generation flowchart:
```
START
│
├─▶ [Initialization]
│   ├─▶ Parse Command Line Arguments:
│   │    - Input/output paths, simulation parameters
│   │
│   └─▶ Initialize Parameters (params):
│        - Set defaults: darkrate=0.02, nsites=30, T=10000, etc.
│
├─▶ Create Output Directory
│   └─▶ IF path doesn't exist → os.makedirs()
│
├─▶ File Handling
│   ├─▶ Search for TIFF Files:
│   │    - Check 1x/2x/8x downsampling versions
│   │    - Filter: Ends with "_Ch2.ome.tif", excludes "DENOISED"
│   │
│   └─▶ IF no files found → ERROR EXIT
│
└─▶ For Each TIFF File:
     │
     ├─▶ Metadata Processing
     │   ├─▶ Load Metadata:
     │   │    - ScanImageTiffReader().metadata()
     │   │
     │   └─▶ Extract Pixel Resolution:
     │        ├─▶ Regex search: r'pixelResolutionXY": \[(.*?)\]'
     │        ├─▶ IF found → Convert to [int,int] list
     │        └─▶ ELSE → Use default [125,45] → params["IMsz"] = [45,125]
     │
     ├─▶ Load Image Volume
     │   ├─▶ Read TIFF: tifffile.imread() → mov (ZXYT)
     │   │
     │   └─▶ Preprocessing:
     │        ├─▶ Temporal Crop: mov[5:-5] (remove first/last 5 frames)
     │        ├─▶ BG Subtraction: mov - 30th percentile
     │        ├─▶ Normalization: ÷ 99th percentile
     │        └─▶ Median Filter: 3×3×2 kernel
     │
     ├─▶ Create Valid Region Mask
     │   ├─▶ Thresholding:
     │   │    - Binary mask = (filtered_data > min(97th %ile, 4×mean))
     │   │
     │   └─▶ Edge Exclusion:
     │        - Clear borders using params["minDistance"] (mD)
     │        - tmp = np.where(mask) → [Z,Y,X] candidate positions
     │
     ├─▶ Synapse Placement (GT Generation)
     │   ├─▶ IF params["nsites"] > 0:
     │   │    │
     │   │    ├─▶ Random Placement Mode (minDistance ≤ 0):
     │   │    │     - np.random.choice() without spatial checks
     │   │    │     - Add ±0.5px jitter
     │   │    │
     │   │    └─▶ Distance-Constrained Mode:
     │   │         │
     │   │         ├─▶ FOR each synapse:
     │   │         │     │
     │   │         │     ├─▶ WHILE trials < 10,000:
     │   │         │     │     - Pick random candidate
     │   │         │     │     - Check Euclidean distance ≥ minDistance
     │   │         │     │     - IF valid → Place and remove from candidates
     │   │         │     │
     │   │         │     └─▶ ELSE → Raise ValueError
     │   │         │
     │   │         └─▶ Store with sub-pixel offsets (dz,dr,dc)
     │   │
     │   └─▶ ELSE → GT["R/C/Z"] = empty lists
     │
     └─▶ For Each Trial (1 to numTrials):
          │
          ├─▶ Activity Generation
          │   ├─▶ 1. Random Thresholding:
          │   │     - spikes_base = (rand(nsites,T) < activityThresh)
          │   │
          │   ├─▶ 2. Smoothing:
          │   │     - smoothed = uniform_filter1d(spikes_base, size=40) #Moving average across a window
          │   │
          │   ├─▶ 3. Spike Amplitude:
          │   │     - spikes = (rand() < smoothed²) #Biased towards sustained activity 
          │   │     - amp = clip(spikeAmp×N(0,1), minspike, maxspike)
          │   │
          │   └─▶ 4. Calcium Convolution:
          │        - kernel = exp(-t/τ)
          │        - activity = convolve(spikes, kernel)
          │
          ├─▶ Create Base Movie
          │   ├─▶ 1. Initialize:
          │   │     - movie = np.tile(IMVol_Avg, (1,1,1,T))
          │   │
          │   ├─▶ 2. Per-Synapse Processing:
          │   │     │
          │   │     ├─▶ a. Extract Subvolume:
          │   │     │     - S = mov[zz±sw/2, rr±sw, cc±sw] #3D of each synpase
          │   │     │
          │   │     ├─▶ b. Shift Kernel:
          │   │     │     - skernel shifted by (dz,dr,dc)
          │   │     │
          │   │     ├─▶ c. Filter Application:
          │   │     │     - sFilt = S * shifted_kernel #Elementwise shift
          │   │     │
          │   │     └─▶ d. Accumulate Signals:
          │   │         - movie[region] += sFilt × activity[siteN,:] #Add activity signals to movie
          │   │
          │   └─▶ 3. Store ROIs:
          │        - idealFilts[region,siteN] = sFilt
          │
          ├─▶ Motion Simulation
          │   ├─▶ 1. Generate Motion Components:
          │   │     - PC1-3 = conv(N(0,1)^3, boxcar(40)) × envelope × motionAmp #Generate per axis signals
          │   │     - envelope = sin²(cumsum(N(0,1)/20)) #Slow modulation
          │   │
          │   ├─▶ 2. 3D Rotation:
          │   │     - Build Euler matrix R(ψ,θ,φ) #Random ψ, θ, φ angles
          │   │     - motion = R @ [PC1; PC2; PC3] #Apply 3D rotation
          │   │
          │   └─▶ 3. Scaled Motion:
          │        - GT["motionR"] = 1×motion[0] (X-axis) 
          │        - GT["motionC"] = 0.25×motion[1] (Y-axis)
          │        - GT["motionZ"] = 0.15×motion[2] (Z-axis)
          │
          ├─▶ Apply Motion & Noise
          │   ├─▶ 1. Per-Frame Processing:
          │   │     │
          │   │     ├─▶ a. XY Warping:
          │   │     │     - cv2.warpAffine() with M = [[1,0,ΔC],[0,1,ΔR]] #ΔC and ΔR are motionC and motionR in time
          │   │     │
          │   │     ├─▶ b. Z Interpolation:
          │   │     │     - α = fractional part of z-shift
          │   │     │     - blend = (1-α)×frame_below + α×frame_above
          │   │     │
          │   │     └─▶ c. Photon Conversion:
          │   │         - λ = blend × brightness×exp(-t/bleachTau) + darkrate
          │   │
          │   └─▶ 2. Noise Injection:
          │        ├─▶ Poisson: Ad = poisson(λ) × photonScale 
          │        └─▶ Excess Noise: Ad *= clip(excessNoise, 0.5, 2)
          │
          └─▶ Save Outputs
               ├─▶ 1. Reshape Data:
               │     - Ad_reshaped = Ad.transpose(3,0,1,2)
               │
               ├─▶ 2. Write Files:
               │     │
               │     ├─▶ IF writetiff → tifffile.imwrite()
               │     │
               │     └─▶ ELSE → h5py.create_dataset(compression="gzip")
               │
               └─▶ 3. Save Ground Truth:
                    - HDF5: R/C/Z coordinates, motion vectors, activity, ROIs
                    - JSON: params

END
```

```mermaid
%%{init: {'themeVariables': { 'primaryColor': '#E3F2FD','edgeLabelBackground':'#FFF'}}}%%
flowchart TD;
    A([Start]) --> B[Parse CLI Arguments]
    B --> C[Initialize Parameters]
    C --> D[Create Output Directory]
    D --> E{Find TIFF Files?}
    
    E -->|Yes| F[Process Each TIFF File]
    E -->|No| G([Error Exit])
    
    subgraph FileProcessing [TIFF Processing]
        F --> H[Extract Metadata]
        H --> I[Load Image Volume]
        I --> J[Preprocess Data]
        J --> K[Create Valid Mask]
        K --> L{Synapses Needed?}
        
        L -->|Yes| M[Place Synapses]
        L -->|No| N[Skip Placement]
        
        M --> O[Random Mode?]
        O -->|Yes| P[Random Placement]
        O -->|No| Q[Distance-Constrained Placement]
    end
    
    F --> R[For Each Trial]
    
    subgraph TrialSimulation [Trial Processing]
        R --> S[Generate Activity]
        S --> T[Create Base Movie]
        T --> U[Simulate 3D Motion]
        U --> V[Apply Motion Transform]
        V --> W[Add Photon Noise]
        W --> X[Inject Excess Noise]
    end
    
    R --> Y[Save Outputs]
    
    subgraph DataOutput [Output Handling]
        Y --> Z[Reshape Data]
        Z --> AA{TIFF Format?}
        AA -->|Yes| AB[Write TIFF]
        AA -->|No| AC[Write HDF5]
        Y --> AD[Save Ground Truth]
        AD --> AE[Store Coordinates]
        AD --> AF[Save Motion Vectors]
        AD --> AG[Save Activity Data]
    end
    
    Y --> A2([End])

    style A fill:#4CAF50,stroke:#2E7D32
    style G fill:#B71C1C,stroke:#D32F2F
    style E fill:#FFC107,stroke:#FFA000
    style FileProcessing fill:#E1F5FE,stroke:#039BE5
    style TrialSimulation fill:#F3E5F5,stroke:#9C27B0
    style DataOutput fill:#E8F5E9,stroke:#43A047
    style A2 fill:#4CAF50,stroke:#2E7D32

    classDef decision fill:#FFECB3,stroke:#FFA000;
    classDef process fill:#B3E5FC,stroke:#039BE5;
    classDef io fill:#C8E6C9,stroke:#43A047;
    class E,L,O,AA decision
    class B,C,D,H,I,J,K,M,N,P,Q,S,T,U,V,W,X,Y,Z,AD,AE,AF,AG process
    class F,R,AB,AC,AE,AF,AG io

```
