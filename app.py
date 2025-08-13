import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, mapping, box
import rasterio
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from patchify import patchify, unpatchify
import datetime
import torch
import torchvision
import math
from scipy import ndimage
import ee
import tempfile
import zipfile
from pathlib import Path
import requests
import time
import io
import warnings
import sys
import base64
import json
from importlib import import_module
import geopandas as gpd
import subprocess

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="Examining the path of torch.classes")

# Now import streamlit as the first streamlit-related import
import streamlit as st

# Set page configuration - MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(layout="wide", page_title="Water Body Detection Tool")

# Now import other streamlit-related packages
import folium
from streamlit_folium import folium_static, st_folium
import geemap
import segmentation_models_pytorch as smp
from tqdm import tqdm

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = None
if 'saved_patches_paths' not in st.session_state:
    st.session_state.saved_patches_paths = []
if 'patches_shape' not in st.session_state:
    st.session_state.patches_shape = None
if 'patches_info' not in st.session_state:
    st.session_state.patches_info = {}

# Function to download model from Google Drive
def download_model_from_gdrive(gdrive_url, local_filename):
    """
    Download a file from Google Drive using the sharing URL with improved error handling
    """
    try:
        # Extract file ID from the URL
        correct_file_id = "1PwvxqaN7K3OLN8i2cyqqdhS7vburSmkH"
        
        st.info(f"Downloading model from Google Drive (File ID: {correct_file_id})...")
        
        # Try installing gdown if not available
        try:
            import gdown
        except ImportError:
            st.info("Installing gdown library for reliable Google Drive downloads...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
        
        # Try multiple download approaches
        download_methods = [
            f"https://drive.google.com/uc?id={correct_file_id}",
            f"https://drive.google.com/uc?export=download&id={correct_file_id}",
            correct_file_id # Sometimes gdown works with just the file ID
        ]
        
        for i, method in enumerate(download_methods):
            try:
                st.info(f"Trying download method {i+1}/3...")
                
                # Use gdown with fuzzy matching
                gdown.download(method, local_filename, quiet=False, fuzzy=True)
                
                # Verify the downloaded file
                if os.path.exists(local_filename) and os.path.getsize(local_filename) > 1024:
                    file_size = os.path.getsize(local_filename)
                    
                    # Verify it's a valid PyTorch file
                    try:
                        with open(local_filename, 'rb') as f:
                            header = f.read(10)
                            if header.startswith(b'\x80\x02') or header.startswith(b'\x80\x03') or header.startswith(b'PK'):
                                st.success(f"Model downloaded successfully! Size: {file_size / (1024*1024):.1f} MB")
                                return local_filename
                            else:
                                st.warning(f"Method {i+1}: Downloaded file doesn't appear to be a valid PyTorch model.")
                                if os.path.exists(local_filename):
                                    os.remove(local_filename)
                    except Exception as e:
                        st.warning(f"Method {i+1}: Error verifying downloaded file: {e}")
                        if os.path.exists(local_filename):
                            os.remove(local_filename)
                else:
                    st.warning(f"Method {i+1}: Downloaded file is empty or doesn't exist")
                    if os.path.exists(local_filename):
                        os.remove(local_filename)
                
            except Exception as e:
                st.warning(f"Download method {i+1} failed: {str(e)}")
                if os.path.exists(local_filename):
                    os.remove(local_filename)
                continue
        
        # If all methods failed, show manual instructions
        st.error("All automatic download methods failed.")
        return None
        
    except Exception as e:
        st.error(f"Error in download function: {str(e)}")
        return None

# Model download section
def handle_model_download():
    """Handle model download with manual upload fallback"""
    gdrive_model_url = "https://drive.google.com/file/d/1PwvxqaN7K3OLN8i2cyqqdhS7vburSmkH/view?usp=drive_link"
    model_path = "best_model.pth"
    
    # Check if model exists locally
    if os.path.exists(model_path):
        st.success("Model found locally!")
        return model_path
    
    # Try automatic download
    st.info("Model not found locally. Attempting to download from Google Drive...")
    
    downloaded_model_path = download_model_from_gdrive(gdrive_model_url, model_path)
    
    if downloaded_model_path:
        return downloaded_model_path
    
    # Manual download fallback
    st.error("**Automatic download failed. Please download manually:**")
    
    st.markdown(f"""
    ### Manual Download Instructions:
    
    1. **Click this link to open Google Drive:** 
    [Download Model File](https://drive.google.com/file/d/1PwvxqaN7K3OLN8i2cyqqdhS7vburSmkH/view)
    
    2. **If you see a permission error:**
    - The file owner needs to change sharing to "Anyone with the link can view"
    - Or you may need to request access from the file owner
    
    3. **Click the Download button** (usually in the top-right corner)
    
    4. **Upload the downloaded file using the uploader below**
    """)
    
    # File uploader for manual upload
    uploaded_file = st.file_uploader(
        "Upload the model file (best_model.pth) after manual download:",
        type=['pt', 'pth'],
        help="Download the file manually from Google Drive and upload it here"
    )
    
    if uploaded_file is not None:
        # Save the uploaded file
        with open(model_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        file_size = os.path.getsize(model_path)
        st.success(f"Model uploaded successfully! Size: {file_size / (1024*1024):.1f} MB")
        return model_path
    
    return None

# Initialize the model download
model_path = handle_model_download()

if model_path is None:
    st.error("Model is required to run the application. Please download and upload the model file.")
    st.stop()

# Verify the model file
if os.path.exists(model_path):
    try:
        file_size = os.path.getsize(model_path)
        st.info(f"Model file size: {file_size / (1024*1024):.1f} MB")
        
        # Try to load just the header to verify it's a valid PyTorch file
        with open(model_path, 'rb') as f:
            header = f.read(10)
            if not (header.startswith(b'\x80\x02') or header.startswith(b'\x80\x03') or header.startswith(b'PK')):
                st.error("The model file appears to be corrupted or invalid.")
                st.error(f"File header: {header}")
                st.info("Please re-download the model file.")
                os.remove(model_path)
                st.stop()
            else:
                st.success("Model file appears to be valid!")
        
    except Exception as e:
        st.error(f"Error verifying model file: {e}")
        st.stop()

# Earth Engine Authentication
def authenticate_earth_engine():
    """Authenticate with Google Earth Engine using service account from environment variable"""
    try:
        # Get the base64 encoded key from environment variable
        base64_key = os.environ.get('GOOGLE_EARTH_ENGINE_KEY_BASE64')
        
        if not base64_key:
            st.error("""
            **Earth Engine authentication is required to use this application.**
            
            Please set the `GOOGLE_EARTH_ENGINE_KEY_BASE64` environment variable with your service account key.
            
            1. Create a service account in Google Cloud Console
            2. Generate a JSON key for the service account
            3. Convert the JSON key to base64:
            ```python
            import base64
            with open('your-key.json', 'r') as f:
                print(base64.b64encode(f.read().encode()).decode())
            ```
            4. Set this as an environment variable in your Posit Cloud environment
            """)
            st.stop()
            return False
        
        # Decode the base64 string
        key_json = base64.b64decode(base64_key).decode()
        key_data = json.loads(key_json)
        
        # Create a temporary file for the key
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as key_file:
            json.dump(key_data, key_file)
            key_file_path = key_file.name
        
        # Initialize Earth Engine with the service account credentials
        credentials = ee.ServiceAccountCredentials(
            key_data['client_email'],
            key_file_path
        )
        ee.Initialize(credentials)
        
        # Clean up the temporary file
        os.unlink(key_file_path)
        
        st.success("✅ Successfully authenticated with Google Earth Engine!")
        return True
        
    except Exception as e:
        st.error(f"❌ Error authenticating with Earth Engine: {str(e)}")
        st.info("Please check your service account key and environment variable.")
        return False

# Authenticate with Earth Engine
if not authenticate_earth_engine():
    st.stop()

# Load water segmentation model
@st.cache_resource
def load_water_model(model_path):
    """Load the water segmentation model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.info(f"Using device: {device}")
        
        # Try to load the model
        try:
            checkpoint = torch.load(model_path, map_location=device)
            
            # Determine if it's a full model or state dict
            if isinstance(checkpoint, torch.nn.Module):
                st.info("Loaded model directly")
                model = checkpoint
            else:
                # Try to initialize the model architecture
                st.info("Loaded state dict, initializing model architecture")
                
                # Initialize with the right architecture for water detection
                model = smp.Unet(
                    encoder_name='resnet34',
                    encoder_weights=None,
                    in_channels=6, # 6 bands for water detection
                    classes=1,
                    activation=None
                ).to(device)
                
                # Load the state dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            
            # Set model to evaluation mode
            model.eval()
            
            # Test with a dummy input to verify shape
            dummy_input = torch.zeros((1, 6, 224, 224), device=device)
            try:
                with torch.no_grad():
                    _ = model(dummy_input)
                st.info("Model test successful with input shape (1, 6, 224, 224)")
            except Exception as test_error:
                st.warning(f"Model test failed: {test_error}")
                st.info("Trying alternative architectures...")
                
                # If test failed, try alternative architectures
                for encoder in ['resnet18', 'resnet34', 'resnet50']:
                    for in_channels in [6, 12]:
                        try:
                            st.info(f"Trying {encoder} with {in_channels} input channels")
                            model = smp.Unet(
                                encoder_name=encoder,
                                encoder_weights=None,
                                in_channels=in_channels,
                                classes=1,
                                activation=None
                            ).to(device)
                            
                            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                                model.load_state_dict(checkpoint['model_state_dict'])
                            else:
                                model.load_state_dict(checkpoint)
                            
                            model.eval()
                            dummy_input = torch.zeros((1, in_channels, 224, 224), device=device)
                            with torch.no_grad():
                                _ = model(dummy_input)
                            st.success(f"Success with {encoder} and {in_channels} input channels")
                            break
                        except Exception:
                            continue
            
            st.session_state.model_loaded = True
            st.success("✅ Water segmentation model loaded successfully!")
            return model, device
            
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            st.session_state.model_loaded = False
            return None, None
        
    except Exception as e:
        st.error(f"❌ Error loading water model: {str(e)}")
        st.session_state.model_loaded = False
        return None, None

# Function to download Sentinel-2 data
def download_sentinel2_data(roi_geojson, start_date, end_date):
    """Download Sentinel-2 data for the given ROI and date range"""
    try:
        # Create Earth Engine geometry from GeoJSON
        roi_ee = ee.Geometry(roi_geojson)
        
        # Get Sentinel-2 collection
        collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
            .filterBounds(roi_ee) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
        
        # Get the least cloudy image
        image = collection.sort('CLOUDY_PIXEL_PERCENTAGE').first()
        
        # Select all bands
        all_bands = image.select(['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'])
        
        # Clip to ROI
        clipped = all_bands.clip(roi_ee)
        
        # Get download URL
        url = clipped.getDownloadUrl({
            'scale': 10,
            'crs': 'EPSG:4326',
            'format': 'GEO_TIFF',
            'region': roi_ee
        })
        
        # Download the image
        response = requests.get(url)
        if response.status_code == 200:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                tmp_file.write(response.content)
            return tmp_file.name
        else:
            st.error(f"Failed to download: {response.status_code}")
            return None
        
    except Exception as e:
        st.error(f"Error downloading Sentinel-2 data: {str(e)}")
        return None

# Function to process a single region
def process_single_region(roi_geojson, image_path, region_number, progress_placeholder, status_placeholder):
    """Process a single region for water detection"""
    try:
        # 1. CLIPPING STEP
        status_placeholder.info("Step 1/4: Clipping image to ROI...")
        
        # Create a temporary directory for outputs
        temp_dir = tempfile.mkdtemp()
        
        # Clip the image to ROI
        with rasterio.open(image_path) as src:
            # Create mask from ROI
            roi_shape = [mapping(Polygon(roi_geojson['coordinates'][0]))]
            
            # Mask the image
            out_image, out_transform = mask(src, roi_shape, crop=True)
            
            # Update metadata
            clipped_meta = src.meta.copy()
            clipped_meta.update({
                "driver": "GTiff",
                "height": out_image.shape[1],
                "width": out_image.shape[2],
                "transform": out_transform
            })
            
            # Extract water bands
            water_bands_img = out_image[[1, 2, 3, 7, 10, 11]] # B2, B3, B4, B8, B11, B12
            
            # Debug info
            st.info(f"Clipped image shape: {water_bands_img.shape}")
            st.info(f"Number of bands for water detection: {water_bands_img.shape[0]}")
        
        progress_placeholder.progress(25)
        status_placeholder.success("Step 1/4: Image clipped successfully")
        
        # 2. PATCHING STEP
        status_placeholder.info("Step 2/4: Creating patches...")
        
        # Prepare the image for patching (keep raw values, no normalization)
        img_for_patching = np.moveaxis(water_bands_img, 0, -1) # Change to H x W x C format
        
        # Debug info
        st.info(f"Image shape for patching: {img_for_patching.shape}")
        st.info(f"Number of bands: {water_bands_img.shape[0]}")
        
        # Create patches
        patch_size = 224
        
        # Check if image is big enough for patching
        if img_for_patching.shape[0] < patch_size or img_for_patching.shape[1] < patch_size:
            status_placeholder.error(f"Image too small for patching: {img_for_patching.shape[:2]}. Need at least {patch_size}x{patch_size}")
            return False
        
        patches = patchify(img_for_patching, (patch_size, patch_size, water_bands_img.shape[0]), step=patch_size)
        
        # Handle different patch array dimensions
        if len(patches.shape) == 6:
            # patchify returns 6D array when step < patch_size
            patches = patches.squeeze(axis=(2, 5))
        elif len(patches.shape) == 5:
            # Sometimes returns 5D array
            patches = patches.squeeze()
        
        st.info(f"Patches shape after squeeze: {patches.shape}")
        
        # Create output directory
        base_dir = os.path.dirname(image_path)
        output_folder = os.path.join(base_dir, f"water_patches_region{region_number}")
        os.makedirs(output_folder, exist_ok=True)
        
        # Save patches
        num_patches = patches.shape[0] * patches.shape[1]
        saved_paths = []
        
        # Create a sub-progress bar for patches
        patch_progress = st.progress(0)
        
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                patch = patches[i, j]
                
                # Debug first patch
                if i == 0 and j == 0:
                    st.info(f"First patch shape: {patch.shape}")
                    st.info(f"First patch dtype: {patch.dtype}")
                
                # NO NORMALIZATION - keep raw values for model input
                # Convert to rasterio format (C x H x W)
                patch_for_saving = np.moveaxis(patch, -1, 0)
                
                # Create filename
                patch_name = f"water_region{region_number}_{i}_{j}.tif"
                output_file_path = os.path.join(output_folder, patch_name)
                saved_paths.append(output_file_path)
                
                # Save the patch with raw values
                with rasterio.open(
                    output_file_path,
                    'w',
                    driver='GTiff',
                    height=patch_size,
                    width=patch_size,
                    count=patch_for_saving.shape[0],
                    dtype=patch_for_saving.dtype, # Keep original dtype
                    crs=clipped_meta.get('crs'),
                    transform=clipped_meta.get('transform')
                ) as dst:
                    for band_idx in range(patch_for_saving.shape[0]):
                        band_data = patch_for_saving[band_idx]
                        # Ensure band_data is 2D
                        if band_data.ndim != 2:
                            st.error(f"Band {band_idx} has shape {band_data.shape}, expected 2D")
                            band_data = band_data.reshape(patch_size, patch_size)
                        dst.write(band_data, band_idx+1)
                
                # Update patch progress
                patch_count = i * patches.shape[1] + j + 1
                patch_progress.progress(patch_count / num_patches)
        
        # Store the saved paths in session state
        st.session_state.saved_patches_paths = saved_paths
        st.session_state.patches_shape = patches.shape
        st.session_state.patches_info = {
            'region_number': region_number,
            'output_folder': output_folder,
            'patch_size': patch_size,
            'num_bands': water_bands_img.shape[0] # Store number of bands
        }
        
        # Display sample patches
        num_samples = min(6, num_patches)
        if num_samples > 0:
            st.subheader("Sample Patches")
            fig, axes = plt.subplots(1, num_samples, figsize=(15, 4))
            if num_samples == 1:
                axes = [axes]
            
            for idx, ax in enumerate(axes):
                if idx < num_patches:
                    i, j = idx // patches.shape[1], idx % patches.shape[1]
                    patch = patches[i, j]
                    
                    # For visualization, use RGB bands
                    rgb_bands = [2, 1, 0] # B4, B3, B2
                    patch_rgb = np.zeros((patch_size, patch_size, 3), dtype=np.float32)
                    for b, band in enumerate(rgb_bands):
                        if band < patch.shape[-1]:
                            band_data = patch[:, :, band]
                            min_val = np.percentile(band_data, 2)
                            max_val = np.percentile(band_data, 98)
                            patch_rgb[:, :, b] = np.clip((band_data - min_val) / (max_val - min_val), 0, 1)
                    
                    ax.imshow(patch_rgb)
                    ax.set_title(f"Patch {i}_{j}")
                    ax.axis('off')
            
            st.pyplot(fig)
        
        progress_placeholder.progress(50)
        status_placeholder.success(f"Step 2/4: Created {num_patches} patches successfully")
        
        # 3. WATER DETECTION STEP
        status_placeholder.info("Step 3/4: Detecting water bodies...")
        
        # Load model if not already loaded
        if not st.session_state.model_loaded:
            with st.spinner("Loading water segmentation model..."):
                model, device = load_water_model(model_path)
                if model is not None:
                    st.session_state.model = model
                    st.session_state.device = device
                    st.session_state.model_loaded = True
                else:
                    status_placeholder.error("Failed to load water segmentation model.")
                    return False
        
        # Get saved paths
        patches_info = st.session_state.patches_info
        saved_paths = st.session_state.saved_patches_paths
        patches_shape = st.session_state.patches_shape
        
        # Create output directory for water masks
        base_dir = os.path.dirname(image_path)
        water_folder = os.path.join(base_dir, f"water_masks_region{region_number}")
        os.makedirs(water_folder, exist_ok=True)
        
        # Track water detection results
        water_results = []
        water_paths = []
        
        # Create a sub-progress bar for detection
        detect_progress = st.progress(0)
        
        # Process each patch
        total_patches = len(saved_paths)
        for idx, patch_path in enumerate(saved_paths):
            try:
                # Extract i, j from filename
                filename = os.path.basename(patch_path)
                parts = filename.split('_')
                i = int(parts[-2])
                j = int(parts[-1].split('.')[0])
                
                # Read the patch (raw values, no scaling)
                with rasterio.open(patch_path) as src:
                    patch = src.read() # This will be in (C, H, W) format
                    patch_meta = src.meta.copy()
                
                # Debug info for first patch
                if idx == 0:
                    st.info(f"First patch read shape: {patch.shape}")
                    st.info(f"Expected shape: ({patches_info['num_bands']}, {patch_size}, {patch_size})")
                
                # Verify patch shape
                if patch.shape != (patches_info['num_bands'], patch_size, patch_size):
                    st.error(f"Unexpected patch shape: {patch.shape}")
                    st.error(f"Expected: ({patches_info['num_bands']}, {patch_size}, {patch_size})")
                    
                    # Try to diagnose the issue
                    total_elements = patch.size
                    st.info(f"Total elements in patch: {total_elements}")
                    st.info(f"Patch dimensions: {patch.shape}")
                    
                    # Skip this patch
                    continue
                
                # Create RGB version for display
                rgb_bands = [2, 1, 0] # B4, B3, B2
                rgb_patch = np.zeros((patch.shape[1], patch.shape[2], 3), dtype=np.float32)
                
                for b, band in enumerate(rgb_bands):
                    if band < patch.shape[0]:
                        band_data = patch[band]
                        min_val = np.percentile(band_data, 2)
                        max_val = np.percentile(band_data, 98)
                        rgb_patch[:, :, b] = np.clip((band_data - min_val) / (max_val - min_val), 0, 1)
                
                # Convert to torch tensor (keep raw values for model)
                img_tensor = torch.tensor(patch.astype(np.float32), dtype=torch.float32)
                
                # Debug tensor shape
                if idx == 0:
                    st.info(f"Tensor shape before unsqueeze: {img_tensor.shape}")
                
                img_tensor = img_tensor.unsqueeze(0) # Add batch dimension
                
                if idx == 0:
                    st.info(f"Tensor shape after unsqueeze: {img_tensor.shape}")
                
                # Verify tensor shape is correct for model
                if img_tensor.shape != (1, patches_info['num_bands'], patch_size, patch_size):
                    st.error(f"Invalid tensor shape for model: {img_tensor.shape}")
                    st.error(f"Expected: (1, {patches_info['num_bands']}, {patch_size}, {patch_size})")
                    continue
                
                # Perform water detection
                with torch.inference_mode():
                    try:
                        prediction = st.session_state.model(img_tensor.to(st.session_state.device))
                        prediction = torch.sigmoid(prediction).cpu()
                    except RuntimeError as e:
                        st.error(f"Model inference error: {str(e)}")
                        st.error(f"Input shape: {img_tensor.shape}")
                        st.error(f"Model expects input channels: Check your model architecture")
                        raise e
                
                # Convert prediction to numpy
                pred_np = prediction.squeeze().numpy()
                
                # Debug prediction shape
                if idx == 0:
                    st.info(f"Prediction shape: {pred_np.shape}")
                
                # Create binary mask (threshold at 0.5)
                water_mask = (pred_np > 0.5).astype(np.uint8) * 255
                
                # Save water mask
                output_filename = f"water_mask_region{region_number}_{i}_{j}.tif"
                output_path = os.path.join(water_folder, output_filename)
                water_paths.append(output_path)
                
                # Save as GeoTIFF
                patch_meta.update({
                    'count': 1,
                    'dtype': 'uint8'
                })
                
                # Ensure water_mask is 2D
                if water_mask.ndim == 3:
                    water_mask = water_mask.squeeze()
                
                with rasterio.open(output_path, 'w', **patch_meta) as dst:
                    dst.write(water_mask.reshape(1, water_mask.shape[0], water_mask.shape[1]))
                
                # Add to display list (limit to 6)
                if len(water_results) < 6:
                    water_results.append({
                        'path': output_path,
                        'i': i,
                        'j': j,
                        'mask': water_mask,
                        'rgb_original': rgb_patch
                    })
                
                # Update detection progress
                detect_progress.progress((idx + 1) / total_patches)
                
            except Exception as e:
                st.error(f"Error processing patch {patch_path}: {str(e)}")
                import traceback
                st.error("Full traceback:")
                st.error(traceback.format_exc())
                
                # Try to provide more diagnostic information
                try:
                    with rasterio.open(patch_path) as src:
                        st.error(f"Patch metadata: {src.meta}")
                        st.error(f"Patch shape from file: {src.read().shape}")
                except:
                    pass
        
        # Display water detection results
        if water_results:
            st.subheader("Water Detection Results (Sample)")
            fig, axes = plt.subplots(2, len(water_results), figsize=(15, 8))
            if len(water_results) == 1:
                axes = axes.reshape(2, 1)
            
            for idx, result in enumerate(water_results):
                # Original RGB
                axes[0, idx].imshow(result['rgb_original'])
                axes[0, idx].set_title(f"Original Patch {result['i']}_{result['j']}")
                axes[0, idx].axis('off')
                
                # Water mask
                axes[1, idx].imshow(result['mask'], cmap='Blues')
                axes[1, idx].set_title(f"Water Mask {result['i']}_{result['j']}")
                axes[1, idx].axis('off')
            
            st.pyplot(fig)
        
        progress_placeholder.progress(75)
        status_placeholder.success(f"Step 3/4: Water detection completed for {len(water_paths)} patches")
        
        # 4. RECONSTRUCTION STEP
        status_placeholder.info("Step 4/4: Reconstructing full water mask...")
        
        # Reconstruct the full mask from patches
        reconstructed_shape = (
            patches_shape[0] * patch_size,
            patches_shape[1] * patch_size
        )
        
        reconstructed_mask = np.zeros(reconstructed_shape, dtype=np.uint8)
        
        # Place each patch in the correct position
        for water_path in water_paths:
            # Extract i, j from filename
            filename = os.path.basename(water_path)
            parts = filename.split('_')
            i = int(parts[-2])
            j = int(parts[-1].split('.')[0])
            
            # Read the water mask
            with rasterio.open(water_path) as src:
                patch_mask = src.read(1)
            
            # Place in reconstructed mask
            row_start = i * patch_size
            row_end = row_start + patch_size
            col_start = j * patch_size
            col_end = col_start + patch_size
            
            reconstructed_mask[row_start:row_end, col_start:col_end] = patch_mask
        
        # Save the reconstructed mask
        output_filename = f"water_mask_full_region{region_number}.tif"
        output_path = os.path.join(base_dir, output_filename)
        
        # Use the original clipped metadata
        final_meta = clipped_meta.copy()
        final_meta.update({
            'count': 1,
            'dtype': 'uint8'
        })
        
        with rasterio.open(output_path, 'w', **final_meta) as dst:
            dst.write(reconstructed_mask.reshape(1, reconstructed_mask.shape[0], reconstructed_mask.shape[1]))
        
        # Display the final result
        st.subheader("Final Water Mask")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original image (RGB composite)
        rgb_composite = np.zeros((water_bands_img.shape[1], water_bands_img.shape[2], 3), dtype=np.float32)
        for b, band in enumerate([2, 1, 0]): # B4, B3, B2
            if band < water_bands_img.shape[0]:
                band_data = water_bands_img[band]
                min_val = np.percentile(band_data, 2)
                max_val = np.percentile(band_data, 98)
                rgb_composite[:, :, b] = np.clip((band_data - min_val) / (max_val - min_val), 0, 1)
        
        ax1.imshow(rgb_composite)
        ax1.set_title(f"Original Image - Region {region_number}")
        ax1.axis('off')
        
        # Water mask
        ax2.imshow(reconstructed_mask, cmap='Blues')
        ax2.set_title(f"Water Mask - Region {region_number}")
        ax2.axis('off')
        
        st.pyplot(fig)
        
        # Calculate water statistics
        water_pixels = np.sum(reconstructed_mask > 0)
        total_pixels = reconstructed_mask.size
        water_percentage = (water_pixels / total_pixels) * 100
        
        st.info(f"Water coverage: {water_percentage:.2f}% ({water_pixels:,} pixels out of {total_pixels:,})")
        
        progress_placeholder.progress(100)
        status_placeholder.success(f"✅ Processing completed for Region {region_number}!")
        
        # Create download button for the final mask
        with open(output_path, 'rb') as f:
            st.download_button(
                label=f"Download Water Mask - Region {region_number}",
                data=f.read(),
                file_name=output_filename,
                mime='image/tiff'
            )
        
        return True
        
    except Exception as e:
        st.error(f"Error processing region {region_number}: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

# Main Streamlit app
def main():
    st.title("🌊 Water Body Detection Tool")
    st.markdown("Detect water bodies in satellite imagery using deep learning")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("📍 Region Selection")
        
        # Option 1: Draw on map
        st.subheader("Option 1: Draw Region")
        
        # Create a map centered on a default location
        m = folium.Map(location=[35.6892, 51.3890], zoom_start=10)
        
        # Add drawing controls
        draw = folium.plugins.Draw(
            export=True,
            position='topleft',
            draw_options={
                'rectangle': True,
                'polygon': True,
                'circle': False,
                'marker': False,
                'circlemarker': False,
                'polyline': False
            }
        )
        draw.add_to(m)
        
        # Display map
        map_data = st_folium(m, key="map", height=400, width=None)
        
        # Option 2: Upload GeoJSON
        st.subheader("Option 2: Upload GeoJSON")
        uploaded_geojson = st.file_uploader("Upload GeoJSON file", type=['geojson', 'json'])
        
        # Option 3: Manual coordinates
        st.subheader("Option 3: Manual Coordinates")
        coords_text = st.text_area(
            "Enter coordinates (format: lon,lat per line)",
            placeholder="51.3890,35.6892\n51.4890,35.6892\n51.4890,35.7892\n51.3890,35.7892"
        )
        
        # Date selection
        st.header("📅 Date Selection")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.date(2024, 1, 1))
        with col2:
            end_date = st.date_input("End Date", datetime.date(2024, 12, 31))
        
        # Process button
        process_button = st.button("🚀 Start Processing", type="primary", use_container_width=True)
    
    # Main area
    st.header("🗺️ Selected Region")
    
    # Initialize variables
    roi_geojson = None
    regions = []
    
    # Get ROI from different sources
    if map_data and map_data.get('all_drawings'):
        # Get drawn shapes
        for drawing in map_data['all_drawings']:
            if drawing['type'] in ['Polygon', 'Rectangle']:
                roi_geojson = drawing['geometry']
                regions.append(roi_geojson)
        st.success(f"✅ Region selected from map")
    
    elif uploaded_geojson is not None:
        # Parse uploaded GeoJSON
        try:
            geojson_data = json.load(uploaded_geojson)
            if geojson_data['type'] == 'FeatureCollection':
                for feature in geojson_data['features']:
                    if feature['geometry']['type'] == 'Polygon':
                        regions.append(feature['geometry'])
            elif geojson_data['type'] == 'Polygon':
                regions.append(geojson_data)
            st.success(f"✅ Loaded {len(regions)} region(s) from GeoJSON")
        except Exception as e:
            st.error(f"Error parsing GeoJSON: {str(e)}")
    
    elif coords_text:
        # Parse manual coordinates
        try:
            lines = coords_text.strip().split('\n')
            coords = []
            for line in lines:
                lon, lat = map(float, line.split(','))
                coords.append([lon, lat])
            
            if len(coords) >= 3:
                # Close the polygon
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                
                roi_geojson = {
                    "type": "Polygon",
                    "coordinates": [coords]
                }
                regions.append(roi_geojson)
                st.success("✅ Region created from manual coordinates")
            else:
                st.warning("⚠️ Need at least 3 coordinates to create a polygon")
        except Exception as e:
            st.error(f"Error parsing coordinates: {str(e)}")
    
    # Display selected regions
    if regions:
        # Create a map showing all regions
        display_map = folium.Map(location=[35.6892, 51.3890], zoom_start=10)
        
        for idx, region in enumerate(regions):
            # Add region to map
            folium.GeoJson(
                region,
                name=f"Region {idx + 1}",
                style_function=lambda x: {
                    'fillColor': '#3388ff',
                    'color': '#3388ff',
                    'weight': 2,
                    'fillOpacity': 0.3
                }
            ).add_to(display_map)
            
            # Calculate bounds
            coords = region['coordinates'][0]
            lats = [c[1] for c in coords]
            lons = [c[0] for c in coords]
            bounds = [[min(lats), min(lons)], [max(lats), max(lons)]]
            display_map.fit_bounds(bounds)
        
        # Display the map
        folium_static(display_map)
        
        # Show region info
        st.info(f"📍 {len(regions)} region(s) selected")
    else:
        st.warning("⚠️ No region selected. Please select a region using one of the methods above.")
    
    # Process regions
    if process_button and regions:
        st.header("🔄 Processing")
        
        # Create columns for progress tracking
        progress_col, status_col = st.columns([2, 3])
        with progress_col:
            overall_progress = st.progress(0)
            progress_text = st.empty()
        with status_col:
            status_placeholder = st.empty()
        
        # Download Sentinel-2 data for all regions
        with st.spinner("Downloading Sentinel-2 imagery..."):
            # For simplicity, we'll use the bounding box of all regions
            all_coords = []
            for region in regions:
                all_coords.extend(region['coordinates'][0])
            
            lats = [c[1] for c in all_coords]
            lons = [c[0] for c in all_coords]
            
            # Create bounding box
            bbox_geojson = {
                "type": "Polygon",
                "coordinates": [[
                    [min(lons), min(lats)],
                    [max(lons), min(lats)],
                    [max(lons), max(lats)],
                    [min(lons), max(lats)],
                    [min(lons), min(lats)]
                ]]
            }
            
            # Download Sentinel-2 data
            zip_path = download_sentinel2_data(
                bbox_geojson,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if zip_path:
                # Extract the downloaded zip file
                extract_dir = tempfile.mkdtemp()
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                # Find the .tif file
                tif_files = list(Path(extract_dir).glob('*.tif'))
                if tif_files:
                    image_path = str(tif_files[0])
                    st.success(f"✅ Downloaded Sentinel-2 image: {os.path.basename(image_path)}")
                else:
                    st.error("❌ No TIF file found in downloaded data")
                    return
            else:
                st.error("❌ Failed to download Sentinel-2 data")
                return
        
        # Process each region
        total_regions = len(regions)
        results = []
        
        for idx, region in enumerate(regions):
            progress_text.text(f"Processing Region {idx + 1} of {total_regions}")
            overall_progress.progress((idx) / total_regions)
            
            # Create a sub-container for this region
            with st.container():
                st.subheader(f"Region {idx + 1}")
                
                # Create placeholders for this region
                region_progress = st.progress(0)
                region_status = st.empty()
                
                # Process the region
                success = process_single_region(
                    region,
                    image_path,
                    idx + 1,
                    region_progress,
                    region_status
                )
                
                results.append(success)
        
        # Final summary
        overall_progress.progress(100)
        successful = sum(results)
        st.success(f"✅ Processing complete! Successfully processed {successful}/{total_regions} regions")
        
        # Cleanup
        try:
            os.remove(zip_path)
            import shutil
            shutil.rmtree(extract_dir)
        except:
            pass

# Run the app
if __name__ == "__main__":
    main()
