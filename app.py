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

# Function to download model from Google Drive with fixed URL and better error handling
@st.cache_data
def download_model_from_gdrive(gdrive_url, local_filename):
    """
    Download a file from Google Drive using the sharing URL with improved error handling
    
    Parameters:
    gdrive_url (str): Google Drive sharing URL
    local_filename (str): Local filename to save the downloaded file
    
    Returns:
    str: Path to the downloaded file or None if download failed
    """
    try:
        # Extract file ID from the new URL
        # URL: https://drive.google.com/file/d/1PwvxqaN7K3OLN8i2cyqqdhS7vburSmkH/view?usp=drive_link
        correct_file_id = "1PwvxqaN7K3OLN8i2cyqqdhS7vburSmkH"
        
        st.info(f"Downloading model from Google Drive (File ID: {correct_file_id})...")
        
        # Use gdown library which is more reliable for Google Drive downloads
        try:
            import gdown
        except ImportError:
            st.info("Installing gdown library for reliable Google Drive downloads...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown
        
        # Try multiple download approaches
        download_methods = [
            f"https://drive.google.com/uc?id={correct_file_id}",
            f"https://drive.google.com/file/d/{correct_file_id}/view",
            correct_file_id  # Sometimes gdown works with just the file ID
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
                            st.warning(f"Method {i+1}: Downloaded file doesn't appear to be a valid PyTorch model. Header: {header}")
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
        
        # If all gdown methods failed, try manual requests approach
        st.info("All gdown methods failed. Trying manual download approach...")
        return manual_download_fallback(correct_file_id, local_filename)
        
    except Exception as e:
        st.error(f"Error in download function: {str(e)}")
        return None

def manual_download_fallback(file_id, local_filename):
    """
    Fallback manual download method using requests
    """
    try:
        import requests
        
        # Try different download URLs
        urls_to_try = [
            f"https://drive.google.com/uc?export=download&id={file_id}",
            f"https://drive.google.com/uc?export=download&id={file_id}&confirm=t",
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download",
        ]
        
        session = requests.Session()
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        
        for i, url in enumerate(urls_to_try):
            try:
                st.info(f"Trying manual method {i+1}/3...")
                response = session.get(url, headers=headers, stream=True, timeout=30)
                
                if response.status_code == 200:
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    
                    if 'text/html' not in content_type:
                        # Create progress bar
                        total_size = int(response.headers.get('content-length', 0))
                        downloaded_size = 0
                        
                        if total_size > 0:
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                        
                        # Download the file
                        with open(local_filename, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    downloaded_size += len(chunk)
                                    
                                    if total_size > 0:
                                        progress = downloaded_size / total_size
                                        progress_bar.progress(progress)
                                        status_text.text(f"Downloaded: {downloaded_size / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB")
                        
                        if total_size > 0:
                            progress_bar.empty()
                            status_text.empty()
                        
                        # Verify file
                        if os.path.exists(local_filename) and os.path.getsize(local_filename) > 1024:
                            file_size = os.path.getsize(local_filename)
                            st.success(f"Manual download successful! Size: {file_size / (1024*1024):.1f} MB")
                            return local_filename
                    else:
                        st.warning(f"Method {i+1} returned HTML instead of file")
                
            except Exception as e:
                st.warning(f"Manual method {i+1} failed: {e}")
                continue
        
        # All methods failed - provide manual instructions
        st.error("All automatic download methods failed. Please download manually:")
        
        st.info("**Manual Download Instructions:**")
        st.markdown(f"""
        1. **Open this  in a new browser tab:** 
        https://drive.google.com/file/d/{file_id}/view
        
        2. **If you see a permission error:**
        - The file owner needs to change sharing to "Anyone with the  can view"
        - Or you may need to request access
        
        3. **Click the Download button** (usually in the top-right corner)
        
        4. **Save the file as:** `{local_filename}`
        
        5. **Upload it using the file uploader below**
        """)
        
        # File uploader as backup
        uploaded_file = st.file_uploader(
            f"Upload the model file ({local_filename}) after manual download:",
            type=['pt', 'pth'],
            help="Download the file manually from Google Drive and upload it here"
        )
        
        if uploaded_file is not None:
            with open(local_filename, 'wb') as f:
                f.write(uploaded_file.read())
            
            file_size = os.path.getsize(local_filename)
            st.success(f"Model uploaded successfully! Size: {file_size / (1024*1024):.1f} MB")
            return local_filename
        
        return None
        
    except Exception as e:
        st.error(f"Manual download fallback failed: {e}")
        return None

# Updated model loading section with the correct URL and file ID
gdrive_model_url = "https://drive.google.com/file/d/1PwvxqaN7K3OLN8i2cyqqdhS7vburSmkH/view?usp=sharing"
model_path = "best_model.pth"  # Updated filename to match your preference

# Download model if it doesn't exist locally
if not os.path.exists(model_path):
    st.info("Model not found locally. Downloading from Google Drive...")
    
    # Try downloading with the corrected URL
    downloaded_model_path = download_model_from_gdrive(gdrive_model_url, model_path)
    
    if downloaded_model_path is None:
        st.error("Automatic download failed. Please use the manual download option above.")
        st.stop()
else:
    st.success("Model found locally!")

# Verify the model file before proceeding
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
            
            # Show file content preview for debugging
            try:
                with open(model_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read(200)
                st.code(content, language='text')
            except:
                pass
            
            os.remove(model_path)
            st.stop()
        else:
            st.success("Model file appears to be valid!")
            
    except Exception as e:
        st.error(f"Error verifying model file: {e}")
        st.stop()

# Initialize Earth Engine with service account from environment variable
@st.cache_resource
def initialize_earth_engine():
    try:
        # Check if Earth Engine is already initialized
        ee.Initialize()
        return True, "Earth Engine already initialized"
    except Exception as e:
        try:
            # Try service account authentication from environment variable
            base64_key = os.environ.get('GOOGLE_EARTH_ENGINE_KEY_BASE64')
            
            if not base64_key:
                return False, "Earth Engine service account key not found in environment variables."
            
            # Decode the base64 string
            key_json = base64.b64decode(base64_key).decode()
            key_data = json.loads(key_json)
            
            # Create a temporary file for the key
            key_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
            with open(key_file.name, 'w') as f:
                json.dump(key_data, f)
            
            # Initialize Earth Engine with the service account credentials
            credentials = ee.ServiceAccountCredentials(
                key_data['client_email'],
                key_file.name
            )
            ee.Initialize(credentials)
            
            # Clean up the temporary file
            os.unlink(key_file.name)
            
            return True, "Successfully authenticated with Earth Engine!"
        except Exception as auth_error:
            return False, f"Authentication failed: {str(auth_error)}"

# Initialize Earth Engine
ee_initialized, ee_message = initialize_earth_engine()
if ee_initialized:
    st.sidebar.success(ee_message)
else:
    st.sidebar.error(ee_message)
    st.error("Earth Engine authentication is required to use this application.")
    st.info("""
    Please set the GOOGLE_EARTH_ENGINE_KEY_BASE64 environment variable with your service account key.
    
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

# Create tabs for different pages
tab1, tab2 = st.tabs(["Region Selection & Download", "Water Body Detection"])

# Global variables
if 'drawn_polygons' not in st.session_state:
    st.session_state.drawn_polygons = []

if 'last_map_data' not in st.session_state:
    st.session_state.last_map_data = None

if 'clipped_img' not in st.session_state:
    st.session_state.clipped_img = None

if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    
if 'saved_patches_paths' not in st.session_state:
    st.session_state.saved_patches_paths = []

if 'water_detection_result' not in st.session_state:
    st.session_state.water_detection_result = None

# Define Sentinel-2 bands to use for water detection
WATER_DETECTION_BANDS = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
WATER_BAND_INDICES = [1, 2, 3, 7, 10, 11]  # 0-indexed positions in S2_BANDS

# All Sentinel-2 bands for download
S2_BANDS = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
S2_NAMES = ['Aerosols', 'Blue', 'Green', 'Red', 'Red Edge 1', 'Red Edge 2', 
            'Red Edge 3', 'NIR', 'Red Edge 4', 'Water Vapor', 'SWIR1', 'SWIR2']

# Function to download Sentinel-2 imagery
def download_sentinel2_with_gees2(date, polygon, cloud_cover_limit=10):
    """
    Download Sentinel-2 Level-2A imagery for a specific date and region.
    """
    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    try:
        # Parse the date
        year, month, day = date.split('-')
        
        # Define date range (±15 days from selected date)
        start_date = (datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.timedelta(days=15)).strftime('%Y-%m-%d')
        end_date = (datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days=15)).strftime('%Y-%m-%d')
        
        # Display the region area in square kilometers
        area_sq_km = polygon.area * 111 * 111  # Approximate conversion from degrees to km²
        status_placeholder.info(f"Selected region area: ~{area_sq_km:.2f} km². Searching for Sentinel-2 images...")
        
        # Create a temporary directory for downloading
        temp_dir = tempfile.mkdtemp()
        output_file = os.path.join(temp_dir, f"sentinel2_{date}_median.tif")
        
        # Convert polygon to GeoJSON format
        geojson = {"type": "Polygon", "coordinates": [list(polygon.exterior.coords)]}
        
        # Initialize Earth Engine geometry
        ee_geometry = ee.Geometry.Polygon(geojson['coordinates'])
        
        # Create Earth Engine image collection
        status_placeholder.info(f"Creating Earth Engine image collection around {date} (±15 days) with cloud cover < {cloud_cover_limit}%...")
        
        # Filter Sentinel-2 collection
        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                     .filterBounds(ee_geometry)
                     .filterDate(start_date, end_date)
                     .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_cover_limit)))
        
        # Get the count of images
        count = collection.size().getInfo()
        status_placeholder.info(f"Found {count} Sentinel-2 images with cloud cover < {cloud_cover_limit}%")
        
        if count == 0:
            status_placeholder.warning(f"No Sentinel-2 images found around {date} with cloud cover < {cloud_cover_limit}%")
            return None
        
        # Create median composite - NO SCALING APPLIED, using raw Level-2A values
        status_placeholder.info(f"Creating median composite from {count} images...")
        median_image = collection.median().select(S2_BANDS)
        
        # Create a directory for individual band downloads
        bands_dir = os.path.join(temp_dir, "bands")
        os.makedirs(bands_dir, exist_ok=True)
        
        # Manually download each band using Earth Engine's getDownloadURL
        status_placeholder.info("Downloading bands individually...")
        band_files = []
        
        # Get the region bounds for download
        region = ee_geometry.bounds().getInfo()['coordinates']
        
        for i, band in enumerate(S2_BANDS):
            try:
                status_placeholder.info(f"Downloading band {band} ({i+1}/{len(S2_BANDS)})...")
                
                # Create a filename for this band
                band_file = os.path.join(bands_dir, f"{band}.tif")
                
                # Get the download URL for this specific band
                url = median_image.select(band).getDownloadURL({
                    'scale': 10,  # 10m resolution
                    'region': region,
                    'format': 'GEO_TIFF',
                    'bands': [band]
                })
                
                # Download the file
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(band_file, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    band_files.append(band_file)
                    progress_placeholder.progress((i + 1) / len(S2_BANDS))
                else:
                    status_placeholder.error(f"Failed to download band {band}: HTTP status {response.status_code}")
            except Exception as e:
                status_placeholder.error(f"Error downloading band {band}: {str(e)}")
        
        # Check if we have downloaded all bands
        if len(band_files) == len(S2_BANDS):
            status_placeholder.info("All bands downloaded. Creating multiband GeoTIFF...")
            
            # Read the first band to get metadata
            with rasterio.open(band_files[0]) as src:
                meta = src.meta.copy()
            
            # Update metadata for multiband output
            meta.update(count=len(band_files))
            
            # Create the output file
            with rasterio.open(output_file, 'w', **meta) as dst:
                for i, band_file in enumerate(band_files):
                    with rasterio.open(band_file) as src:
                        dst.write(src.read(1), i+1)
            
            status_placeholder.success("Successfully created multiband GeoTIFF")
            return output_file
        else:
            status_placeholder.error(f"Only downloaded {len(band_files)}/{len(S2_BANDS)} bands")
            return None
        
    except Exception as e:
        status_placeholder.error(f"Error downloading Sentinel-2 data: {str(e)}")
        return None

# Helper function to normalize image data
def normalized(img):
    """Normalize image data to range [0, 1]"""
    min_val = np.nanmin(img)
    max_val = np.nanmax(img)
    
    if max_val == min_val:
        return np.zeros_like(img)
    
    img_norm = (img - min_val) / (max_val - min_val)
    return img_norm

# Function to determine UTM zone from longitude
def get_utm_zone(longitude):
    """Determine the UTM zone for a given longitude"""
    return math.floor((longitude + 180) / 6) + 1

# Function to determine UTM EPSG code from longitude and latitude
def get_utm_epsg(longitude, latitude):
    """Determine the EPSG code for UTM zone based on longitude and latitude"""
    zone_number = get_utm_zone(longitude)
    
    # Northern hemisphere if latitude >= 0, Southern hemisphere if latitude < 0
    if latitude >= 0:
        # Northern hemisphere EPSG: 326xx where xx is the UTM zone
        return f"EPSG:326{zone_number:02d}"
    else:
        # Southern hemisphere EPSG: 327xx where xx is the UTM zone
        return f"EPSG:327{zone_number:02d}"

# Function to convert coordinate system from WGS-84 to appropriate UTM Zone
def convert_to_utm(src_path, dst_path, polygon=None):
    """Reproject a GeoTIFF from WGS-84 to the appropriate UTM zone"""
    with rasterio.open(src_path) as src:
        # If polygon is provided, use its centroid to determine UTM zone
        if polygon:
            centroid = polygon.centroid
            lon, lat = centroid.x, centroid.y
            dst_crs = get_utm_epsg(lon, lat)
            st.info(f"Automatically detected UTM zone: {get_utm_zone(lon)} ({dst_crs})")
        else:
            # Use the center of the image to determine UTM zone
            bounds = src.bounds
            center_lon = (bounds.left + bounds.right) / 2
            center_lat = (bounds.bottom + bounds.top) / 2
            dst_crs = get_utm_epsg(center_lon, center_lat)
            st.info(f"Automatically detected UTM zone: {get_utm_zone(center_lon)} ({dst_crs})")
        
        # Calculate the transformation parameters
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        
        # Update the metadata
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        # Create the output file
        with rasterio.open(dst_path, 'w', **kwargs) as dst:
            # Reproject each band
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )
        
        return dst_path, dst_crs

@st.cache_resource
def load_water_model(model_path):
    """Load the water segmentation model"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        st.info(f"Using device: {device}")
        
        # Initialize your water segmentation model architecture here
        # Modify this based on your specific model architecture
        model = smp.UnetPlusPlus(
            encoder_name='efficientnet-b3',
            encoder_weights=None,
            in_channels=6,  # 6 bands for water detection (B2, B3, B4, B8, B11, B12)
            classes=2,
            activation=None
        ).to(device)
        
        # Load the model weights
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                st.info("Model loaded from checkpoint dictionary.")
            else:
                model.load_state_dict(checkpoint)
                st.info("Model loaded from state dictionary.")
        else:
            # If it's the model itself
            model = checkpoint
            st.info("Model loaded directly.")
        
        model.eval()
        st.session_state.model_loaded = True
        st.success("✅ Water segmentation model loaded successfully!")
        return model, device
    except Exception as e:
        st.error(f"❌ Error loading water model: {str(e)}")
        st.session_state.model_loaded = False
        return None, None

# Function to process image for water detection
def process_water_detection(image_path, selected_polygon, region_number):
    """Process image for water body detection"""
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    try:
        # 1. CLIPPING STEP
        status_placeholder.info("Step 1/4: Clipping image...")
        progress_placeholder.progress(0)
        
        # Open the raster file
        with rasterio.open(image_path) as src:
            # Check if the polygon overlaps with the raster bounds
            raster_bounds = box(*src.bounds)
            polygon_shapely = selected_polygon
            
            if not raster_bounds.intersects(polygon_shapely):
                status_placeholder.error("Error: The selected region doesn't overlap with the downloaded Sentinel-2 image.")
                return False
            
            # Convert polygon to GeoJSON format for rasterio
            geoms = [mapping(selected_polygon)]
            
            try:
                # Perform the clipping
                clipped_img, clipped_transform = mask(src, geoms, crop=True)
                
                # Check if the clipped image has valid data
                if clipped_img.size == 0 or np.all(clipped_img == 0):
                    status_placeholder.error("Error: Clipping resulted in an empty image.")
                    return False
                
            except ValueError as e:
                status_placeholder.error(f"Error during clipping: {str(e)}")
                return False
            
            # Store the clipped image metadata
            clipped_meta = src.meta.copy()
            clipped_meta.update({
                "height": clipped_img.shape[1],
                "width": clipped_img.shape[2],
                "transform": clipped_transform
            })
            
            # Save the clipped image to a temporary file
            temp_dir = os.path.dirname(image_path)
            os.makedirs(os.path.join(temp_dir, "temp"), exist_ok=True)
            temp_clipped_path = os.path.join(temp_dir, "temp", f"temp_clipped_region{region_number}.tif")
            
            with rasterio.open(temp_clipped_path, 'w', **clipped_meta) as dst:
                dst.write(clipped_img)
            
            # Convert to appropriate UTM Zone
            utm_clipped_path = os.path.join(temp_dir, "temp", f"utm_clipped_region{region_number}.tif")
            utm_path, utm_crs = convert_to_utm(temp_clipped_path, utm_clipped_path, selected_polygon)
            
            # Read back the UTM version
            with rasterio.open(utm_path) as src_utm:
                clipped_img = src_utm.read()
                clipped_meta = src_utm.meta.copy()
            
            # Extract only the bands needed for water detection (NO SCALING APPLIED)
            water_bands_img = clipped_img[WATER_BAND_INDICES]
            
            # Store in session state
            st.session_state.clipped_img = water_bands_img
            st.session_state.clipped_meta = clipped_meta
            st.session_state.region_number = region_number
            
            # Check if the clipped image is large enough
            if water_bands_img.shape[1] < 224 or water_bands_img.shape[2] < 224:
                status_placeholder.error(f"The clipped image is too small ({water_bands_img.shape[1]}x{water_bands_img.shape[2]} pixels). Please select a larger area (minimum 224x224 pixels).")
                return False
            
            # Display the clipped image (RGB visualization)
            rgb_bands = [2, 1, 0]  # B4, B3, B2 for RGB
            
            rgb = np.zeros((water_bands_img.shape[1], water_bands_img.shape[2], 3), dtype=np.float32)
            
            # Normalize and assign bands for visualization only
            for i, band in enumerate(rgb_bands):
                if band < water_bands_img.shape[0]:
                    band_data = water_bands_img[band]
                    # Simple contrast stretch for visualization
                    min_val = np.percentile(band_data, 2)
                    max_val = np.percentile(band_data, 98)
                    rgb[:, :, i] = np.clip((band_data - min_val) / (max_val - min_val), 0, 1)
            
            # Display the RGB image
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(rgb)
            ax.set_title(f"Clipped Sentinel-2 Image (Region {region_number})")
            ax.axis('off')
            st.pyplot(fig)
            
            progress_placeholder.progress(25)
            status_placeholder.success("Step 1/4: Clipping completed successfully")
            
            # 2. PATCHING STEP
            status_placeholder.info("Step 2/4: Creating patches...")
            
            # Prepare the image for patching (keep raw values, no normalization)
            img_for_patching = np.moveaxis(water_bands_img, 0, -1)  # Change to H x W x C format
            
            # Create patches
            patch_size = 224
            patches = patchify(img_for_patching, (patch_size, patch_size, water_bands_img.shape[0]), step=patch_size)
            patches = patches.squeeze()
            
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
                        dtype=patch_for_saving.dtype,  # Keep original dtype
                        crs=clipped_meta.get('crs'),
                        transform=clipped_meta.get('transform')
                    ) as dst:
                        for band_idx in range(patch_for_saving.shape[0]):
                            band_data = patch_for_saving[band_idx].reshape(patch_size, patch_size)
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
                'patch_size': patch_size
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
                        rgb_bands = [2, 1, 0]  # B4, B3, B2
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
                        patch = src.read()  # This will be in (C, H, W) format
                        patch_meta = src.meta.copy()
                    
                    # Create RGB version for display
                    rgb_bands = [2, 1, 0]  # B4, B3, B2
                    rgb_patch = np.zeros((patch.shape[1], patch.shape[2], 3), dtype=np.float32)
                    
                    for b, band in enumerate(rgb_bands):
                        if band < patch.shape[0]:
                            band_data = patch[band]
                            min_val = np.percentile(band_data, 2)
                            max_val = np.percentile(band_data, 98)
                            rgb_patch[:, :, b] = np.clip((band_data - min_val) / (max_val - min_val), 0, 1)
                    
                    # Convert to torch tensor (keep raw values for model)
                    img_tensor = torch.tensor(patch.astype(np.float32), dtype=torch.float32)
                    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
                    
                    # Perform water detection
                    with torch.inference_mode():
                        prediction = st.session_state.model(img_tensor.to(st.session_state.device))
                        prediction = torch.sigmoid(prediction).cpu()
                    
                    # Convert prediction to numpy
                    pred_np = prediction.squeeze().numpy()
                    
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
            
            # Store water detection paths in session state
            st.session_state.water_paths = water_paths
            st.session_state.water_shape = patches_shape
            
            # Display sample water detection results
            if water_results:
                st.subheader("Sample Water Detection Results")
                
                # Create a figure with subplots - 2 rows: originals and masks
                num_samples = len(water_results)
                fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
                
                # First row: Original Sentinel-2 RGB images
                for idx, result in enumerate(water_results):
                    axes[0, idx].imshow(result['rgb_original'])
                    axes[0, idx].set_title(f"Original {result['i']}_{result['j']}")
                    axes[0, idx].axis('off')
                
                # Second row: Water masks
                for idx, result in enumerate(water_results):
                    axes[1, idx].imshow(result['mask'], cmap='Blues')
                    axes[1, idx].set_title(f"Water Mask {result['i']}_{result['j']}")
                    axes[1, idx].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            progress_placeholder.progress(75)
            status_placeholder.success(f"Step 3/4: Detected water in {total_patches} patches successfully")
            
            # 4. RECONSTRUCTION STEP
            status_placeholder.info("Step 4/4: Reconstructing full water mask...")
            
            # Get water detection paths
            patches_info = st.session_state.patches_info
            water_paths = st.session_state.water_paths
            patches_shape = st.session_state.water_shape
            clipped_meta = st.session_state.clipped_meta
            
            # Load all water masks
            patches = []
            patch_indices = []
            
            for path in water_paths:
                # Extract i, j from filename
                filename = os.path.basename(path)
                parts = filename.split('_')
                i = int(parts[-2])
                j = int(parts[-1].split('.')[0])
                
                # Read the patch
                with rasterio.open(path) as src:
                    patch = src.read(1)  # Read first band
                    patches.append(patch)
                    patch_indices.append((i, j))
            
            # Determine grid dimensions
            i_vals = [idx[0] for idx in patch_indices]
            j_vals = [idx[1] for idx in patch_indices]
            max_i = max(i_vals) + 1
            max_j = max(j_vals) + 1
            
            # Create empty grid to hold patches
            patch_size = patches_info['patch_size']
            grid = np.zeros((max_i, max_j, patch_size, patch_size), dtype=np.uint8)
            
            # Fill the grid with patches
            for (i, j), patch in zip(patch_indices, patches):
                grid[i, j] = patch
            
            # Reconstruct the full image using unpatchify
            reconstructed_water_mask = unpatchify(grid, (max_i * patch_size, max_j * patch_size))
            
            # Create output filename
            base_dir = os.path.dirname(image_path)
            output_filename = f"water_mask_full_region{region_number}.tif"
            output_path = os.path.join(base_dir, output_filename)
            
            # Save as GeoTIFF
            out_meta = clipped_meta.copy()
            out_meta.update({
                'count': 1,
                'height': reconstructed_water_mask.shape[0],
                'width': reconstructed_water_mask.shape[1],
                'dtype': 'uint8'
            })
            
            with rasterio.open(output_path, 'w', **out_meta) as dst:
                dst.write(reconstructed_water_mask, 1)
            
            # Store the result in session state
            st.session_state.water_detection_result = reconstructed_water_mask
            st.session_state.water_detection_path = output_path
            
            # Display the reconstructed water mask
            col1, col2 = st.columns(2)
            
            with col1:
                # Display original RGB image
                fig, ax = plt.subplots(figsize=(10, 10))
                
                # Create RGB visualization from clipped image
                clipped_img = st.session_state.clipped_img
                rgb_bands = [2, 1, 0]  # B4, B3, B2
                rgb = np.zeros((clipped_img.shape[1], clipped_img.shape[2], 3), dtype=np.float32)
                
                for i, band in enumerate(rgb_bands):
                    if band < clipped_img.shape[0]:
                        band_data = clipped_img[band]
                        min_val = np.percentile(band_data, 2)
                        max_val = np.percentile(band_data, 98)
                        rgb[:, :, i] = np.clip((band_data - min_val) / (max_val - min_val), 0, 1)
                
                ax.imshow(rgb)
                ax.set_title("Original Sentinel-2 Image (RGB)")
                ax.axis('off')
                st.pyplot(fig)
            
            with col2:
                # Display water mask
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(reconstructed_water_mask, cmap='Blues')
                ax.set_title("Detected Water Bodies")
                ax.axis('off')
                st.pyplot(fig)
            
            # Calculate water statistics
            total_pixels = reconstructed_water_mask.size
            water_pixels = np.sum(reconstructed_water_mask > 0)
            water_percentage = (water_pixels / total_pixels) * 100
            
            # Estimate water area (assuming 10m resolution)
            water_area_m2 = water_pixels * 100  # 10m x 10m = 100 m²
            water_area_km2 = water_area_m2 / 1_000_000
            
            # Display statistics
            st.subheader("Water Detection Statistics")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Water Coverage", f"{water_percentage:.2f}%")
            
            with col2:
                st.metric("Water Area", f"{water_area_km2:.2f} km²")
            
            with col3:
                st.metric("Total Pixels", f"{water_pixels:,}")
            
            # Provide download link
            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download Water Mask (GeoTIFF)",
                    data=file,
                    file_name=output_filename,
                    mime="image/tiff"
                )
            
            progress_placeholder.progress(100)
            status_placeholder.success("All processing steps completed successfully!")
            
            return True
            
    except Exception as e:
        status_placeholder.error(f"Error in processing: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False

# First tab - Region Selection
with tab1:
    st.header("Select Region of Interest and Date")
    
    # Add warning about region size
    st.info("💡 For optimal results, select regions smaller than 40 sq km.")
    
    # Create a folium map centered at a default location
    m = folium.Map(location=[35.6892, 51.3890], zoom_start=10)  # Default to Tehran
    
    # Add drawing tools to the map
    draw = folium.plugins.Draw(
        export=True,
        position='topleft',
        draw_options={
            'polyline': False,
            'rectangle': True,
            'polygon': True,
            'circle': False,
            'marker': False,
            'circlemarker': False
        }
    )
    m.add_child(draw)
    
    # Use st_folium to capture the drawn shapes
    map_data = st_folium(m, width=800, height=500)
    
    # Process the drawn shapes from map_data
    if map_data is not None and 'last_active_drawing' in map_data and map_data['last_active_drawing'] is not None:
        drawn_shape = map_data['last_active_drawing']
        if 'geometry' in drawn_shape:
            geometry = drawn_shape['geometry']
            
            if geometry['type'] == 'Polygon':
                # Extract coordinates from the GeoJSON
                coords = geometry['coordinates'][0]  # First element contains the exterior ring
                polygon = Polygon(coords)
                
                # Store in session state
                st.session_state.last_drawn_polygon = polygon
                
                # Display the UTM zone for this polygon
                centroid = polygon.centroid
                utm_zone = get_utm_zone(centroid.x)
                utm_epsg = get_utm_epsg(centroid.x, centroid.y)
                
                # Calculate approximate area in square kilometers
                area_sq_km = polygon.area * 111 * 111  # Approximate conversion from degrees to km²
                
                st.success(f"Shape captured in UTM Zone {utm_zone} ({utm_epsg})! Area: ~{area_sq_km:.2f} km². Click 'Save Selected Region' to save it.")
                
                # Warn if area is large
                if area_sq_km > 40:
                    st.warning(f"Selected area is large ({area_sq_km:.2f} km²). Processing may take longer.")
    
    # Add a button to save the drawn polygons
    if st.button("Save Selected Region"):
        if 'last_drawn_polygon' in st.session_state:
            # Check if this polygon is already saved
            if not any(p.equals(st.session_state.last_drawn_polygon) for p in st.session_state.drawn_polygons):
                st.session_state.drawn_polygons.append(st.session_state.last_drawn_polygon)
                st.success(f"Region saved! Total regions: {len(st.session_state.drawn_polygons)}")
            else:
                st.info("This polygon is already saved.")
        else:
            st.warning("Please draw a polygon on the map first")
    
    # For demonstration purposes - keep the manual entry option
    with st.expander("Manually Enter Polygon Coordinates (For Testing)"):
        col1, col2 = st.columns(2)
        with col1:
            lat_input = st.text_input("Latitude coordinates (comma separated)", "35.68, 35.70, 35.69, 35.68")
        with col2:
            lon_input = st.text_input("Longitude coordinates (comma separated)", "51.38, 51.39, 51.40, 51.38")
        
        if st.button("Add Test Polygon"):
            try:
                lats = [float(x.strip()) for x in lat_input.split(",")]
                lons = [float(x.strip()) for x in lon_input.split(",")]
                if len(lats) == len(lons) and len(lats) >= 3:
                    coords = list(zip(lons, lats))  # GeoJSON format is [lon, lat]
                    test_polygon = Polygon(coords)
                    st.session_state.last_drawn_polygon = test_polygon
                    
                    # Display the UTM zone for this polygon
                    centroid = test_polygon.centroid
                    utm_zone = get_utm_zone(centroid.x)
                    utm_epsg = get_utm_epsg(centroid.x, centroid.y)
                    
                    # Calculate approximate area
                    area_sq_km = test_polygon.area * 111 * 111  # Approximate conversion from degrees to km²
                    
                    st.success(f"Test polygon created in UTM Zone {utm_zone} ({utm_epsg})! Area: ~{area_sq_km:.2f} km². Click 'Save Selected Region' to save it.")
                    
                    # Warn if area is large
                    if area_sq_km > 40:
                        st.warning(f"Selected area is large ({area_sq_km:.2f} km²). Processing may take longer.")
                else:
                    st.error("Please provide at least 3 coordinate pairs")
            except ValueError:
                st.error("Invalid coordinates. Please enter numeric values.")

    # Display saved regions with delete options
    if st.session_state.drawn_polygons:
        st.subheader("Saved Regions")
        
        # Display each region with a delete button
        for i, poly in enumerate(st.session_state.drawn_polygons):
            col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
            
            with col1:
                # Display region information
                st.write(f"Region {i+1}: {poly.wkt[:60]}...")
            
            with col2:
                # Display the UTM zone for this polygon
                centroid = poly.centroid
                utm_zone = get_utm_zone(centroid.x)
                utm_epsg = get_utm_epsg(centroid.x, centroid.y)
                st.write(f"UTM Zone: {utm_zone} ({utm_epsg})")
            
            with col3:
                # Display area
                area_sq_km = poly.area * 111 * 111  # Approximate conversion from degrees to km²
                st.write(f"Area: ~{area_sq_km:.2f} km²")
            
            with col4:
                # Add a delete button for each region
                if st.button("Delete", key=f"delete_region_{i}"):
                    st.session_state.drawn_polygons.pop(i)
                    st.rerun()
    
    # Date selection section
    st.subheader("Select Date for Analysis", divider="blue")
    
    # Create a beautiful card-like container for date selection
    date_selection_container = st.container()
    with date_selection_container:
        st.markdown("""
        <style>
        .date-selection-card {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .date-selection-header {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #1E88E5;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Initialize session state variables for date selection if they don't exist
        if 'selected_date' not in st.session_state:
            st.session_state.selected_date = datetime.date(2023, 6, 15)
        
        # Date selection
        st.markdown('<div class="date-selection-card">', unsafe_allow_html=True)
        st.markdown('<div class="date-selection-header">Select Date for Sentinel-2 Image</div>', unsafe_allow_html=True)
        
        selected_date = st.date_input(
            "Choose a date (±15 days will be used for image search)",
            value=st.session_state.selected_date,
            min_value=datetime.date(2015, 6, 23),  # Sentinel-2A launch date
            max_value=datetime.date.today(),
            key="date_selector"
        )
        st.session_state.selected_date = selected_date
        
        # Display the selected date range
        start_date = selected_date - datetime.timedelta(days=15)
        end_date = selected_date + datetime.timedelta(days=15)
        st.info(f"Search period: {start_date} to {end_date}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Cloud cover limit selection
    cloud_cover_limit = st.slider(
        "Maximum Cloud Cover (%)",
        min_value=0,
        max_value=100,
        value=20,
        step=5,
        help="Lower values give clearer images but may reduce available imagery"
    )
    
    # Download and process section
    st.subheader("Download and Process Sentinel-2 Data", divider="green")
    
    if st.session_state.drawn_polygons:
        # Region selection for processing
        region_options = [f"Region {i+1}" for i in range(len(st.session_state.drawn_polygons))]
        selected_region_idx = st.selectbox(
            "Select region to process:",
            options=range(len(region_options)),
            format_func=lambda x: region_options[x]
        )
        
        selected_polygon = st.session_state.drawn_polygons[selected_region_idx]
        
        # Display selected region info
        col1, col2, col3 = st.columns(3)
        with col1:
            centroid = selected_polygon.centroid
            st.metric("Center Longitude", f"{centroid.x:.4f}°")
        with col2:
            st.metric("Center Latitude", f"{centroid.y:.4f}°")
        with col3:
            area_sq_km = selected_polygon.area * 111 * 111
            st.metric("Area", f"{area_sq_km:.2f} km²")
        
        # Download button
        if st.button("🛰️ Download Sentinel-2 Data", type="primary"):
            date_str = selected_date.strftime('%Y-%m-%d')
            
            with st.spinner("Downloading Sentinel-2 data..."):
                downloaded_path = download_sentinel2_with_gees2(
                    date_str, 
                    selected_polygon, 
                    cloud_cover_limit
                )
            
            if downloaded_path:
                st.session_state.downloaded_image_path = downloaded_path
                st.session_state.selected_polygon = selected_polygon
                st.session_state.selected_region_idx = selected_region_idx
                st.success("✅ Sentinel-2 data downloaded successfully!")
                st.info("🔄 Go to the 'Water Body Detection' tab to analyze the image.")
            else:
                st.error("❌ Failed to download Sentinel-2 data. Try adjusting the date or cloud cover limit.")
        
    else:
        st.warning("Please draw and save at least one region on the map first.")

# Second tab - Water Body Detection
with tab2:
    st.header("Water Body Detection")
    
    # Check if we have downloaded data
    if 'downloaded_image_path' not in st.session_state:
        st.warning("⚠️ Please download Sentinel-2 data first in the 'Region Selection & Download' tab.")
        st.stop()
    
    # Display current data info
    with st.expander("Current Data Information", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"📅 **Date:** {st.session_state.selected_date}")
        
        with col2:
            region_num = st.session_state.selected_region_idx + 1
            st.info(f"🗺️ **Region:** {region_num}")
        
        with col3:
            polygon = st.session_state.selected_polygon
            area_sq_km = polygon.area * 111 * 111
            st.info(f"📏 **Area:** {area_sq_km:.2f} km²")
    
    # Model information
    st.subheader("Model Information", divider="blue")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Model:** Water Segmentation Model")
        st.info("**Architecture:** U-Net with ResNet34 encoder")
    with col2:
        st.info("**Input Bands:** B2, B3, B4, B8, B11, B12")
        st.info("**Input Size:** 224x224 pixels")
    
    # Processing parameters
    st.subheader("Processing Parameters", divider="green")
    
    col1, col2 = st.columns(2)
    
    with col1:
        threshold = st.slider(
            "Detection Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Threshold for water/no-water classification"
        )
    
    with col2:
        apply_morphology = st.checkbox(
            "Apply Morphological Operations",
            value=True,
            help="Apply erosion/dilation to clean up the water mask"
        )
        
        if apply_morphology:
            kernel_size = st.slider(
                "Morphological Kernel Size",
                min_value=1,
                max_value=7,
                value=3,
                step=2,
                help="Size of morphological operations kernel"
            )
    
    # Start processing button
    if st.button("🔍 Start Water Body Detection", type="primary"):
        # Load model
        with st.spinner("Loading model..."):
            model, device = load_water_model(model_path)
        
        if model is not None:
            st.session_state.model = model
            st.session_state.device = device
            st.session_state.model_loaded = True
            st.session_state.threshold = threshold
            
            # Start processing
            success = process_water_detection(
                st.session_state.downloaded_image_path,
                st.session_state.selected_polygon,
                st.session_state.selected_region_idx + 1
            )
            
            if success and apply_morphology:
                st.info("Applying morphological operations...")
                
                # Apply morphological operations to the result
                if 'water_detection_result' in st.session_state:
                    # Apply erosion followed by dilation (opening operation)
                    from scipy import ndimage
                    
                    # Convert to binary
                    binary_mask = (st.session_state.water_detection_result > 0).astype(np.uint8)
                    
                    # Create structuring element
                    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
                    
                    # Apply opening (erosion followed by dilation)
                    opened = ndimage.binary_opening(binary_mask, structure=kernel)
                    
                    # Apply closing (dilation followed by erosion)
                    cleaned_mask = ndimage.binary_closing(opened, structure=kernel).astype(np.uint8) * 255
                    
                    # Update the result
                    st.session_state.water_detection_result = cleaned_mask
                    
                    # Save the cleaned result
                    if 'water_detection_path' in st.session_state:
                        base_path = st.session_state.water_detection_path
                        cleaned_path = base_path.replace('.tif', '_cleaned.tif')
                        
                        # Read metadata from original
                        with rasterio.open(base_path) as src:
                            meta = src.meta.copy()
                        
                        # Save cleaned mask
                        with rasterio.open(cleaned_path, 'w', **meta) as dst:
                            dst.write(cleaned_mask, 1)
                        
                        st.session_state.water_detection_path = cleaned_path
                    
                    st.success("✅ Morphological operations applied successfully!")
            
        else:
            st.error("❌ Failed to load the water segmentation model. Please check the model path.")
    
    # Display results if available
    if 'water_detection_result' in st.session_state and st.session_state.water_detection_result is not None:
        st.subheader("Water Detection Results", divider="blue")
        
        # Create side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Original Sentinel-2 Image (RGB)**")
            
            # Create RGB visualization
            clipped_img = st.session_state.clipped_img
            rgb_bands = [2, 1, 0]  # B4, B3, B2 for RGB
            rgb = np.zeros((clipped_img.shape[1], clipped_img.shape[2], 3), dtype=np.float32)
            
            for i, band in enumerate(rgb_bands):
                if band < clipped_img.shape[0]:
                    band_data = clipped_img[band]
                    min_val = np.percentile(band_data, 2)
                    max_val = np.percentile(band_data, 98)
                    rgb[:, :, i] = np.clip((band_data - min_val) / (max_val - min_val), 0, 1)
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(rgb)
            ax.set_title("Original Image")
            ax.axis('off')
            st.pyplot(fig)
        
        with col2:
            st.write("**Detected Water Bodies**")
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(st.session_state.water_detection_result, cmap='Blues')
            ax.set_title("Water Detection Result")
            ax.axis('off')
            st.pyplot(fig)
        
        # Overlay visualization
        st.write("**Overlay: Water Bodies on Original Image**")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(rgb)
        
        # Create water overlay (semi-transparent blue)
        water_overlay = np.zeros((*st.session_state.water_detection_result.shape, 4))
        water_mask = st.session_state.water_detection_result > 0
        water_overlay[water_mask] = [0, 0.5, 1, 0.6]  # Semi-transparent blue
        
        ax.imshow(water_overlay)
        ax.set_title("Water Bodies Overlay")
        ax.axis('off')
        st.pyplot(fig)
        
        # Statistics
        st.subheader("Detection Statistics")
        
        total_pixels = st.session_state.water_detection_result.size
        water_pixels = np.sum(st.session_state.water_detection_result > 0)
        water_percentage = (water_pixels / total_pixels) * 100
        
        # Estimate water area (assuming 10m resolution)
        water_area_m2 = water_pixels * 100  # 10m x 10m = 100 m²
        water_area_km2 = water_area_m2 / 1_000_000
        water_area_hectares = water_area_m2 / 10_000
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Water Coverage", f"{water_percentage:.2f}%")
        
        with col2:
            st.metric("Water Area (km²)", f"{water_area_km2:.3f}")
        
        with col3:
            st.metric("Water Area (hectares)", f"{water_area_hectares:.1f}")
        
        with col4:
            st.metric("Water Pixels", f"{water_pixels:,}")
        
        # Download section
        st.subheader("Download Results")
        
        if 'water_detection_path' in st.session_state:
            col1, col2 = st.columns(2)
            
            with col1:
                # Download water mask
                with open(st.session_state.water_detection_path, "rb") as file:
                    st.download_button(
                        label="📥 Download Water Mask (GeoTIFF)",
                        data=file,
                        file_name=f"water_mask_{st.session_state.selected_date}.tif",
                        mime="image/tiff"
                    )
            
            with col2:
                # Create and download a summary report
                report_data = {
                    "date": str(st.session_state.selected_date),
                    "region": f"Region {st.session_state.selected_region_idx + 1}",
                    "total_area_km2": float(st.session_state.selected_polygon.area * 111 * 111),
                    "water_coverage_percent": float(water_percentage),
                    "water_area_km2": float(water_area_km2),
                    "water_area_hectares": float(water_area_hectares),
                    "total_pixels": int(total_pixels),
                    "water_pixels": int(water_pixels),
                    "model": "U-Net with ResNet34",
                    "threshold": float(threshold) if 'threshold' in st.session_state else 0.5
                }
                
                report_json = json.dumps(report_data, indent=2)
                
                st.download_button(
                    label="📊 Download Analysis Report (JSON)",
                    data=report_json,
                    file_name=f"water_analysis_report_{st.session_state.selected_date}.json",
                    mime="application/json"
                )
        
        # Additional analysis tools
        with st.expander("🔧 Advanced Analysis Tools"):
            st.write("**Threshold Analysis**")
            
            if st.button("Analyze Different Thresholds"):
                thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
                threshold_results = []
                
                for thresh in thresholds:
                    # Apply threshold to the raw prediction (if available)
                    # For now, we'll simulate this with the current result
                    temp_mask = st.session_state.water_detection_result > (thresh * 255)
                    temp_water_pixels = np.sum(temp_mask)
                    temp_percentage = (temp_water_pixels / total_pixels) * 100
                    temp_area_km2 = (temp_water_pixels * 100) / 1_000_000
                    
                    threshold_results.append({
                        'threshold': thresh,
                        'water_percentage': temp_percentage,
                        'water_area_km2': temp_area_km2
                    })
                
                # Create a plot
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                thresholds_list = [r['threshold'] for r in threshold_results]
                percentages = [r['water_percentage'] for r in threshold_results]
                areas = [r['water_area_km2'] for r in threshold_results]
                
                ax1.plot(thresholds_list, percentages, 'b-o')
                ax1.set_xlabel('Threshold')
                ax1.set_ylabel('Water Coverage (%)')
                ax1.set_title('Water Coverage vs Threshold')
                ax1.grid(True)
                
                ax2.plot(thresholds_list, areas, 'r-o')
                ax2.set_xlabel('Threshold')
                ax2.set_ylabel('Water Area (km²)')
                ax2.set_title('Water Area vs Threshold')
                ax2.grid(True)
                
                plt.tight_layout()
                st.pyplot(fig)
