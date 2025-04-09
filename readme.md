



Old app.py

# streamlit_dashboard/app.py

import streamlit as st
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import traceback
import io

# Add parent directory to path to find modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Explicitly import the utils module
from utils import predict_and_visualize, load_model

# Page configuration
st.set_page_config(
    page_title="RoadNet: Satellite Road Segmentation",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def cached_load_model(model_path="saved_models/roadnet_unet_weights.pth", use_checkpoint=False):
    """
    Load model with caching, supporting both direct weights and checkpoints
    """
    # Check if PyTorch is available
    if 'torch' not in sys.modules:
        st.error("❌ PyTorch is not properly installed or imported.")
        st.stop()
    
    # Handle checkpoint path
    if use_checkpoint:
        checkpoint_path = os.path.join(parent_dir, "best_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            st.info(f"Loading from checkpoint: {checkpoint_path}")
            return load_model(checkpoint_path)
        else:
            st.warning(f"Checkpoint not found at {checkpoint_path}")
    
    # Check various potential paths for the model file
    potential_paths = [
        model_path,
        os.path.join(parent_dir, model_path),
        os.path.join(os.path.dirname(__file__), model_path)
    ]
    
    # Try each path
    for path in potential_paths:
        if os.path.exists(path):
            st.info(f"Found model at: {path}")
            try:
                model = load_model(path)
                if model is not None:
                    return model
            except Exception as e:
                st.error(f"Error loading model from {path}: {str(e)}")
                continue
    
    # If we get here, no model was loaded successfully
    # Final fallback to checkpoint if we haven't tried it yet
    if not use_checkpoint:
        checkpoint_path = os.path.join(parent_dir, "best_checkpoint.pth")
        if os.path.exists(checkpoint_path):
            st.info(f"Trying fallback checkpoint at: {checkpoint_path}")
            try:
                model = load_model(checkpoint_path)
                if model is not None:
                    return model
            except Exception as e:
                st.error(f"Error loading fallback checkpoint: {str(e)}")

    # No model could be loaded
    st.error("❌ Failed to load model from any location. Please ensure model files exist.")
    available_files = []
    for path in potential_paths + [os.path.join(parent_dir, "best_checkpoint.pth")]:
        if os.path.exists(path):
            available_files.append(path)
    if available_files:
        st.info(f"Available model files: {', '.join(available_files)}")
    st.stop()

def main():
    # App header
    st.title("🛣️ RoadNet - Satellite Road Segmentation")
    st.markdown("""
    ### Detect roads in satellite imagery using a UNet++ model
    Upload a satellite image and the model will identify and highlight road networks.
    """)

    # Sidebar content
    with st.sidebar:
        st.header("Model Settings")
        threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5, 0.01,
                             help="Adjust the confidence threshold for road detection")
        
        # Add model selection options
        use_checkpoint = st.checkbox("Use checkpoint file instead of weights", value=False,
                                   help="Use best_checkpoint.pth instead of saved model weights")
        
        st.info("Higher threshold = more precision, less recall")

        st.header("About")
        st.markdown("""
        This app uses a UNet++ model with EfficientNet-B4 backbone trained to 
        detect road networks in satellite imagery.
        """)
        
        # Add hardware info
        st.header("System Info")
        device_info = f"Using: {'GPU' if torch.cuda.is_available() else 'CPU'}"
        if torch.cuda.is_available():
            device_info += f" ({torch.cuda.get_device_name(0)})"
        st.info(device_info)

    # Load model with appropriate path
    with st.spinner("Loading model..."):
        try:
            model = cached_load_model(use_checkpoint=use_checkpoint)
            st.success("✅ Model loaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to load model: {str(e)}")
            st.exception(e)
            st.stop()

    # File uploader
    uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png"])

    # Process image if uploaded
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            
            # Display the uploaded image
            st.subheader("Uploaded Image")
            st.image(image, width=600)

            # Button to trigger prediction
            if st.button("Run Road Detection", type="primary"):
                with st.spinner("Analyzing satellite image..."):
                    try:
                        results = predict_and_visualize(model, image, threshold)

                        # Display results
                        st.subheader("Road Detection Results")
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown("**Confidence Map**")
                            fig, ax = plt.subplots(figsize=(5, 5))
                            im = ax.imshow(results['prediction_confidence'], cmap='viridis')
                            plt.colorbar(im, ax=ax)
                            ax.set_title("Road Confidence")
                            ax.axis('off')
                            st.pyplot(fig)

                        with col2:
                            st.markdown("**Binary Road Mask**")
                            st.image(results['binary_mask'] * 255, width=300)

                        with col3:
                            st.markdown("**Road Overlay**")
                            st.image(results['overlay'], width=300)

                        st.subheader("Final Road Detection Result")
                        st.image(results['overlay'], caption="Red areas indicate detected roads")

                        # Additional stats
                        confidence = results['prediction_confidence']
                        st.markdown(f"""
                        **Detection Statistics:**
                        - Mean Confidence: {confidence.mean():.4f}
                        - Max Confidence: {confidence.max():.4f}
                        - Road Coverage: {(results['binary_mask'].sum() / results['binary_mask'].size) * 100:.2f}% of image area
                        """)

                        # Download result
                        result_image = Image.fromarray(results['overlay'])
                        buf = io.BytesIO()
                        result_image.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="Download Result",
                            data=byte_im,
                            file_name="road_detection_result.png",
                            mime="image/png"
                        )
                    except Exception as e:
                        st.error(f"Error processing image: {str(e)}")
                        st.exception(e)
                        st.info("Try adjusting the threshold or using a different model loading option.")
        except Exception as e:
            st.error(f"Error opening image: {str(e)}")
            st.exception(e)

if __name__ == "__main__":
    main()


old utils.py

# streamlit-dashboard/utils.py

# Ensure imports are at the top level
import torch
import numpy as np
from PIL import Image
import os
import sys
import segmentation_models_pytorch as smp
import torch.serialization
try:
    from numpy.core.multiarray import scalar
except ImportError:
    scalar = None

def prepare_image(pil_image):
    """
    Resize, normalize image, return torch tensor and original numpy.
    """
    image_np = np.array(pil_image.resize((512, 512)))  # Resize
    image = image_np.astype(np.float32) / 255.0
    image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()
    return image, image_np

def load_model(weights_path="saved_models/roadnet_unet_weights.pth"):
    """
    Loads Unet model and weights from path.
    """
    try:
        # Import torch again to ensure it's available in this scope
        import torch
        
        # Clear CUDA memory cache and check if CUDA is available
        torch.cuda.empty_cache()
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        # Initialize the model architecture
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Make sure the file exists
        if not os.path.exists(weights_path):
            print(f"Warning: Weights file not found at {weights_path}")
            # Try with a relative path adjustment
            if "../" not in weights_path and os.path.exists(f"../{weights_path}"):
                weights_path = f"../{weights_path}"
                print(f"Found weights at {weights_path}")
        
        # Add scalar to safe globals first to avoid potential issues
        if scalar is not None:
            torch.serialization.add_safe_globals([scalar])
        
        # Try various loading methods for compatibility
        try:
            # First try the simplest approach with map_location
            checkpoint = torch.load(weights_path, map_location=device)
            print(f"Loaded checkpoint from {weights_path}")
            
            # If checkpoint is a dict with model_state_dict
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model checkpoint from epoch {checkpoint.get('epoch', 'unknown')} with loss {checkpoint.get('best_loss', 'unknown')}")
            else:
                # Direct state dict
                model.load_state_dict(checkpoint)
                print("Loaded model state dict directly")
                
        except Exception as e:
            print(f"Standard loading failed: {e}")
            try:
                # For PyTorch 2.6+ try with weights_only parameter
                checkpoint = torch.load(weights_path, weights_only=False, map_location=device)
                print("Loaded checkpoint with weights_only=False")
                
                # Handle the state dict
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded model checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    model.load_state_dict(checkpoint)
                    print("Loaded model state dict directly with weights_only=False")
            except Exception as final_error:
                print(f"All loading attempts failed. Final error: {final_error}")
                return None
        
        model.eval()
        model.to(device)
        return model
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None  # catch silent errors in Streamlit

def predict_and_visualize(model, pil_image, threshold=0.5):
    """
    Predict and overlay road network mask.
    """
    if model is None:
        raise ValueError("Model is not loaded properly (NoneType).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_tensor, image_np = prepare_image(pil_image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        # Handle different model output formats
        output = model(image_tensor)
        if isinstance(output, dict) and 'out' in output:
            output = output['out']
        prob = torch.sigmoid(output).squeeze().cpu().numpy()

    binary_mask = (prob > threshold).astype(np.uint8)
    overlay = image_np.copy()
    overlay[binary_mask == 1] = [255, 0, 0]
    blended = (0.6 * image_np + 0.4 * overlay).astype(np.uint8)

    return {
        'original': image_np,
        'prediction_confidence': prob,
        'binary_mask': binary_mask,
        'overlay': blended
    }
