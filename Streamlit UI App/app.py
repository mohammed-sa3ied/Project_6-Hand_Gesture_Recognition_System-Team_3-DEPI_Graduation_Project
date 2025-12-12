import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


st.set_page_config(page_title="Guesture  ", page_icon="🖐️", layout="wide")

IMG_SIZE = 128
LABELS_MAP = {
    0: '01_palm', 1: '02_l', 2: '03_fist', 3: '04_fist_moved',
    4: '05_thumb', 5: '06_index', 6: '07_ok', 7: '08_palm_moved',
    8: '09_c', 9: '10_down'
}


# 2. LOAD MODEL

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('hand_gesture_model.keras')
        return model
    except:
        return None

model = load_model()


# 3. PROCESSING LOGIC

def get_action(gesture_name):
    """Maps gesture to system action"""
    if "fist" in gesture_name: return "DRAG / GRAB", "error"
    if "palm" in gesture_name: return "STOP / PAUSE", "warning"
    if "ok" in gesture_name: return "CONFIRM SELECTION", "success"
    if "index" in gesture_name: return "MOVE POINTER", "info"
    if "down" in gesture_name: return "SCROLL DOWN", "primary"
    if "l" in gesture_name: return "GO BACK", "secondary"
    if "c" in gesture_name: return "COPY TEXT", "success"
    return "WAITING...", "secondary"

# ==========================================
# 4. STREAMLIT UI
# ==========================================-
st.sidebar.header("⚙️ Sensitivity Control")
st.sidebar.info("Adjust these sliders until your hand is WHITE and background is BLACK in the 'Mask' image.")


min_value = st.sidebar.slider("Min Skin Threshold", 0, 255, 133)
max_value = st.sidebar.slider("Max Skin Threshold", 0, 255, 173)

st.title("  Hand Gesture Recognition")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader(" Camera Input")
    camera_image = st.camera_input("Click to Capture")

with col2:
    st.subheader(" AI Processing (The Mask)")
    
    if camera_image:
        
        image = Image.open(camera_image)
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) # Convert to BGR
        
        
        img_YCrCb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
        
        lower_skin = np.array([0, min_value, 77], dtype=np.uint8)
        upper_skin = np.array([255, max_value, 127], dtype=np.uint8)
        
       
        skin_mask = cv2.inRange(img_YCrCb, lower_skin, upper_skin)
        
        
        kernel = np.ones((5,5), np.uint8)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
        

        st.image(skin_mask, caption="What the Model Sees", width=300)
        
        resized_mask = cv2.resize(skin_mask, (IMG_SIZE, IMG_SIZE))
        
       
        img_normal = resized_mask
        img_flipped = cv2.flip(resized_mask, 1)
        
        input_normal = img_normal.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
        input_flipped = img_flipped.reshape(1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
        
        if model:
            
            pred_normal = model.predict(input_normal, verbose=0)
            pred_flipped = model.predict(input_flipped, verbose=0)
          
            if np.max(pred_flipped) > np.max(pred_normal):
                final_pred = pred_flipped
                status = "Auto-Flipped (Right Hand)"
            else:
                final_pred = pred_normal
                status = "Normal (Left Hand)"
            
            idx = np.argmax(final_pred)
            conf = np.max(final_pred)
            gesture_name = LABELS_MAP.get(idx, "Unknown")
           
            action_text, msg_type = get_action(gesture_name)
            
            st.markdown("---")
            st.metric("Prediction", f"{gesture_name}", f"{conf*100:.1f}% Confidence")
            
          
            if msg_type == "error": st.error(f"ACTION: {action_text}")
            elif msg_type == "warning": st.warning(f"ACTION: {action_text}")
            elif msg_type == "success": st.success(f"ACTION: {action_text}")
            elif msg_type == "info": st.info(f"ACTION: {action_text}")
            else: st.write(f"ACTION: {action_text}")
            
    else:
        st.info("Waiting for snapshot...")

if not model:
    st.sidebar.error(" Model 'hand_gesture_model.keras' not found!")
else:
    st.sidebar.success(" Model Loaded")