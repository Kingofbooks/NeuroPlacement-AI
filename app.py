import whisper
import streamlit as st
import numpy as np
import pickle
import re
import pandas as pd
from streamlit_mic_recorder import mic_recorder
from difflib import SequenceMatcher

# ==================== ACTIVATION FUNCTIONS ====================

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def relu(z):
    """ReLU activation function"""
    return np.maximum(z, 0)

def leaky_relu(z):
    """Leaky ReLU activation function"""
    return np.maximum(0.01 * z, z)

# ==================== NEURAL NETWORK ====================

def forward_feed(x, weights, biases, activation_type):
    """Forward pass through neural network"""
    a = x
    activations = [x]
    
    for i in range(len(weights) - 1):
        z = np.dot(weights[i], a) + biases[i]
        
        if activation_type == "relu":
            a = relu(z)
        elif activation_type == "leaky_relu":
            a = leaky_relu(z)
        elif activation_type == "sigmoid":
            a = sigmoid(z)
        
        activations.append(a)
    
    z_final = weights[-1] @ a + biases[-1]
    y_pred = sigmoid(z_final)
    activations.append(y_pred)
    
    return y_pred, activations

# ==================== MODEL LOADING (CACHED) ====================

@st.cache_resource
def load_whisper_model():
    """Load Whisper model (cached for performance)"""
    return whisper.load_model("base")

@st.cache_resource
def load_placement_model():
    """Load placement prediction model (cached for performance)"""
    try:
        with open("model/placement_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Error: model/placement_model.pkl not found!")
        return None

# ==================== HELPER FUNCTIONS ====================

def word_to_number(word):
    """Convert written numbers to digits"""
    word_map = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
        "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20
    }
    word_lower = word.lower().strip()
    return word_map.get(word_lower, None)

def fuzzy_match(text, target, threshold=0.6):
    """Fuzzy match to handle typos and variations"""
    words = text.lower().split()
    for word in words:
        ratio = SequenceMatcher(None, word, target.lower()).ratio()
        if ratio >= threshold:
            return True
    return False

def extract_number_from_match(match_str):
    """Extract numeric value from matched string"""
    if not match_str:
        return None
    
    # Try to convert written numbers
    num = word_to_number(match_str)
    if num is not None:
        return float(num)
    
    # Try to extract digits
    digits = re.search(r'(\d+(?:[.,]\d+)?)', match_str)
    if digits:
        return float(digits.group(1).replace(',', '.'))
    
    return None

# ==================== EXTRACTION FUNCTION ====================

def extract_features(text):
    """Extract features using fuzzy keyword detection + intelligent number extraction"""
    text_lower = text.lower()
    data = {}
    
    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
        "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20"
    }
    text_normalized = text_lower
    for word, digit in number_words.items():
        text_normalized = re.sub(r'\b' + word + r'\b', digit, text_normalized)
    
    def find_nearby_number(text, keywords, window=20):
        """Find a number near given keywords using fuzzy matching"""
        for keyword in keywords:
            idx = text.find(keyword)
            if idx != -1:
                start = max(0, idx - window)
                end = min(len(text), idx + len(keyword) + window)
                window_text = text[start:end]
                numbers = re.findall(r'(\d+(?:[.,]\d+)?)', window_text)
                if numbers:
                    return float(numbers[-1].replace(',', '.'))
            
            for i in range(len(text) - len(keyword)):
                substring = text[i:i+len(keyword)]
                ratio = SequenceMatcher(None, substring, keyword).ratio()
                if ratio > 0.7:  
                    start = max(0, i - window)
                    end = min(len(text), i + len(keyword) + window)
                    window_text = text[start:end]
                    numbers = re.findall(r'(\d+(?:[.,]\d+)?)', window_text)
                    if numbers:
                        return float(numbers[-1].replace(',', '.'))
        return None
    
    data["cgpa"] = find_nearby_number(text_normalized, ["cgpa", "chgps", "cgp"], window=15)
    
    data["internships"] = find_nearby_number(text_normalized, ["internship", "intern", "interns"], window=20)
    
    data["projects"] = find_nearby_number(text_normalized, ["project", "projects"], window=20)
    
    data["workshops"] = find_nearby_number(text_normalized, ["workshop", "certificate", "certification"], window=20)
    
    data["aptitude"] = find_nearby_number(text_normalized, ["aptitude", "apt", "attitude"], window=15)
    
    data["skills"] = find_nearby_number(text_normalized, ["soft skill", "skills rating", "skill rating", "skill"], window=15)
    
    extracurricular_keywords = ["extracurricular", "extra curricular", "extracirricular"]
    extracurricular_no = ["no extra", "no extracurricular", "no activities"]
    has_ext = any(kw in text_normalized for kw in extracurricular_keywords)
    has_no_ext = any(kw in text_normalized for kw in extracurricular_no)
    data["extracurricular"] = 0 if has_no_ext else (1 if has_ext else 0)
    
    training_keywords = ["placement training", "training completed", "completed training"]
    training_no = ["no training", "not training"]
    has_training = any(kw in text_normalized for kw in training_keywords)
    has_no_training = any(kw in text_normalized for kw in training_no)
    data["training"] = 0 if has_no_training else (1 if has_training else 0)
    
    data["ssc"] = find_nearby_number(text_normalized, ["ssc", "s.s.c", "s s c"], window=15)
    
    data["hsc"] = find_nearby_number(text_normalized, ["hsc", "h.s.c", "h s c", "intermediate"], window=15)
    
    return data

# ==================== PREDICTION FUNCTION ====================

def make_prediction(features, model_data):
    """Make placement prediction from features"""
    required_features = ["cgpa", "internships", "projects", "workshops", "aptitude", 
                        "skills", "extracurricular", "training", "ssc", "hsc"]
    
    for feature in required_features:
        if features.get(feature) is None:
            features[feature] = 0
    
    missing = [f for f in required_features if features.get(f) == 0]
    if missing:
        st.info(f"Using default value (0) for: {', '.join(missing)}")
    
    input_df = pd.DataFrame({
        "CGPA": [features["cgpa"]],
        "Internships": [features["internships"]],
        "Projects": [features["projects"]],
        "Workshops/Certifications": [features["workshops"]],
        "AptitudeTestScore": [features["aptitude"]],
        "SoftSkillsRating": [features["skills"]],
        "SSC_Marks": [features["ssc"]],
        "HSC_Marks": [features["hsc"]],
        "ExtracurricularActivities_Yes": [1 if features["extracurricular"] == 1 else 0],
        "PlacementTraining_Yes": [1 if features["training"] == 1 else 0],
    })
    
    input_scaled = model_data["scaler"].transform(input_df).T
    prediction, _ = forward_feed(
        input_scaled,
        model_data["weights"],
        model_data["biases"],
        "leaky_relu"
    )
    
    return prediction.item() * 100

# ==================== STREAMLIT UI ====================

st.set_page_config(page_title="NeuroPlacement AI", page_icon="🎙️")
st.title("🎙️ NeuroPlacement AI")
st.markdown("---")

whisper_model = load_whisper_model()
placement_model = load_placement_model()

if placement_model is None:
    st.stop()

tab1, tab2, tab3 = st.tabs(["🎤 Voice Input", "📋 Manual Entry", "🧪 Model Testing"])

# ==================== TAB 1: VOICE INPUT ====================

with tab1:
    st.subheader("📝 Record Your Information")
    st.info("""
    **📢 Example of what you can say:**
    
    *"My CGPA is 8.5, I have completed 2 internships, worked on 3 projects, attended 2 workshops, aptitude score 85, soft skills rating 8, yes I have done extracurricular activities, yes I attended placement training, SSC percentage 90, and HSC percentage 92"*
    
    Or simply mention each field:
    - "CGPA: 8.5"
    - "2 internships"
    - "3 projects"
    - "2 certifications"
    - "aptitude 85"
    - "skills rating 8"
    - "extracurricular yes"
    - "placement training yes"
    - "SSC 90"
    - "HSC 92"
    """)
    
    st.markdown("---")
    st.subheader("🎤 Recording Controls")
    st.info("👉 **Click START, speak your information, then click STOP when done.**")
    
    audio = mic_recorder(
        start_prompt="🎤 Start Recording",
        stop_prompt="⏹️ Stop Recording",
        just_once=False,
    )

    if audio:
        with open("temp.wav", "wb") as f:
            f.write(audio["bytes"])
        
        st.audio(audio["bytes"], format="audio/wav")
        
        with st.spinner("🔄 Transcribing audio..."):
            result = whisper_model.transcribe("temp.wav")
            text = result["text"].lower()
        
        st.success("✅ Transcription complete!")
        st.write("### 📋 Transcribed Text")
        st.write(f"*{text}*")
        
        features = extract_features(text)
        st.write("### 📊 Extracted Features")
        st.json(features)
        
        with st.spinner("🤖 Analyzing placement probability..."):
            probability = make_prediction(features, placement_model)
        
        if probability is not None:
            st.markdown("---")
            if probability >= 70:
                st.success(f"✅ Placement Probability: **{probability:.2f}%** (Strong chance!)")
            elif probability >= 50:
                st.info(f"📈 Placement Probability: **{probability:.2f}%** (Moderate chance)")
            else:
                st.warning(f"⚠️ Placement Probability: **{probability:.2f}%** (Low probability)")

# ==================== TAB 2: MANUAL ENTRY ====================

with tab2:
    st.subheader("✍️ Enter Your Information Manually")
    
    with st.form("placement_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cgpa = st.number_input(
                "CGPA",
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                value=7.5,
                help="Your GPA (0.0 - 10.0)"
            )
        
        with col2:
            internships = st.number_input(
                "Internships",
                min_value=0,
                max_value=10,
                step=1,
                value=2,
                help="Number of internships completed"
            )
        
        with col3:
            projects = st.number_input(
                "Projects",
                min_value=0,
                max_value=20,
                step=1,
                value=3,
                help="Number of projects completed"
            )
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            workshops = st.number_input(
                "Workshops/Certifications",
                min_value=0,
                max_value=10,
                step=1,
                value=2,
                help="Number of workshops or certifications"
            )
        
        with col5:
            aptitude = st.number_input(
                "Aptitude Score",
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                value=75.0,
                help="Aptitude test score (0-100)"
            )
        
        with col6:
            skills = st.number_input(
                "Soft Skills Rating",
                min_value=0.0,
                max_value=10.0,
                step=0.1,
                value=7.5,
                help="Soft skills rating (0-10)"
            )
        
        col7, col8, col9 = st.columns(3)
        
        with col7:
            extracurricular = st.selectbox(
                "Extracurricular Activities",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                help="Participated in extracurricular activities?"
            )
        
        with col8:
            training = st.selectbox(
                "Placement Training",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                help="Attended placement training?"
            )
        
        with col9:
            ssc = st.number_input(
                "SSC Percentage",
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                value=85.0,
                help="SSC board exam percentage (0-100)"
            )
        
        col10 = st.columns(1)[0]
        with col10:
            hsc = st.number_input(
                "HSC Percentage",
                min_value=0.0,
                max_value=100.0,
                step=0.5,
                value=88.0,
                help="HSC board exam percentage (0-100)"
            )
        
        submitted = st.form_submit_button("🚀 Predict Placement", use_container_width=True)
    
    if submitted:
        features = {
            "cgpa": cgpa,
            "internships": int(internships),
            "projects": int(projects),
            "workshops": int(workshops),
            "aptitude": aptitude,
            "skills": skills,
            "extracurricular": extracurricular,
            "training": training,
            "ssc": ssc,
            "hsc": hsc
        }
        
        st.write("### 📊 Your Information")
        st.json(features)
        
        with st.spinner("🤖 Analyzing placement probability..."):
            probability = make_prediction(features, placement_model)
        
        if probability is not None:
            st.markdown("---")
            if probability >= 70:
                st.success(f"✅ Placement Probability: **{probability:.2f}%** (Strong chance!)")
            elif probability >= 50:
                st.info(f"📈 Placement Probability: **{probability:.2f}%** (Moderate chance)")
            else:
                st.warning(f"⚠️ Placement Probability: **{probability:.2f}%** (Low probability)")

# ==================== TAB 3: MODEL TESTING ====================

with tab3:
    st.subheader("🧪 Model Testing & Validation")
    st.markdown("Test different candidate profiles to evaluate model performance")
    
    # Define all test cases
    test_cases = {
        "TEST 1: Very Strong Candidate": {
            "description": "Expected: VERY HIGH probability",
            "data": {
                "cgpa": 9.1,
                "internships": 2,
                "projects": 3,
                "workshops": 3,
                "aptitude": 90,
                "skills": 4.8,
                "ssc": 90,
                "hsc": 88,
                "extracurricular": 1,
                "training": 1,
            }
        },
        "TEST 2: Weak Candidate": {
            "description": "Expected: LOW probability",
            "data": {
                "cgpa": 6.5,
                "internships": 0,
                "projects": 0,
                "workshops": 0,
                "aptitude": 60,
                "skills": 3.0,
                "ssc": 55,
                "hsc": 57,
                "extracurricular": 0,
                "training": 0,
            }
        },
        "TEST 3: Strong Academics, No Skills": {
            "description": "Expected: Moderate prediction - Tests feature balance",
            "data": {
                "cgpa": 8.8,
                "internships": 0,
                "projects": 0,
                "workshops": 0,
                "aptitude": 65,
                "skills": 3.2,
                "ssc": 88,
                "hsc": 85,
                "extracurricular": 0,
                "training": 0,
            }
        },
        "TEST 4: Mid CGPA, Strong Skills": {
            "description": "Expected: Surprisingly decent prediction",
            "data": {
                "cgpa": 7.0,
                "internships": 2,
                "projects": 3,
                "workshops": 3,
                "aptitude": 88,
                "skills": 4.7,
                "ssc": 70,
                "hsc": 72,
                "extracurricular": 1,
                "training": 1,
            }
        },
        "TEST 5: Excellent Communication Only": {
            "description": "Expected: Tests if model overfocuses on soft skills",
            "data": {
                "cgpa": 7.2,
                "internships": 0,
                "projects": 1,
                "workshops": 0,
                "aptitude": 68,
                "skills": 4.8,
                "ssc": 65,
                "hsc": 67,
                "extracurricular": 1,
                "training": 0,
            }
        },
        "TEST 6: High Aptitude, Low Everything Else": {
            "description": "Expected: Tests feature sensitivity",
            "data": {
                "cgpa": 6.8,
                "internships": 0,
                "projects": 0,
                "workshops": 0,
                "aptitude": 90,
                "skills": 3.1,
                "ssc": 60,
                "hsc": 61,
                "extracurricular": 0,
                "training": 0,
            }
        },
        "TEST 7: Extreme Input": {
            "description": "Expected: Model should not explode (tests robustness)",
            "data": {
                "cgpa": 10.0,
                "internships": 5,
                "projects": 10,
                "workshops": 10,
                "aptitude": 100,
                "skills": 5.0,
                "ssc": 100,
                "hsc": 100,
                "extracurricular": 1,
                "training": 1,
            }
        },
        "TEST 8: All Average": {
            "description": "Expected: Medium probability",
            "data": {
                "cgpa": 7.5,
                "internships": 1,
                "projects": 2,
                "workshops": 1,
                "aptitude": 75,
                "skills": 4.0,
                "ssc": 70,
                "hsc": 72,
                "extracurricular": 0,
                "training": 1,
            }
        }
    }
    
    if st.button("▶️ RUN ALL TESTS", use_container_width=True, key="run_all_tests"):
        st.markdown("---")
        results = []
        
        for test_name, test_info in test_cases.items():
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader(test_name)
                st.caption(test_info["description"])
            
            probability = make_prediction(test_info["data"], placement_model)
            
            if probability is not None:
                with col2:
                    if probability >= 70:
                        st.success(f"**{probability:.2f}%**")
                    elif probability >= 50:
                        st.info(f"**{probability:.2f}%**")
                    else:
                        st.warning(f"**{probability:.2f}%**")
                
                results.append({
                    "Test": test_name,
                    "Probability (%)": f"{probability:.2f}",
                    "Status": "✅ Strong" if probability >= 70 else ("⚠️ Moderate" if probability >= 50 else "❌ Low")
                })
            
            with st.expander(f"📊 View Details for {test_name}"):
                st.json(test_info["data"])
            
            st.divider()
        
        st.subheader("📋 Test Summary")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
    
    st.markdown("---")
    st.subheader("🔄 TEST 9: Training Impact Test")
    st.caption("Compare predictions WITH and WITHOUT Placement Training for the same candidate")
    
    if st.button("🧪 Run Training Impact Test", use_container_width=True, key="training_test"):
        base_profile = {
            "cgpa": 7.5,
            "internships": 1,
            "projects": 2,
            "workshops": 1,
            "aptitude": 75,
            "skills": 4.0,
            "ssc": 70,
            "hsc": 72,
            "extracurricular": 0,
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("**Without Placement Training**")
            profile_without = {**base_profile, "training": 0}
            prob_without = make_prediction(profile_without, placement_model)
            if prob_without is not None:
                st.metric("Probability", f"{prob_without:.2f}%")
        
        with col2:
            st.info("**With Placement Training**")
            profile_with = {**base_profile, "training": 1}
            prob_with = make_prediction(profile_with, placement_model)
            if prob_with is not None:
                st.metric("Probability", f"{prob_with:.2f}%")
        
        if prob_without is not None and prob_with is not None:
            improvement = prob_with - prob_without
            st.markdown("---")
            if improvement > 0:
                st.success(f"🚀 Training Impact: **+{improvement:.2f}%** improvement")
            elif improvement < 0:
                st.warning(f"⚠️ Training Impact: **{improvement:.2f}%** (negative impact)")
            else:
                st.info("ℹ️ Training Impact: No change")
    
    st.markdown("---")
    st.subheader("🎯 TEST 10: Soft Skills Impact")
    st.caption("Compare predictions with LOW vs HIGH soft skills for the same candidate")
    
    if st.button("🧪 Run Soft Skills Impact Test", use_container_width=True, key="skills_test"):
        base_profile = {
            "cgpa": 7.5,
            "internships": 1,
            "projects": 2,
            "workshops": 1,
            "aptitude": 75,
            "ssc": 70,
            "hsc": 72,
            "extracurricular": 0,
            "training": 1,
        }
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.warning("**Low Soft Skills (3.0/5)**")
            profile_low_skills = {**base_profile, "skills": 3.0}
            prob_low = make_prediction(profile_low_skills, placement_model)
            if prob_low is not None:
                st.metric("Probability", f"{prob_low:.2f}%")
        
        with col2:
            st.success("**High Soft Skills (4.8/5)**")
            profile_high_skills = {**base_profile, "skills": 4.8}
            prob_high = make_prediction(profile_high_skills, placement_model)
            if prob_high is not None:
                st.metric("Probability", f"{prob_high:.2f}%")
        
        if prob_low is not None and prob_high is not None:
            improvement = prob_high - prob_low
            st.markdown("---")
            st.info(f"📈 Soft Skills Impact: **+{improvement:.2f}%** improvement")