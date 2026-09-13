# """
# CutisAI - 4-SHOT HIGH-ACCURACY CLINICAL DERMATOLOGY PLATFORM
# -------------------------------------------------------------
# Features:
#   - 3-Step Clinical Intake: Gender, Age Group & Skin Concerns (with 'I don't know' mode)
#   - 4-Shot Multi-Angle Scanning (Front, Right Cheek, Left Cheek, Forehead T-Zone)
#   - Multi-Angle Exposure Normalization (CLAHE in LAB space)
#   - Multi-View Spatial Fusion Algorithm (Consistent, repeatable scoring)
#   - Pimple & Acne Spot Detection & Personalized Medicine Guide
#   - Age- & Gender-Calibrated "What to Avoid" triggers
#   - 100% Free Kitchen Homecare & Pharmacy Picks
#   - Automatic Browser Opener + Instant Demo Mode

# HOW TO RUN:
#   python single_file_app.py
# """

import os
import sys
import base64
import threading
import time
import webbrowser
import unittest.mock as mock
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template_string

# -------------------------------------------------------------------
# 1. OPENCV & MEDIAPIPE COMPUTER VISION ENGINE
# -------------------------------------------------------------------
# Guard against Windows AppLocker / AppControl DLL restriction on matplotlib's ft2font
if 'matplotlib' not in sys.modules:
    sys.modules['matplotlib'] = mock.MagicMock()
if 'matplotlib.pyplot' not in sys.modules:
    sys.modules['matplotlib.pyplot'] = mock.MagicMock()

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    MEDIAPIPE_AVAILABLE = False
    print(f"[CutisAI Notice] MediaPipe not loaded: {e}. Falling back to OpenCV Cascade.")


class SkinAnalyzer:
    def __init__(self):
        self.mp_available = MEDIAPIPE_AVAILABLE
        self.face_mesh = None
        self.hands = None
        if self.mp_available:
            try:
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.4
                )
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=True, max_num_hands=2, min_detection_confidence=0.4
                )
            except Exception as e:
                print(f"[CutisAI] FaceMesh initialization warning: {e}. Switching to OpenCV Haar fallback.")
                self.mp_available = False

        # Facial Dermatological Zones
        self.FOREHEAD_IDXS = [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109, 151, 9]
        self.NOSE_T_ZONE_IDXS = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2, 98, 327, 126, 355]
        self.LEFT_CHEEK_IDXS = [116, 123, 147, 213, 138, 135, 136, 150, 149, 176, 148, 152, 58, 172]
        self.RIGHT_CHEEK_IDXS = [345, 352, 376, 433, 367, 364, 365, 379, 378, 400, 377, 152, 288]
        self.CHIN_IDXS = [175, 199, 200, 18, 152, 396, 377, 400, 150, 149, 176, 148]

    def _get_polygon_points(self, landmarks, indices, w, h):
        pts = []
        for idx in indices:
            if idx < len(landmarks):
                lm = landmarks[idx]
                pts.append([int(lm.x * w), int(lm.y * h)])
        return np.array(pts, dtype=np.int32) if pts else None

    def _create_convex_mask(self, shape, pts):
        mask = np.zeros(shape[:2], dtype=np.uint8)
        if pts is not None and len(pts) >= 3:
            cv2.fillConvexPoly(mask, cv2.convexHull(pts), 255)
        return mask

    def normalize_lighting(self, bgr_img):
        """Standardizes brightness across webcam shots using CLAHE in LAB space."""
        lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        l_balanced = cv2.addWeighted(l, 0.4, cl, 0.6, 0)
        limg = cv2.merge((l_balanced, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def _extract_skin_features(self, bgr_img, mask):
        skin_pixels_count = cv2.countNonZero(mask)
        if skin_pixels_count < 120:
            return None

        hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
        b, g, r = cv2.split(bgr_img)
        h, s, v = cv2.split(hsv)

        # 1. Sebum / Oiliness (Specular Highlights in HSV)
        mean_v = cv2.mean(v, mask=mask)[0]
        mean_s = cv2.mean(s, mask=mask)[0]
        gloss_threshold_v = min(242, max(190, mean_v * 1.22))
        gloss_mask = cv2.inRange(v, int(gloss_threshold_v), 255)
        desat_mask = cv2.inRange(s, 0, int(max(32, mean_s * 0.72)))
        specular_mask = cv2.bitwise_and(cv2.bitwise_and(gloss_mask, desat_mask), mask)
        specular_ratio = (cv2.countNonZero(specular_mask) / skin_pixels_count) * 100.0
        oiliness_score = float(np.clip(specular_ratio * 12.0, 10.0, 96.0))

        # 2. Moisture & Texture (Laplacian Micro-Roughness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        _, lap_std = cv2.meanStdDev(np.abs(laplacian), mask=mask)
        roughness_metric = float(lap_std[0][0])
        smoothness_factor = max(0.0, 100.0 - (roughness_metric * 4.0))
        moisture_score = float(np.clip(smoothness_factor * 0.85 + 10.0, 20.0, 95.0))

        # 3. Pores & Clarity
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, bh_std = cv2.meanStdDev(blackhat, mask=mask)
        clarity_score = float(np.clip(100.0 - (float(bh_std[0][0]) * 8.2), 15.0, 98.0))

        # 4. Redness & Sensitivity: (R-G)/(R+G)
        denom = r.astype(np.float32) + g.astype(np.float32) + 1e-5
        erythema_map = ((r.astype(np.float32) - g.astype(np.float32)) / denom) * 100.0
        erythema_mean = float(cv2.mean(erythema_map, mask=mask)[0])
        redness_score = float(np.clip((erythema_mean - 10.0) * 3.2, 5.0, 95.0))

        # 5. Pimple / Inflammatory Acne Spot Detection
        red_diff = cv2.subtract(r, g)
        mean_rd, std_rd = cv2.meanStdDev(red_diff, mask=mask)
        pimple_thresh = int(mean_rd[0][0] + 1.7 * std_rd[0][0])
        pimple_thresh = max(1, min(254, pimple_thresh))
        _, p_mask = cv2.threshold(red_diff, pimple_thresh, 255, cv2.THRESH_BINARY)
        p_mask = cv2.bitwise_and(p_mask, mask)

        p_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        p_clean = cv2.morphologyEx(p_mask, cv2.MORPH_OPEN, p_kernel)
        contours, _ = cv2.findContours(p_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pimple_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 14 <= area <= 320:
                pimple_count += 1

        return {
            "oiliness": round(oiliness_score, 1),
            "moisture": round(moisture_score, 1),
            "clarity": round(clarity_score, 1),
            "roughness": round(roughness_metric, 2),
            "redness": round(redness_score, 1),
            "pimple_count": int(pimple_count)
        }

    def analyze_single_frame(self, bgr_img, angle="front"):
        """Analyzes a specific angle frame with pose-aware segmentation."""
        norm_img = self.normalize_lighting(bgr_img)
        h, w, _ = norm_img.shape
        rgb = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)

        results = None
        if self.mp_available and self.face_mesh:
            try:
                results = self.face_mesh.process(rgb)
            except Exception as e:
                print(f"[CutisAI] FaceMesh processing warning: {e}")

        if not results or not results.multi_face_landmarks:
            # Fallback OpenCV face detector
            gray = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.2, 4)
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.ellipse(mask, (fx + fw//2, fy + fh//2), (fw//3, fh//2), 0, 0, 360, 255, -1)
                metrics = self._extract_skin_features(norm_img, mask)
                if metrics:
                    return {"success": True, "metrics": metrics, "zones": {"t_oil": metrics["oiliness"], "u_oil": metrics["oiliness"]}}
            return {"success": False, "error": f"Could not detect face in {angle.upper()} shot. Keep face steady and well-lit."}

        lms = results.multi_face_landmarks[0].landmark
        m_forehead = self._create_convex_mask(norm_img.shape, self._get_polygon_points(lms, self.FOREHEAD_IDXS, w, h))
        m_tzone = self._create_convex_mask(norm_img.shape, self._get_polygon_points(lms, self.NOSE_T_ZONE_IDXS, w, h))
        m_lcheek = self._create_convex_mask(norm_img.shape, self._get_polygon_points(lms, self.LEFT_CHEEK_IDXS, w, h))
        m_rcheek = self._create_convex_mask(norm_img.shape, self._get_polygon_points(lms, self.RIGHT_CHEEK_IDXS, w, h))
        m_chin = self._create_convex_mask(norm_img.shape, self._get_polygon_points(lms, self.CHIN_IDXS, w, h))

        full_mask = cv2.bitwise_or(cv2.bitwise_or(m_forehead, m_tzone), cv2.bitwise_or(cv2.bitwise_or(m_lcheek, m_rcheek), m_chin))
        metrics = self._extract_skin_features(norm_img, full_mask)
        t_metrics = self._extract_skin_features(norm_img, cv2.bitwise_or(m_forehead, m_tzone))
        u_metrics = self._extract_skin_features(norm_img, cv2.bitwise_or(m_lcheek, m_rcheek))

        if not metrics:
            return {"success": False, "error": "Skin area too small or shaded."}

        t_oil = t_metrics["oiliness"] if t_metrics else metrics["oiliness"]
        u_oil = u_metrics["oiliness"] if u_metrics else metrics["oiliness"]

        return {
            "success": True,
            "metrics": metrics,
            "zones": {
                "t_oil": round(t_oil, 1),
                "u_oil": round(u_oil, 1)
            }
        }

    def fuse_4_shots(self, shots_dict):
        """
        4-View Clinical Spatial Fusion Algorithm.
        Averages out single-frame camera noise and lighting glare for consistent results.
        """
        results = {}
        for angle in ["front", "right", "left", "top"]:
            if angle in shots_dict and shots_dict[angle] is not None:
                res = self.analyze_single_frame(shots_dict[angle], angle=angle)
                results[angle] = res

        valid_angles = [a for a, r in results.items() if r.get("success")]
        if not valid_angles:
            return {"success": False, "error": "Failed to analyze facial angles. Ensure steady lighting and clear camera view."}

        # Weighted Multiview Fusion
        weights = {
            "front": {"oil": 0.25, "moist": 0.30, "clarity": 0.30, "red": 0.30},
            "right": {"oil": 0.20, "moist": 0.25, "clarity": 0.25, "red": 0.25},
            "left":  {"oil": 0.20, "moist": 0.25, "clarity": 0.25, "red": 0.25},
            "top":   {"oil": 0.35, "moist": 0.20, "clarity": 0.20, "red": 0.20}
        }

        tot_w_oil = sum(weights[a]["oil"] for a in valid_angles)
        tot_w_moist = sum(weights[a]["moist"] for a in valid_angles)
        tot_w_clarity = sum(weights[a]["clarity"] for a in valid_angles)
        tot_w_red = sum(weights[a]["red"] for a in valid_angles)

        fused_oil = sum(results[a]["metrics"]["oiliness"] * weights[a]["oil"] for a in valid_angles) / tot_w_oil
        fused_moist = sum(results[a]["metrics"]["moisture"] * weights[a]["moist"] for a in valid_angles) / tot_w_moist
        fused_clarity = sum(results[a]["metrics"]["clarity"] * weights[a]["clarity"] for a in valid_angles) / tot_w_clarity
        fused_red = sum(results[a]["metrics"]["redness"] * weights[a]["red"] for a in valid_angles) / tot_w_red

        max_pimples = max(results[a]["metrics"]["pimple_count"] for a in valid_angles)

        t_oils = [results[a]["zones"]["t_oil"] for a in valid_angles if "zones" in results[a]]
        u_oils = [results[a]["zones"]["u_oil"] for a in valid_angles if "zones" in results[a]]
        final_t_oil = float(np.median(t_oils)) if t_oils else fused_oil
        final_u_oil = float(np.median(u_oils)) if u_oils else fused_oil

        fused_oil = round(fused_oil, 1)
        fused_moist = round(fused_moist, 1)
        fused_clarity = round(fused_clarity, 1)
        fused_red = round(fused_red, 1)

        # Clinical Skin Classification
        if max_pimples >= 4:
            skin_type = "Active Acne-Prone Skin (Inflammatory Breakout)"
        elif fused_red > 65:
            skin_type = "Sensitive & Reactive (High Erythema)"
        elif (final_t_oil - final_u_oil) > 20 and final_t_oil > 50:
            skin_type = "Combination Skin (Oily T-Zone / Dry Cheeks)"
        elif fused_oil >= 60:
            skin_type = "Oily Skin (High Sebum Excretion)"
        elif fused_moist <= 42 or fused_oil <= 25:
            skin_type = "Dry / Xerosis-Prone Skin (Low Moisture)"
        else:
            skin_type = "Balanced / Normal Skin (Healthy Barrier)"

        return {
            "success": True,
            "skin_type": skin_type,
            "shots_analyzed": len(valid_angles),
            "angles_used": valid_angles,
            "overall": {
                "oiliness": fused_oil,
                "moisture": fused_moist,
                "clarity": fused_clarity,
                "redness": fused_red,
                "pimple_count": max_pimples
            },
            "zones": {
                "t_zone_oiliness": round(final_t_oil, 1),
                "u_zone_oiliness": round(final_u_oil, 1)
            }
        }


# -------------------------------------------------------------------
# 2. PERSONALIZED MEDICINE & DERMATOLOGY KNOWLEDGE BASE
# -------------------------------------------------------------------
def get_recommendations(skin_type, metrics, user_profile=None):
    if user_profile is None:
        user_profile = {}

    gender = user_profile.get("gender", "Unspecified")
    age_group = user_profile.get("age_group", "Unspecified")
    concerns = user_profile.get("concerns", [])
    unknown_skin = "I don't know about my skin" in concerns or not concerns

    pimples = metrics.get("pimple_count", 0)

    # Personalized summary banner
    if unknown_skin:
        discovery_note = f"🔍 Autonomous CutisAI Discovery: You requested a fresh diagnostic baseline. Based on 4-angle computer-vision triangulation, your skin profile is diagnosed as '{skin_type}' with {round(metrics.get('oiliness', 0))}% sebum gloss and {pimples} inflammatory spots."
    else:
        concerns_str = ", ".join(concerns)
        discovery_note = f"📋 Correlated Clinical Diagnosis: Cross-referencing your reported concerns ({concerns_str}) with 4-view optical scan confirms '{skin_type}'."

    # Age-calibrated nuances
    age_notes = []
    if age_group == "Under 18":
        age_notes.append("Pubertal Androgen Activity: Sebum glands respond aggressively to hormonal surges. Prioritize gentle BHA unclogging over harsh scrubs.")
    elif age_group == "18 - 25":
        age_notes.append("Peak Hormonal Sebum Excretion: High cellular turnover requires non-comedogenic hydration and diligent PM spot treatments.")
    elif age_group == "26 - 35":
        age_notes.append("Adult Barrier Transition: Focus on preventing post-inflammatory hyperpigmentation (PIH) with Niacinamide and barrier ceramides.")
    elif age_group in ["36 - 50", "50+"]:
        age_notes.append("Epidermal Lipid Preservation: Natural ceramide production slows down. Shield skin against trans-epidermal water loss (TEWL).")

    # Gender-calibrated nuances
    gender_notes = []
    if gender == "Male":
        gender_notes.append("Dermal Thickness & Shaving Folliculitis: Thicker stratum corneum with coarse hair follicles. Disinfect multi-blade razors and avoid alcohol aftershaves.")
    elif gender == "Female":
        gender_notes.append("Cyclical Hormonal Fluctuations: Luteal phase progesterone shifts can cause pre-menstrual jawline flare-ups. Use non-pore-clogging mineral SPF.")

    pimple_medicine_guide = {
        "detected_count": pimples,
        "severity": "Mild (1-2 pimples)" if pimples <= 2 else ("Moderate (3-5 pimples)" if pimples <= 5 else "Severe (>5 inflammatory acne spots)"),
        "topical_medicines": [
            {
                "name": "Benzoyl Peroxide 2.5% Gel",
                "brand_example": "Benxop 2.5% / Persol AC 2.5% Gel (~₹120 - ₹150)",
                "how_to_use": "Apply a tiny micro-dot directly onto the active pimple head at bedtime after washing face. Do NOT rub across entire face.",
                "action": "Kills acne-causing bacteria (C. acnes) within 48 hours via free-radical oxygen release. Non-antibiotic, preventing bacterial resistance.",
                "badge": "Top Dermatologist Pick"
            },
            {
                "name": "Clindamycin 1% + Nicotinamide 4% Gel",
                "brand_example": "Clindac-A / Faceclin Gel (~₹140 - ₹180)",
                "how_to_use": "Apply a small dot on painful, swollen pimples twice daily after cleansing.",
                "action": "Topical antibiotic halts bacterial multiplication while Nicotinamide (Vitamin B3) soothes surrounding inflammatory erythema.",
                "badge": "For Red & Pus-Filled Spots"
            },
            {
                "name": "Hydrocolloid Pimple Spot Patch / Salicylic Acid 2%",
                "brand_example": "Sirona Pimple Patch / Minimalist Spot Corrector (~₹199)",
                "how_to_use": "Stick the patch directly over the pimple head before sleeping. Peel off in morning.",
                "action": "Absorbs trapped inflammatory exudate overnight while physically preventing fingernail picking, excoriation, and scarring.",
                "badge": "Overnight Zero-Scarring"
            },
            {
                "name": "Adapalene 0.1% Gel (Topical Retinoid)",
                "brand_example": "Adaferin 0.1% Gel (~₹280)",
                "how_to_use": "Only at night, pea-sized amount over acne-prone area. Always wear SPF 50+ the next morning.",
                "action": "Normalizes keratinization and follicular turnover inside the pore, preventing micro-comedones from developing into breakouts.",
                "badge": "Long-Term Prevention"
            }
        ],
        "natural_zero_cost_medicines": [
            {
                "name": "Ice Cube Cold Compress",
                "prep": "Wrap a single ice cube in a clean cotton handkerchief. Press gently onto the throbbing pimple for 60-90 seconds.",
                "why": "Instantly constricts swollen micro-capillaries, reducing pimple redness and throbbing pain within minutes."
            },
            {
                "name": "Fresh Neem Leaf & Kasturi Turmeric Spot Dab",
                "prep": "Crush 4-5 fresh Neem leaves with a pinch of pure Kasturi Haldi into a fine paste. Dab directly onto the pimple head for 20 mins, then rinse.",
                "why": "Neem contains natural azadirachtin which is a potent herbal antibacterial, while curcumin reduces swelling without burning the skin."
            }
        ],
        "prescription_warning": "⚠️ IMPORTANT MEDICAL RULE: Do NOT self-medicate with oral antibiotic pills (like Doxycycline, Minocycline) or oral Isotretinoin/Accutane without visiting a certified dermatologist in person. Oral pills require blood tests and doctor supervision. Topical spot gels above are safe Over-The-Counter remedies."
    }

    base_avoid = []
    if "Oily" in skin_type or "Acne" in skin_type:
        diagnosis_desc = "Sebum Hyper-excretion & Multi-Zone Follicular Congestion"
        base_avoid = [
            {
                "category": "☀️ Peak UV Sun Exposure (11 AM - 3 PM)",
                "avoid": "Never stand in harsh direct afternoon sun without shade or physical face cover.",
                "why": "Sun UV rays oxidize natural skin sebum (squalene) into squalene peroxide, which directly plugs pores and causes severe cystic acne breakouts!",
                "zero_cost_fix": "Cover your face with a clean cotton scarf/cloth or wear a wide-brim cap/umbrella when outside."
            },
            {
                "category": "🚿 Steaming Hot Water Face Wash",
                "avoid": "Never wash your oily face with steaming hot water.",
                "why": "Hot water dissolves your natural lipid barrier completely. Your sebaceous glands panic and produce 2x MORE rebound oil within 30 minutes!",
                "zero_cost_fix": "Always wash your face with cool or room-temperature lukewarm water only."
            },
            {
                "category": "🖐️ Touching Face & Popping Pimples",
                "avoid": "Avoid resting your chin/cheeks on hands or picking at shiny spots.",
                "why": "Fingers transfer Staphylococcus bacteria and oil directly into open pores, turning sebum into inflamed red pustules.",
                "zero_cost_fix": "Keep hands strictly off the face; pat forehead sweat with a clean cotton handkerchief."
            },
            {
                "category": "🍽️ Sugary & High-Glycemic Food Triggers",
                "avoid": "Avoid sodas, energy drinks, bakery pastries, excess dairy, and deep-fried junk.",
                "why": "High-sugar foods spike Insulin & IGF-1 hormones, which trigger your sebaceous glands to produce thicker, stickier sebum.",
                "zero_cost_fix": "Drink plain water with lemon, and eat whole fruits instead of sugary fruit juices."
            }
        ]
        homecare = [
            {
                "title": "Multani Mitti (Fuller's Earth) & Rosewater Pack",
                "cost": "₹0 - ₹15 (Pantry Item)",
                "ingredients": "1 tbsp Multani Mitti + 2 tbsp Pure Rosewater",
                "instructions": "Make smooth paste. Apply for 10-12 mins (rinse before it fully cracks). Use 2x a week.",
                "why": "Natural magnesium clay adsorbs deep pore oil without stripping skin's moisture."
            },
            {
                "title": "Green Tea & Mint Pore Ice Cubes",
                "cost": "₹5 (Kitchen Remedy)",
                "ingredients": "1 brewed green tea bag frozen into ice cubes",
                "instructions": "Wrap in clean cloth and glide over T-zone for 60 seconds every morning.",
                "why": "EGCG polyphenol antioxidants suppress sebaceous gland activity and soothe redness."
            }
        ]
        commercial = [
            {"category": "Cleanser", "active": "Salicylic Acid 2% (BHA) + Zinc", "product": "Minimalist Salicylic Cleanser (~₹299)"},
            {"category": "Serum", "active": "Niacinamide 10% + Zinc 1%", "product": "The Derma Co 10% Niacinamide (~₹499)"},
            {"category": "Moisturizer", "active": "Oil-Free Hyaluronic Water Gel", "product": "Pond's Super Light Gel (~₹240)"},
            {"category": "Sunscreen", "active": "Ultra-Matte Dry Touch SPF 50+", "product": "Blynds Emulgel SPF 50 (~₹450)"}
        ]
    elif "Dry" in skin_type:
        diagnosis_desc = "Epidermal Lipid Deficiency & Stratum Corneum Dehydration"
        base_avoid = [
            {
                "category": "☀️ Unprotected Sun Exposure & Dry Winds",
                "avoid": "Avoid sitting under direct midday sun or fast dry wind.",
                "why": "Dry skin has an already compromised lipid shield. Sun directly accelerates trans-epidermal water loss, causing burning, premature fine lines, and flaking.",
                "zero_cost_fix": "Stay in shade during peak afternoon (11 AM - 3 PM) and wrap a soft cotton dupatta/cloth when traveling."
            },
            {
                "category": "🚿 Long Hot Showers & Harsh Soaps",
                "avoid": "Avoid long steaming showers and alkaline bar soaps (Lifebuoy/Dettol on face).",
                "why": "Standard soaps have a high pH (~9) which destroys the natural acidic mantle (~5.5), stripping the remaining protective ceramides.",
                "zero_cost_fix": "Splash with plain cold/lukewarm water in the morning. Skip morning soap entirely!"
            },
            {
                "category": "🧣 Rough Towel Friction (Ragdana)",
                "avoid": "Never rub your face vigorously with a bath towel.",
                "why": "Friction on dry, flaky skin causes microscopic epidermal tears, irritation, and sensitivity.",
                "zero_cost_fix": "Always pat dry softly with a clean cotton t-shirt or soft cloth."
            }
        ]
        homecare = [
            {
                "title": "Raw Honey & Fresh Malai (Milk Cream) Barrier Quench",
                "cost": "₹0 (Kitchen Staple)",
                "ingredients": "1 tsp Raw Honey + 1/2 tsp Fresh Malai (milk cream)",
                "instructions": "Mix and apply on face for 15-20 mins. Wipe off with warm damp cloth. 3x a week.",
                "why": "Honey pulls water into cells while malai supplies essential milk ceramides and lipids."
            }
        ]
        commercial = [
            {"category": "Cleanser", "active": "Gentle Hydrating Lotion (Non-Foaming)", "product": "Cetaphil Gentle Skin Cleanser (~₹325)"},
            {"category": "Serum", "active": "Hyaluronic Acid 2% + Vitamin B5", "product": "The Ordinary / Minimalist HA (~₹699)"},
            {"category": "Moisturizer", "active": "Ceramides Complex 1, 3, 6-II", "product": "Bioderma Atoderm Crème / Re'equil (~₹450)"},
            {"category": "Sunscreen", "active": "Dewy Nourishing Sunscreen SPF 50+", "product": "The Derma Co Hyaluronic Sun Gel (~₹499)"}
        ]
    else:
        diagnosis_desc = "Combination Sebum Imbalance (Oily T-Zone with Normal/Dry Cheeks)"
        base_avoid = [
            {
                "category": "☀️ Direct Afternoon Sunlight",
                "avoid": "Avoid walking unprotected in direct sunlight between 11 AM - 3 PM.",
                "why": "UV rays trigger sweat and excess sebum on your forehead/nose while simultaneously parching and drying your cheeks.",
                "zero_cost_fix": "Use a light umbrella or soft cloth face wrap to block UV rays naturally."
            },
            {
                "category": "🧼 Over-washing With Soap",
                "avoid": "Avoid washing your face 4-5 times a day just because your forehead feels oily.",
                "why": "Frequent soaping dehydrates your cheeks, worsening the difference between T-zone and cheeks.",
                "zero_cost_fix": "Wash only twice a day. During afternoon, simply dab your forehead with a dry tissue."
            }
        ]
        homecare = [
            {
                "title": "Zone-Mapping Dual Homecare Mask",
                "cost": "₹0 - ₹10",
                "ingredients": "T-Zone: Multani Mitti | Cheeks: Honey & Curd",
                "instructions": "Apply clay on forehead/nose and honey-curd on cheeks simultaneously. Rinse after 10 mins.",
                "why": "Controls forehead grease without stripping the delicate dry cheek skin."
            }
        ]
        commercial = [
            {"category": "Cleanser", "active": "Balanced pH 5.5 Gel Cleanser", "product": "Simple Refreshing Face Wash (~₹280)"},
            {"category": "Serum", "active": "Niacinamide 5% + Centella", "product": "Minimalist Niacinamide 05% (~₹550)"},
            {"category": "Moisturizer", "active": "Lightweight Hydrating Water Cream", "product": "Dot & Key 72HR Hydrating Gel (~₹495)"},
            {"category": "Sunscreen", "active": "Weightless Aqua Gel SPF 50+", "product": "Aqualogica Radiance Dewy SPF 50 (~₹449)"}
        ]

    # Gender-specific trigger injection
    if gender == "Male":
        base_avoid.append({
            "category": "🪒 Dry Shaving & Dull Multi-Blade Razors",
            "avoid": "Never shave dry or use a razor cartridge more than 5-6 times.",
            "why": "Dull razors cause micro-lacerations and push facial Staphylococcus bacteria into follicles, triggering barber's itch (folliculitis) and red bumps.",
            "zero_cost_fix": "Shave in the direction of hair growth only after a warm shower with plenty of lubricated lather."
        })

    return {
        "clinical_diagnosis": diagnosis_desc,
        "discovery_note": discovery_note,
        "patient_profile": {
            "gender": gender,
            "age_group": age_group,
            "concerns": concerns,
            "unknown_mode": unknown_skin,
            "clinical_notes": age_notes + gender_notes
        },
        "pimple_guide": pimple_medicine_guide,
        "what_to_avoid": base_avoid,
        "homecare": homecare,
        "commercial": commercial
    } 

# -------------------------------------------------------------------
# 3. COMPLETE MULTI-STEP ONBOARDING + 4-SHOT GUIDED HTML5/JS/CSS
# -------------------------------------------------------------------
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CutisAI - Clinical 4-Shot Dermatology Scanner</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
:root { 
  --bg:#090D14; 
  --card:rgba(18,24,38,0.85); 
  --card-hover:rgba(26,35,55,0.95);
  --border:rgba(255,255,255,0.08); 
  --border-focus:rgba(0,229,255,0.5);
  --primary:#00E5FF; 
  --green:#10B981; 
  --amber:#FFB800; 
  --rose:#EC4899; 
  --red:#EF4444; 
}
* { margin:0; padding:0; box-sizing:border-box; font-family:'Plus Jakarta Sans', sans-serif; }
body { background:var(--bg); color:#F8FAFC; min-height:100vh; padding:1.5rem; }

.header { max-width:1150px; margin:0 auto 1.5rem; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem; }
.logo { font-size:1.6rem; font-weight:800; } .logo span { color:var(--primary); }
.badge-4shot { background:linear-gradient(135deg, rgba(0,229,255,0.2), rgba(139,92,246,0.2)); border:1px solid rgba(0,229,255,0.4); color:var(--primary); font-size:0.75rem; font-weight:800; padding:4px 12px; border-radius:20px; }

.container { max-width:1150px; margin:0 auto; }

/* Top Clinical Intake Progress Stepper */
.intake-progress { display:grid; grid-template-columns:repeat(4, 1fr); gap:0.6rem; margin-bottom:1.5rem; max-width:850px; margin-left:auto; margin-right:auto; }
.prog-step { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:0.6rem; text-align:center; transition:all 0.3s ease; opacity:0.5; }
.prog-step.active { opacity:1; border-color:var(--primary); background:rgba(0,229,255,0.08); box-shadow:0 0 12px rgba(0,229,255,0.2); }
.prog-step.done { opacity:1; border-color:var(--green); background:rgba(16,185,129,0.08); }
.prog-step-num { font-size:0.65rem; font-weight:800; text-transform:uppercase; color:#94A3B8; }
.prog-step-title { font-size:0.82rem; font-weight:700; color:#FFF; margin-top:2px; }

/* Generic Wizard Card */
.wizard-card { background:var(--card); border:1px solid var(--border); border-radius:22px; padding:2rem; max-width:760px; margin:0 auto 2rem; text-align:center; animation:fadeIn 0.4s ease-out; }
@keyframes fadeIn { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }

.wizard-title { font-size:1.6rem; font-weight:800; color:#FFF; margin-bottom:0.4rem; }
.wizard-desc { font-size:0.92rem; color:#94A3B8; margin-bottom:1.75rem; line-height:1.5; }

/* Option Selection Grids */
.option-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:1rem; margin-bottom:2rem; }
.option-card { background:rgba(255,255,255,0.03); border:1px solid var(--border); border-radius:16px; padding:1.2rem 1rem; cursor:pointer; transition:all 0.25s ease; user-select:none; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:0.5rem; }
.option-card:hover { border-color:rgba(0,229,255,0.35); transform:translateY(-3px); background:rgba(255,255,255,0.05); }
.option-card.selected { border-color:var(--primary); background:rgba(0,229,255,0.12); box-shadow:0 0 20px rgba(0,229,255,0.25); }
.option-icon { font-size:2.2rem; }
.option-label { font-size:0.95rem; font-weight:700; color:#FFF; }
.option-sub { font-size:0.75rem; color:#94A3B8; }

/* Concerns Grid (Multi-Select) */
.concern-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(210px, 1fr)); gap:0.85rem; margin-bottom:2rem; text-align:left; }
.concern-card { background:rgba(255,255,255,0.03); border:1px solid var(--border); border-radius:14px; padding:1rem; cursor:pointer; transition:all 0.2s ease; display:flex; align-items:center; gap:0.75rem; }
.concern-card:hover { border-color:rgba(0,229,255,0.3); background:rgba(255,255,255,0.05); }
.concern-card.selected { border-color:var(--primary); background:rgba(0,229,255,0.12); box-shadow:0 0 16px rgba(0,229,255,0.2); }
.concern-card.unknown-mode { border-color:var(--amber); background:rgba(255,184,0,0.12); box-shadow:0 0 20px rgba(255,184,0,0.25); }
.concern-check { width:20px; height:20px; border-radius:6px; border:2px solid #64748B; display:flex; align-items:center; justify-content:center; font-size:0.75rem; font-weight:800; color:#040810; transition:all 0.2s; flex-shrink:0; }
.concern-card.selected .concern-check { background:var(--primary); border-color:var(--primary); }
.concern-card.unknown-mode .concern-check { background:var(--amber); border-color:var(--amber); }

/* Navigation Buttons */
.wizard-actions { display:flex; justify-content:center; gap:1rem; align-items:center; }
.btn-next { background:linear-gradient(135deg, #00E5FF, #0284C7); color:#040810; font-size:1rem; font-weight:800; border:none; padding:0.85rem 2rem; border-radius:14px; cursor:pointer; box-shadow:0 0 20px rgba(0,229,255,0.3); transition:all 0.2s; display:inline-flex; align-items:center; gap:0.5rem; }
.btn-next:hover { transform:scale(1.02); }
.btn-next:disabled { opacity:0.35; cursor:not-allowed; transform:none; box-shadow:none; }
.btn-back { background:rgba(255,255,255,0.06); border:1px solid var(--border); color:#CBD5E1; font-weight:700; padding:0.85rem 1.6rem; border-radius:14px; cursor:pointer; transition:all 0.2s; }
.btn-back:hover { background:rgba(255,255,255,0.1); }

/* Patient Banner */
.patient-badge { background:rgba(0,229,255,0.08); border:1px solid rgba(0,229,255,0.25); border-radius:14px; padding:0.75rem 1.25rem; margin-bottom:1.5rem; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.5rem; font-size:0.85rem; }
.badge-edit-btn { color:var(--primary); background:transparent; border:none; cursor:pointer; font-weight:700; text-decoration:underline; font-size:0.8rem; }

/* 4-Shot Stepper Bar inside Scanner */
.stepper-bar { display:grid; grid-template-columns:repeat(4, 1fr); gap:0.75rem; margin-bottom:1.25rem; max-width:850px; margin-left:auto; margin-right:auto; }
.step-item { background:var(--card); border:1px solid var(--border); border-radius:14px; padding:0.75rem; text-align:center; transition:all 0.3s ease; opacity:0.6; }
.step-item.active { opacity:1; border-color:var(--primary); box-shadow:0 0 15px rgba(0,229,255,0.25); background:rgba(0,229,255,0.08); }
.step-item.done { opacity:1; border-color:var(--green); background:rgba(16,185,129,0.08); }
.step-num { font-size:0.7rem; font-weight:800; text-transform:uppercase; color:#94A3B8; }
.step-title { font-size:0.9rem; font-weight:700; color:#FFF; margin-top:2px; }

/* Camera Viewport */
.cam-card { background:var(--card); border:1px solid var(--border); border-radius:22px; padding:1.5rem; max-width:850px; margin:0 auto 2rem; text-align:center; }
.vid-box { position:relative; width:100%; aspect-ratio:4/3; background:#000; border-radius:16px; overflow:hidden; border:1px solid var(--border); }
video { width:100%; height:100%; object-fit:cover; transform:scaleX(-1); }

/* Dynamic Target Alignment Guide */
.guide-overlay { position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); width:240px; height:320px; border:2px dashed var(--primary); border-radius:50%/60% 60% 40% 40%; pointer-events:none; transition:all 0.4s ease; }
.guide-arrow { position:absolute; font-size:2.5rem; color:var(--primary); display:none; animation:bounce 1.2s infinite alternate; }
@keyframes bounce { from { transform:translateX(0); } to { transform:translateX(15px); } }
.arrow-right { right:15%; top:45%; display:none; }
.arrow-left { left:15%; top:45%; display:none; }
.arrow-up { top:15%; left:47%; display:none; }

.prompt-banner { position:absolute; bottom:12px; left:50%; transform:translateX(-50%); background:rgba(9,13,20,0.88); backdrop-filter:blur(8px); border:1px solid rgba(255,255,255,0.1); padding:0.45rem 1.4rem; border-radius:24px; font-size:0.85rem; font-weight:700; color:#FFF; }
.flash { position:absolute; top:0; left:0; width:100%; height:100%; background:#FFF; opacity:0; pointer-events:none; transition:opacity 0.2s ease-out; }

/* Controls */
.cam-controls { display:flex; justify-content:center; gap:0.75rem; margin-top:1.25rem; flex-wrap:wrap; }
.btn-snap { background:linear-gradient(135deg, #00E5FF, #0284C7); color:#040810; font-size:1.05rem; font-weight:800; border:none; padding:0.85rem 2.2rem; border-radius:14px; cursor:pointer; box-shadow:0 0 20px rgba(0,229,255,0.3); display:flex; align-items:center; gap:0.5rem; transition:transform 0.2s; }
.btn-snap:hover { transform:scale(1.02); }
.btn-reset { background:rgba(255,255,255,0.06); border:1px solid var(--border); color:#CBD5E1; font-weight:700; padding:0.85rem 1.4rem; border-radius:14px; cursor:pointer; }
.btn-demo { background:rgba(255,255,255,0.06); border:1px solid var(--border); color:#CBD5E1; font-weight:700; padding:0.85rem 1.4rem; border-radius:14px; cursor:pointer; }
.btn-upload { background:rgba(255,255,255,0.06); border:1px solid var(--border); color:#A5F3FC; font-weight:700; padding:0.85rem 1.4rem; border-radius:14px; cursor:pointer; }

/* 4 Thumbnails Strip */
.thumb-strip { display:grid; grid-template-columns:repeat(4, 1fr); gap:0.75rem; margin-top:1.25rem; }
.thumb-box { background:#04060A; border:1px solid var(--border); border-radius:12px; aspect-ratio:4/3; overflow:hidden; position:relative; display:flex; align-items:center; justify-content:center; cursor:pointer; }
.thumb-box img { width:100%; height:100%; object-fit:cover; transform:scaleX(-1); display:none; position:absolute; top:0; left:0; }
.thumb-label { position:absolute; bottom:4px; left:6px; font-size:0.65rem; font-weight:800; text-transform:uppercase; background:rgba(0,0,0,0.75); padding:2px 6px; border-radius:6px; color:#FFF; z-index:2; }
.thumb-placeholder { font-size:1.4rem; opacity:0.4; }

/* Results Section */
.results { background:var(--card); border:1px solid var(--border); border-radius:22px; padding:2rem; display:none; margin-top:2rem; animation:fadeIn 0.5s ease-out; }
.diag-title { font-size:1.8rem; font-weight:800; color:var(--primary); margin-bottom:0.4rem; }
.discovery-banner { background:rgba(255,184,0,0.1); border:1px solid rgba(255,184,0,0.3); border-radius:12px; padding:0.85rem 1.25rem; color:#FDE68A; font-size:0.88rem; line-height:1.5; margin:1rem 0; }
.pimple-pill { display:inline-flex; align-items:center; gap:0.4rem; background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.4); color:#FCA5A5; padding:0.35rem 0.9rem; border-radius:20px; font-size:0.85rem; font-weight:700; margin:0.5rem 0 1rem; }

.gauges { display:grid; grid-template-columns:repeat(auto-fit, minmax(180px, 1fr)); gap:1rem; margin:1.5rem 0; text-align:center; }
.gauge { background:rgba(255,255,255,0.03); border:1px solid var(--border); border-radius:16px; padding:1.2rem; }
.gauge-val { font-size:2.2rem; font-weight:800; color:#FFF; }
.gauge-lbl { font-size:0.8rem; color:#94A3B8; font-weight:600; text-transform:uppercase; margin-bottom:0.4rem; }

/* Zone Breakdown */
.zone-grid { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1.5rem; }
.zone-card { background:rgba(0,0,0,0.25); border:1px solid var(--border); border-radius:14px; padding:1rem; }

/* Tabs */
.tabs-nav { display:flex; gap:0.5rem; border-bottom:1px solid var(--border); padding-bottom:0.75rem; margin:1.5rem 0; overflow-x:auto; }
.tab-btn { background:transparent; border:none; color:#94A3B8; font-weight:700; padding:0.6rem 1.2rem; border-radius:12px; cursor:pointer; font-size:0.9rem; white-space:nowrap; }
.tab-btn.active { color:#FFF; background:rgba(0,229,255,0.15); border:1px solid rgba(0,229,255,0.3); }
.tab-pane { display:none; }
.tab-pane.active { display:block; }

/* Medicine Cards */
.med-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:1rem; margin-bottom:1.5rem; }
.med-card { background:rgba(0, 229, 255, 0.04); border:1px solid rgba(0, 229, 255, 0.25); border-radius:14px; padding:1.3rem; border-left:4px solid var(--primary); position:relative; }
.med-badge { background:rgba(0, 229, 255, 0.2); color:var(--primary); font-size:0.7rem; font-weight:800; padding:2px 8px; border-radius:10px; float:right; }
.med-name { font-size:1.1rem; font-weight:800; color:#FFF; margin-bottom:0.3rem; }
.med-brand { font-size:0.85rem; color:#A5F3FC; margin-bottom:0.5rem; font-weight:600; }
.med-how { font-size:0.82rem; color:#CBD5E1; margin-bottom:0.4rem; line-height:1.5; }
.med-action { background:rgba(0,0,0,0.3); padding:0.6rem; border-radius:8px; font-size:0.8rem; color:#94A3B8; line-height:1.4; }
.doctor-warn { background:rgba(239, 68, 68, 0.12); border:1px solid rgba(239, 68, 68, 0.35); padding:1rem 1.25rem; border-radius:12px; font-size:0.85rem; color:#FCA5A5; margin-bottom:1.5rem; line-height:1.5; }

/* Avoid Cards */
.avoid-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(320px, 1fr)); gap:1rem; }
.avoid-card { background:rgba(239, 68, 68, 0.05); border:1px solid rgba(239, 68, 68, 0.3); border-radius:14px; padding:1.3rem; border-left:4px solid var(--red); }
.avoid-cat { font-size:0.8rem; font-weight:800; color:#F87171; text-transform:uppercase; margin-bottom:0.3rem; }
.avoid-what { font-size:1.05rem; font-weight:700; color:#FFF; margin-bottom:0.4rem; }
.avoid-why { font-size:0.85rem; color:#CBD5E1; line-height:1.5; margin-bottom:0.6rem; }
.avoid-fix { background:rgba(16, 185, 129, 0.1); border:1px solid rgba(16, 185, 129, 0.25); color:#6EE7B7; padding:0.6rem; border-radius:8px; font-size:0.82rem; }

/* Homecare Cards */
.homecare-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(300px, 1fr)); gap:1rem; }
.homecare-card { background:rgba(16,185,129,0.05); border:1px solid rgba(16,185,129,0.3); border-radius:14px; padding:1.2rem; border-left:4px solid var(--green); }
.cost { background:rgba(16,185,129,0.2); color:#34D399; font-size:0.75rem; font-weight:800; padding:2px 8px; border-radius:10px; float:right; }
.card-sub { font-size:0.85rem; color:#CBD5E1; margin:0.4rem 0; }
.why { background:rgba(0,0,0,0.3); padding:0.6rem; border-radius:8px; font-size:0.8rem; color:#A7F3D0; margin-top:0.6rem; }

#file-upload-input { display:none; }
</style>
</head>
<body>

<div class="header">
  <div class="logo">Cutis<span>AI</span> ✨ <span class="badge-4shot">4-Shot Clinical Dermatology Platform</span></div>
  <div style="font-size:0.85rem; color:#94A3B8;">Multi-Angle Facial Diagnostics & Personalized Care</div>
</div>

<div class="container">

  <!-- Clinical Onboarding Progress Bar -->
  <div class="intake-progress">
    <div class="prog-step active" id="prog-0">
      <div class="prog-step-num">Step 1</div>
      <div class="prog-step-title">👤 Gender</div>
    </div>
    <div class="prog-step" id="prog-1">
      <div class="prog-step-num">Step 2</div>
      <div class="prog-step-title">🎂 Age Group</div>
    </div>
    <div class="prog-step" id="prog-2">
      <div class="prog-step-num">Step 3</div>
      <div class="prog-step-title">🩹 Skin Concerns</div>
    </div>
    <div class="prog-step" id="prog-3">
      <div class="prog-step-num">Step 4</div>
      <div class="prog-step-title">📸 4-Angle Scan</div>
    </div>
  </div>

  <!-- ==================== WIZARD PAGE 1: GENDER ==================== -->
  <div class="wizard-card" id="page-gender">
    <div class="wizard-title">What is your biological sex / gender?</div>
    <div class="wizard-desc">Dermal thickness, follicular androgen sensitivity, and sebum gland volume vary significantly. This calibrates your prescription tolerances.</div>
    
    <div class="option-grid">
      <div class="option-card" onclick="selectGender(this, 'Male')">
        <div class="option-icon">👨</div>
        <div class="option-label">Male</div>
        <div class="option-sub">Thicker dermis & beard follicles</div>
      </div>
      <div class="option-card" onclick="selectGender(this, 'Female')">
        <div class="option-icon">👩</div>
        <div class="option-label">Female</div>
        <div class="option-sub">Cyclical hormonal variance</div>
      </div>
      <div class="option-card" onclick="selectGender(this, 'Non-Binary / Other')">
        <div class="option-icon">🧑</div>
        <div class="option-label">Non-Binary</div>
        <div class="option-sub">Balanced clinical baseline</div>
      </div>
      <div class="option-card" onclick="selectGender(this, 'Prefer not to say')">
        <div class="option-icon">🔒</div>
        <div class="option-label">Prefer not to say</div>
        <div class="option-sub">Standard baseline</div>
      </div>
    </div>

    <div class="wizard-actions">
      <button class="btn-next" id="btn-next-gender" disabled onclick="goToPage('age')">Continue to Age ➔</button>
    </div>
  </div>

  <!-- ==================== WIZARD PAGE 2: AGE ==================== -->
  <div class="wizard-card" id="page-age" style="display:none;">
    <div class="wizard-title">What is your age range?</div>
    <div class="wizard-desc">Epidermal turnover, ceramide synthesis, and trans-epidermal water loss (TEWL) shift across life stages.</div>

    <div class="option-grid">
      <div class="option-card" onclick="selectAge(this, 'Under 18')">
        <div class="option-icon">🎒</div>
        <div class="option-label">Under 18</div>
        <div class="option-sub">Pubertal sebum surge</div>
      </div>
      <div class="option-card" onclick="selectAge(this, '18 - 25')">
        <div class="option-icon">🎓</div>
        <div class="option-label">18 - 25</div>
        <div class="option-sub">Young adult / Acne peak</div>
      </div>
      <div class="option-card" onclick="selectAge(this, '26 - 35')">
        <div class="option-icon">💼</div>
        <div class="option-label">26 - 35</div>
        <div class="option-sub">Adult skin & PIH care</div>
      </div>
      <div class="option-card" onclick="selectAge(this, '36 - 50')">
        <div class="option-icon">🌿</div>
        <div class="option-label">36 - 50</div>
        <div class="option-sub">Collagen & barrier defense</div>
      </div>
      <div class="option-card" onclick="selectAge(this, '50+')">
        <div class="option-icon">🌟</div>
        <div class="option-label">50+</div>
        <div class="option-sub">Deep lipid replenishment</div>
      </div>
    </div>

    <div class="wizard-actions">
      <button class="btn-back" onclick="goToPage('gender')">⬅ Back</button>
      <button class="btn-next" id="btn-next-age" disabled onclick="goToPage('concerns')">Continue to Concerns ➔</button>
    </div>
  </div>

  <!-- ==================== WIZARD PAGE 3: SKIN CONCERNS ==================== -->
  <div class="wizard-card" id="page-concerns" style="display:none;">
    <div class="wizard-title">What skin problems do you notice?</div>
    <div class="wizard-desc">Select any known concerns, or choose <em>"I don't know about my skin"</em> to let CutisAI diagnose you completely from scratch.</div>

    <div class="concern-grid">
      <div class="concern-card" onclick="toggleConcern(this, 'Excessive Oiliness & Shine')">
        <div class="concern-check">✓</div>
        <div>
          <div class="option-label">🛢️ Excess Oiliness & Grease</div>
          <div class="option-sub">Shiny forehead, nose or mid-day oil</div>
        </div>
      </div>

      <div class="concern-card" onclick="toggleConcern(this, 'Active Breakouts & Pimples')">
        <div class="concern-check">✓</div>
        <div>
          <div class="option-label">🔴 Active Pimples & Acne</div>
          <div class="option-sub">Red swollen bumps, whiteheads or cysts</div>
        </div>
      </div>

      <div class="concern-card" onclick="toggleConcern(this, 'Dryness & Flaking')">
        <div class="concern-check">✓</div>
        <div>
          <div class="option-label">🌵 Dryness & Flaking</div>
          <div class="option-sub">Tight feeling after wash, rough patches</div>
        </div>
      </div>

      <div class="concern-card" onclick="toggleConcern(this, 'Redness & Sensitivity')">
        <div class="concern-check">✓</div>
        <div>
          <div class="option-label">🛡️ Redness & Irritation</div>
          <div class="option-sub">Flushing, burning with products, reactive</div>
        </div>
      </div>

      <div class="concern-card" onclick="toggleConcern(this, 'Dark Spots & Pigmentation')">
        <div class="concern-check">✓</div>
        <div>
          <div class="option-label">🌑 Dark Spots & Marks</div>
          <div class="option-sub">Post-pimple brown spots, sun spots</div>
        </div>
      </div>

      <div class="concern-card" onclick="toggleConcern(this, 'Enlarged Pores & Blackheads')">
        <div class="concern-check">✓</div>
        <div>
          <div class="option-label">🕳️ Enlarged Pores</div>
          <div class="option-sub">Visible nose/cheek pores, blackheads</div>
        </div>
      </div>

      <!-- EXCLUSIVE OPTION: I DON'T KNOW ABOUT MY SKIN -->
      <div class="concern-card" id="card-unknown" onclick="selectUnknownSkin(this)" style="grid-column: 1 / -1; border-style:dashed;">
        <div class="concern-check">✓</div>
        <div>
          <div class="option-label" style="color:#FBBF24;">❓ I don't know about my skin</div>
          <div class="option-sub">Unsure about type or issues — run autonomous 4-shot computer-vision diagnosis!</div>
        </div>
      </div>
    </div>

    <div class="wizard-actions">
      <button class="btn-back" onclick="goToPage('age')">⬅ Back</button>
      <button class="btn-next" id="btn-next-concerns" disabled onclick="goToPage('scanner')">Proceed to 4-Shot Scan 📸</button>
    </div>
  </div>

  <!-- ==================== WIZARD PAGE 4: 4-SHOT CAMERA SCANNER ==================== -->
  <div id="page-scanner" style="display:none;">

    <!-- Patient Intake Summary Strip -->
    <div class="patient-badge">
      <div>
        <strong>Patient Profile:</strong>
        <span id="badge-gender-text">Female</span> • 
        <span id="badge-age-text">18-25</span> • 
        <span id="badge-concerns-text">Acne</span>
      </div>
      <button class="badge-edit-btn" onclick="goToPage('gender')">✏️ Edit Intake</button>
    </div>

    <!-- Stepper Bar for 4 camera shots -->
    <div class="stepper-bar">
      <div class="step-item active" id="step-0"><div class="step-num">Shot 1 of 4</div><div class="step-title">👤 Front View</div></div>
      <div class="step-item" id="step-1"><div class="step-num">Shot 2 of 4</div><div class="step-title">➡️ Right Profile</div></div>
      <div class="step-item" id="step-2"><div class="step-num">Shot 3 of 4</div><div class="step-title">⬅️ Left Profile</div></div>
      <div class="step-item" id="step-3"><div class="step-num">Shot 4 of 4</div><div class="step-title">🔼 Upper / Forehead</div></div>
    </div>

    <!-- Camera Viewport -->
    <div class="cam-card">
      <div class="vid-box">
        <video id="cam" autoplay playsinline muted></video>
        <div class="guide-overlay" id="guide"></div>
        <div class="guide-arrow arrow-right" id="arr-right">➔</div>
        <div class="guide-arrow arrow-left" id="arr-left">⬅</div>
        <div class="guide-arrow arrow-up" id="arr-up">⬆</div>
        <div class="flash" id="flash"></div>
        <div class="prompt-banner" id="prompt-banner">Pose 1: Look straight into camera</div>
      </div>

      <!-- Controls -->
      <div class="cam-controls">
        <button class="btn-snap" id="btn-snap">📸 Capture Front Shot</button>
        <button class="btn-upload" id="btn-upload" onclick="document.getElementById('file-upload-input').click()">📁 Upload Photo</button>
        <input type="file" id="file-upload-input" accept="image/*" onchange="handleFileUpload(event)">
        <button class="btn-reset" id="btn-reset">🔄 Reset Shots</button>
        <button class="btn-demo" id="btn-demo">⚡ Instant Demo</button>
      </div>

      <!-- 4 Thumbnails Strip -->
      <div class="thumb-strip">
        <div class="thumb-box" id="tb-0" onclick="jumpToStep(0)"><span class="thumb-placeholder">👤</span><img id="img-0"><span class="thumb-label">1. Front</span></div>
        <div class="thumb-box" id="tb-1" onclick="jumpToStep(1)"><span class="thumb-placeholder">➡️</span><img id="img-1"><span class="thumb-label">2. Right</span></div>
        <div class="thumb-box" id="tb-2" onclick="jumpToStep(2)"><span class="thumb-placeholder">⬅️</span><img id="img-2"><span class="thumb-label">3. Left</span></div>
        <div class="thumb-box" id="tb-3" onclick="jumpToStep(3)"><span class="thumb-placeholder">🔼</span><img id="img-3"><span class="thumb-label">4. Upper</span></div>
      </div>
    </div>

    <!-- Results Section -->
    <div class="results" id="results">
      <div style="font-size:0.8rem; color:var(--primary); font-weight:700; text-transform:uppercase;">Composite 4-Angle Dermatological Diagnosis</div>
      <div class="diag-title" id="diag-title">Analyzing Composite...</div>
      
      <!-- Autonomous Discovery or Correlated Banner -->
      <div class="discovery-banner" id="discovery-banner"></div>
      <div id="diag-desc" style="color:#94A3B8; font-size:0.95rem; margin-bottom:0.75rem;"></div>

      <div class="pimple-pill" id="pimple-pill">
        <span>🔴</span> <span id="pimple-status-text">0 Inflammatory Spots Detected</span>
      </div>

      <!-- Gauges Grid -->
      <div class="gauges">
        <div class="gauge"><div class="gauge-lbl">✨ Multi-View Oiliness</div><div class="gauge-val" id="g-oil">0%</div></div>
        <div class="gauge"><div class="gauge-lbl">💧 Composite Moisture</div><div class="gauge-val" id="g-moist">0%</div></div>
        <div class="gauge"><div class="gauge-lbl">🔍 Texture Clarity</div><div class="gauge-val" id="g-clarity">0%</div></div>
        <div class="gauge"><div class="gauge-lbl">🛡️ Vascular Redness</div><div class="gauge-val" id="g-redness">0%</div></div>
      </div>

      <!-- Zone Mapping -->
      <div class="zone-grid">
        <div class="zone-card">
          <strong style="color:#FFB800;">T-Zone Sebum (Forehead/Nose):</strong> <span id="val-tzone">0%</span>
        </div>
        <div class="zone-card">
          <strong style="color:#EC4899;">U-Zone Sebum (Cheeks):</strong> <span id="val-uzone">0%</span>
        </div>
      </div>

      <!-- Navigation Tabs -->
      <div class="tabs-nav">
        <button class="tab-btn active" onclick="switchTab(this, 'tab-meds')">💊 Pimple Medicine Guide</button>
        <button class="tab-btn" onclick="switchTab(this, 'tab-avoid')">🚫 What to Avoid (Personalized)</button>
        <button class="tab-btn" onclick="switchTab(this, 'tab-homecare')">🌿 100% Free Kitchen Remedies</button>
      </div>

      <!-- Tab 1: Pimple Medicine Guide -->
      <div class="tab-pane active" id="tab-meds">
        <div class="doctor-warn" id="doc-warn-text"></div>
        <h3 style="font-size:1.1rem; color:#FFF; margin-bottom:1rem;">Targeted Topical Spot Medicines (Direct on the Pimple)</h3>
        <div class="med-grid" id="med-grid"></div>
        <h3 style="font-size:1.1rem; color:#10B981; margin:1.5rem 0 1rem;">🌿 Zero-Cost Kitchen Spot Healers</h3>
        <div class="med-grid" id="natural-spot-grid"></div>
      </div>

      <!-- Tab 2: What to Avoid -->
      <div class="tab-pane" id="tab-avoid">
        <div class="avoid-grid" id="avoid-grid"></div>
      </div>

      <!-- Tab 3: Homecare -->
      <div class="tab-pane" id="tab-homecare">
        <div class="homecare-grid" id="homecare-grid"></div>
      </div>
    </div>

  </div>

</div>

<script>
/* ==================== STATE MANAGEMENT ==================== */
const userProfile = {
  gender: "",
  age_group: "",
  concerns: []
};

function selectGender(card, val) {
  document.querySelectorAll('#page-gender .option-card').forEach(c => c.classList.remove('selected'));
  card.classList.add('selected');
  userProfile.gender = val;
  document.getElementById('btn-next-gender').disabled = false;
}

function selectAge(card, val) {
  document.querySelectorAll('#page-age .option-card').forEach(c => c.classList.remove('selected'));
  card.classList.add('selected');
  userProfile.age_group = val;
  document.getElementById('btn-next-age').disabled = false;
}

function toggleConcern(card, val) {
  // If unknown skin was previously selected, remove it
  const unkCard = document.getElementById('card-unknown');
  unkCard.classList.remove('unknown-mode');
  const unkIdx = userProfile.concerns.indexOf("I don't know about my skin");
  if (unkIdx > -1) userProfile.concerns.splice(unkIdx, 1);

  card.classList.toggle('selected');
  const idx = userProfile.concerns.indexOf(val);
  if (idx > -1) {
    userProfile.concerns.splice(idx, 1);
  } else {
    userProfile.concerns.push(val);
  }
  document.getElementById('btn-next-concerns').disabled = (userProfile.concerns.length === 0);
}

function selectUnknownSkin(card) {
  // Clear all other selections
  document.querySelectorAll('.concern-card').forEach(c => {
    c.classList.remove('selected');
  });
  userProfile.concerns = ["I don't know about my skin"];
  card.classList.add('unknown-mode');
  document.getElementById('btn-next-concerns').disabled = false;
}

function updateProgressStepper(pageName) {
  const steps = ['gender', 'age', 'concerns', 'scanner'];
  const curIdx = steps.indexOf(pageName);
  for (let i = 0; i < 4; i++) {
    const sElem = document.getElementById('prog-' + i);
    sElem.classList.remove('active', 'done');
    if (i < curIdx) sElem.classList.add('done');
    else if (i === curIdx) sElem.classList.add('active');
  }
}

function goToPage(target) {
  const pages = ['gender', 'age', 'concerns', 'scanner'];
  pages.forEach(p => {
    const el = document.getElementById('page-' + p);
    if (el) el.style.display = (p === target) ? 'block' : 'none';
  });
  updateProgressStepper(target);

  if (target === 'scanner') {
    // Update summary badge
    document.getElementById('badge-gender-text').innerText = userProfile.gender || 'Not specified';
    document.getElementById('badge-age-text').innerText = userProfile.age_group || 'Not specified';
    document.getElementById('badge-concerns-text').innerText = (userProfile.concerns.length > 0) 
      ? userProfile.concerns.join(', ') 
      : "Autonomous AI Diagnosis";
    initCamera();
  }
}

/* ==================== 4-SHOT CAMERA LOGIC ==================== */
const angles = ["front", "right", "left", "top"];
const prompts = [
  "Pose 1/4: Look straight into camera (Front View)",
  "Pose 2/4: Turn head slightly to the right (~30° Right Cheek)",
  "Pose 3/4: Turn head slightly to the left (~30° Left Cheek)",
  "Pose 4/4: Tilt head slightly downward (Forehead / T-Zone View)"
];
const btnLabels = [
  "📸 Capture Front Shot",
  "📸 Capture Right Profile",
  "📸 Capture Left Profile",
  "📸 Capture Forehead Shot"
];

let currentStep = 0;
let capturedShots = {};
let camInitialized = false;
const vid = document.getElementById('cam');
const btnSnap = document.getElementById('btn-snap');
const promptBanner = document.getElementById('prompt-banner');
const flash = document.getElementById('flash');
const resultsBox = document.getElementById('results');

function initCamera() {
  if (camInitialized) return;
  camInitialized = true;
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })
      .then(s => { vid.srcObject = s; })
      .catch(e => {
        console.log('Camera error or permission denied:', e);
        promptBanner.innerText = "Camera unavailable. Use 'Upload Photo' or 'Instant Demo'.";
      });
  } else {
    promptBanner.innerText = "Webcam not supported. Use 'Upload Photo' or 'Instant Demo'.";
  }
}

function updatePoseGuide(step) {
  currentStep = step;
  if (step < 4) {
    promptBanner.innerText = prompts[step];
    btnSnap.innerText = btnLabels[step];
  } else {
    promptBanner.innerText = "All 4 views captured!";
    btnSnap.innerText = "⚡ Analyze 4 Views";
  }

  // Update stepper items
  for(let i=0; i<4; i++) {
    const sElem = document.getElementById('step-' + i);
    sElem.classList.remove('active', 'done');
    if (capturedShots[angles[i]]) sElem.classList.add('done');
    else if (i === step) sElem.classList.add('active');
  }

  // Directional guidance arrows
  document.getElementById('arr-right').style.display = step === 1 ? 'block' : 'none';
  document.getElementById('arr-left').style.display = step === 2 ? 'block' : 'none';
  document.getElementById('arr-up').style.display = step === 3 ? 'block' : 'none';
}

function jumpToStep(step) {
  updatePoseGuide(step);
}

function triggerFlash() {
  flash.style.opacity = '0.9';
  setTimeout(() => flash.style.opacity = '0', 200);
}

function handleFileUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = function(e) {
    const b64 = e.target.result;
    recordShot(b64);
  };
  reader.readAsDataURL(file);
  event.target.value = '';
}

function recordShot(b64) {
  const angleName = angles[currentStep];
  capturedShots[angleName] = b64;

  const thumbImg = document.getElementById('img-' + currentStep);
  thumbImg.src = b64;
  thumbImg.style.display = 'block';

  if (currentStep < 3) {
    updatePoseGuide(currentStep + 1);
  } else {
    updatePoseGuide(4);
    triggerMultishotAnalysis();
  }
}

btnSnap.onclick = async () => {
  if (currentStep >= 4) {
    triggerMultishotAnalysis();
    return;
  }

  triggerFlash();
  const c = document.createElement('canvas');
  c.width = vid.videoWidth || 640; 
  c.height = vid.videoHeight || 480;
  const ctx = c.getContext('2d');
  ctx.drawImage(vid, 0, 0, c.width, c.height);
  const b64 = c.toDataURL('image/jpeg', 0.88);

  recordShot(b64);
};

async function triggerMultishotAnalysis() {
  btnSnap.innerText = "⚡ Synthesizing 4-Angle Composite Model...";
  btnSnap.disabled = true;

  try {
    const res = await fetch('/api/analyze_multishot', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ 
        shots: capturedShots,
        user_profile: userProfile
      })
    });
    const data = await res.json();
    btnSnap.innerText = "✅ 4-Angle Scan Complete";
    btnSnap.disabled = false;
    if (data.success) {
      render(data);
    } else {
      alert(data.error || 'Analysis failed. Please check frame lighting.');
    }
  } catch(e) {
    btnSnap.innerText = "📸 Retake Scan";
    btnSnap.disabled = false;
    alert("Analysis error. Please check server terminal logs.");
  }
}

document.getElementById('btn-reset').onclick = () => {
  capturedShots = {};
  for(let i=0; i<4; i++) {
    const img = document.getElementById('img-' + i);
    img.src = '';
    img.style.display = 'none';
    const sElem = document.getElementById('step-' + i);
    sElem.classList.remove('done', 'active');
  }
  resultsBox.style.display = 'none';
  btnSnap.disabled = false;
  updatePoseGuide(0);
};

document.getElementById('btn-demo').onclick = async () => {
  const res = await fetch('/api/sample', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ user_profile: userProfile })
  });
  const data = await res.json();
  render(data);
};

function switchTab(btn, tabId) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
  if (btn) btn.classList.add('active');
  const target = document.getElementById(tabId);
  if (target) target.classList.add('active');
}

function render(d) {
  resultsBox.style.display = 'block';
  document.getElementById('diag-title').innerText = d.skin_type;
  document.getElementById('diag-desc').innerText = d.recommendations.clinical_diagnosis;

  // Correlated intake banner
  const discBanner = document.getElementById('discovery-banner');
  discBanner.innerText = d.recommendations.discovery_note || "Scan successfully processed.";

  document.getElementById('g-oil').innerText = Math.round(d.metrics.oiliness) + '%';
  document.getElementById('g-moist').innerText = Math.round(d.metrics.moisture) + '%';
  document.getElementById('g-clarity').innerText = Math.round(d.metrics.clarity) + '%';
  document.getElementById('g-redness').innerText = Math.round(d.metrics.redness) + '%';

  if (d.zones) {
    document.getElementById('val-tzone').innerText = d.zones.t_zone_oiliness + '%';
    document.getElementById('val-uzone').innerText = d.zones.u_zone_oiliness + '%';
  }

  const pCount = d.metrics.pimple_count || 0;
  document.getElementById('pimple-status-text').inn
}
</script>
</body>
</html>"""


app = Flask(__name__)
analyzer = SkinAnalyzer()


def build_response(analysis, user_profile):
    metrics = analysis.get("overall", analysis.get("metrics", {}))
    skin_type = analysis.get("skin_type", "Balanced / Normal Skin (Healthy Barrier)")
    response = dict(analysis)
    response["metrics"] = metrics
    response["recommendations"] = get_recommendations(skin_type, metrics, user_profile)
    return response


@app.get("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.post("/api/sample")
def sample_analysis():
    profile = request.get_json(silent=True) or {}
    analysis = {
        "success": True,
        "skin_type": "Combination Skin (Oily T-Zone / Dry Cheeks)",
        "shots_analyzed": 4,
        "angles_used": ["front", "right", "left", "top"],
        "metrics": {
            "oiliness": 62.0,
            "moisture": 68.0,
            "clarity": 78.0,
            "redness": 24.0,
            "pimple_count": 2,
        },
        "zones": {"t_zone_oiliness": 72.0, "u_zone_oiliness": 42.0},
    }
    return jsonify(build_response(analysis, profile.get("user_profile", profile)))


@app.post("/api/analyze_multishot")
def analyze_multishot():
    payload = request.get_json(silent=True) or {}
    shots = {}
    for angle, encoded in (payload.get("shots") or {}).items():
        try:
            image_data = encoded.split(",", 1)[-1]
            image_bytes = base64.b64decode(image_data)
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            if image is not None:
                shots[angle] = image
        except (ValueError, TypeError):
            continue

    analysis = analyzer.fuse_4_shots(shots)
    if not analysis.get("success"):
        return jsonify(analysis), 400
    return jsonify(build_response(analysis, payload.get("user_profile") or {}))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)