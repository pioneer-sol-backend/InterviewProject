import os
import io
import json
import time
import asyncio
import traceback
from collections import defaultdict
from typing import List, Dict, Any, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from transformers import pipeline
import torch

# ========== LOAD .ENV ==========
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(env_path)

# ========== CREATE APP WITH CORS ==========
app = FastAPI(
    title="AgriAI API - Local Model",
    description="Plant disease detection API using local transformers models",
    version="1.0.0"
)

# CORS - allows browser to call API from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== CONFIGURATION ==========
class Config:
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    WEATHER_KEY = os.getenv("WEATHER_API_KEY", "")
    WEATHER_URL = "https://api.openweathermap.org/data/2.5/forecast"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_URL = "https://api.openai.com/v1/chat/completions"
    MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
    DEVICE = -1  # -1 = CPU, 0 = GPU

    MODEL_OPTIONS = [
        {
            "name": "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
            "task": "image-classification",
            "expected_size_mb": 50
        },
        {
            "name": "microsoft/resnet-50",
            "task": "image-classification",
            "expected_size_mb": 180
        },
        {
            "name": "google/vit-base-patch16-224",
            "task": "image-classification",
            "expected_size_mb": 330
        }
    ]


config = Config()
os.makedirs(config.MODEL_CACHE_DIR, exist_ok=True)

# ========== GLOBAL VARIABLES ==========
classifier = None
loaded_model_info = None


# ========== MODEL LOADING ==========
def load_plant_disease_model():
    """Load plant disease model with fallback options"""
    print("\n" + "=" * 60)
    print("🚀 INITIALIZING PLANT DISEASE MODEL")
    print("=" * 60)
    print(f"🔧 Device: {'CPU' if config.DEVICE == -1 else 'GPU'}")
    print(f"📁 Cache: {config.MODEL_CACHE_DIR}")

    for idx, model_config in enumerate(config.MODEL_OPTIONS, 1):
        model_name = model_config["name"]
        task = model_config["task"]

        print(f"\n📥 [{idx}/{len(config.MODEL_OPTIONS)}] Loading: {model_name}")

        try:
            os.environ['HF_HOME'] = config.MODEL_CACHE_DIR
            os.environ['TRANSFORMERS_CACHE'] = config.MODEL_CACHE_DIR

            start_time = time.time()
            pipeline_instance = pipeline(
                task,
                model=model_name,
                device=config.DEVICE,
                model_kwargs={"cache_dir": config.MODEL_CACHE_DIR}
            )
            load_time = time.time() - start_time

            # Test with dummy image
            test_img = Image.new('RGB', (224, 224), color='green')
            test_result = pipeline_instance(test_img)

            print(f"   ✅ Model loaded! Test: {test_result[0]['label']} ({test_result[0]['score']:.2%})")
            print(f"   ⏱️ Load time: {load_time:.2f}s")

            model_info = {
                "name": model_name,
                "task": task,
                "load_time_seconds": round(load_time, 2),
                "test_confidence": round(test_result[0]['score'], 3)
            }

            return pipeline_instance, model_info

        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:100]}")
            continue

    print("\n❌ CRITICAL: Could not load any model!")
    return None, None


# Load model at startup
classifier, loaded_model_info = load_plant_disease_model()

if classifier:
    print("\n✅ API READY - Model loaded!")
else:
    print("\n⚠️ API STARTING IN LIMITED MODE")


# ========== SERVICES ==========
async def analyze_image_local(image_bytes: bytes) -> List[Dict]:
    """Run plant disease classification locally"""
    if classifier is None:
        raise ValueError("Model not loaded. Check server logs.")

    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize if too large
        max_size = 800
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        predictions = classifier(image)

        return [
            {
                "label": pred["label"],
                "confidence": round(pred["score"], 4),
                "confidence_percent": f"{pred['score'] * 100:.1f}%"
            }
            for pred in predictions
        ]

    except Exception as e:
        raise ValueError(f"Image analysis failed: {str(e)}")


async def get_weather(lat: float, lon: float) -> List[Dict]:
    """Get 5-day weather forecast from OpenWeather"""
    if not config.WEATHER_KEY:
        raise ValueError("WEATHER_API_KEY not found in .env")

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            config.WEATHER_URL,
            params={
                "lat": lat,
                "lon": lon,
                "appid": config.WEATHER_KEY,
                "units": "metric"
            }
        )
        response.raise_for_status()
        data = response.json()

        daily = defaultdict(lambda: {"temps": [], "rain": [], "humidity": []})

        for item in data["list"]:
            date = item["dt_txt"][:10]
            daily[date]["temps"].append(item["main"]["temp"])
            daily[date]["humidity"].append(item["main"]["humidity"])
            daily[date]["rain"].append(item.get("rain", {}).get("3h", 0))

        result = []
        for date, vals in sorted(daily.items()):
            result.append({
                "date": date,
                "temp_max": round(max(vals["temps"]), 1),
                "temp_min": round(min(vals["temps"]), 1),
                "humidity": round(sum(vals["humidity"]) / len(vals["humidity"])),
                "rainfall_mm": round(sum(vals["rain"]), 1),
                "rain_chance": 100 if sum(vals["rain"]) > 5 else (50 if sum(vals["rain"]) > 0 else 0),
                "condition": "Rainy" if sum(vals["rain"]) > 5 else "Cloudy" if sum(vals["rain"]) > 0 else "Clear"
            })

        return result[:5]


def generate_fallback_plan(disease: str, confidence: float, weather: List[Dict], crop_type: str = "unknown") -> Dict:
    """Rule-based fallback recommendations"""
    urgent = confidence > 0.8 or "blight" in disease.lower() or "mildew" in disease.lower()

    warnings = []
    if urgent:
        warnings.append(f"⚠️ URGENT: {disease} detected with {confidence:.1%} confidence")

    actions = []
    for i, day in enumerate(weather):
        if i == 0 and urgent:
            task = f"🚨 IMMEDIATE: Apply emergency treatment for {disease}"
            reason = "Critical detection — immediate action required"
            chemical = "Mancozeb 75% WP @ 2.5g/L water OR Copper oxychloride 50% WP @ 3g/L"
            urgent_flag = True
        elif day["rain_chance"] > 70:
            task = "🌧️ Post-rain inspection + preventive spray"
            reason = f"Heavy rain expected ({day['rain_chance']}%). High humidity increases disease pressure"
            chemical = "Apply protective fungicide (Chlorothalonil 75% WP @ 2g/L) before rain"
            urgent_flag = False
        elif day["temp_max"] > 35:
            task = "💧 Increase irrigation, suspend chemical application"
            reason = f"Extreme heat ({day['temp_max']}°C). Heat stress + chemicals = potential crop damage"
            chemical = None
            urgent_flag = False
        else:
            task = "👁️ Routine monitoring and scouting"
            reason = "Normal conditions. Monitor for disease progression"
            chemical = None
            urgent_flag = False

        actions.append({
            "date": day["date"],
            "task": task,
            "reasoning": reason,
            "is_urgent": urgent_flag,
            "chemical_recommendation": chemical,
            "irrigation_adjustment": "Increase to 2x daily" if day["temp_max"] > 35 else None
        })

    risk = "HIGH" if urgent else "MODERATE" if confidence > 0.6 else "LOW"

    return {
        "diagnosis": disease.replace("_", " ").replace("-", " "),
        "confidence": confidence,
        "overall_risk": risk,
        "weather_summary": f"{len(weather)}-day forecast analyzed",
        "warnings": warnings,
        "daily_actions": actions,
        "recommendation_source": "rule-based (fallback)"
    }


async def generate_plan_with_openai(disease: str, confidence: float, weather: List[Dict], crop_type: str) -> Dict:
    """Generate recommendations using OpenAI"""
    if not config.OPENAI_API_KEY:
        return generate_fallback_plan(disease, confidence, weather, crop_type)

    weather_summary = "\n".join([
        f"Day {i + 1} ({day['date']}): {day['temp_max']}°C max, {day['rain_chance']}% rain"
        for i, day in enumerate(weather)
    ])

    prompt = f"""You are an expert agronomist.

CROP: {crop_type}
DISEASE: {disease} (confidence: {confidence:.1%})
WEATHER:
{weather_summary}

Generate a 5-day action plan as valid JSON with: diagnosis, confidence, overall_risk, weather_summary, warnings, daily_actions (date, task, reasoning, is_urgent, chemical_recommendation, irrigation_adjustment), recommendation_source."""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                config.OPENAI_URL,
                headers={
                    "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are an agricultural expert. Return only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 1500
                }
            )
            response.raise_for_status()
            data = response.json()
            ai_content = data["choices"][0]["message"]["content"]

            if "```json" in ai_content:
                ai_content = ai_content.split("```json")[1].split("```")[0].strip()
            elif "```" in ai_content:
                ai_content = ai_content.split("```")[1].split("```")[0].strip()

            return json.loads(ai_content)

    except Exception as e:
        print(f"⚠️ OpenAI failed: {e}")
        return generate_fallback_plan(disease, confidence, weather, crop_type)


# ========== ENDPOINTS ==========
@app.get("/")
async def root():
    return {
        "name": "AgriAI API",
        "version": "1.0.0",
        "status": "operational" if classifier else "limited",
        "model_loaded": classifier is not None,
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if classifier else "degraded",
        "model_loaded": classifier is not None,
        "model_info": loaded_model_info,
        "weather_api": "configured" if config.WEATHER_KEY else "missing",
        "openai_api": "configured" if config.OPENAI_API_KEY else "missing (fallback)"
    }


@app.post("/analyze")
async def analyze(
        file: UploadFile = File(...),
        crop_type: str = Form(...),
        lat: float = Form(...),
        lon: float = Form(...)
):
    """Main endpoint: Upload crop image for disease detection + weather + action plan"""
    start_time = time.time()

    print(f"\n📥 REQUEST: crop={crop_type}, lat={lat}, lon={lon}, file={file.filename}")

    try:
        # Validate
        if not file.content_type or not file.content_type.startswith("image/"):
            raise HTTPException(400, "File must be an image")

        image_bytes = await file.read()
        print(f"📁 File size: {len(image_bytes) / 1024:.1f} KB")

        if len(image_bytes) > config.MAX_IMAGE_SIZE:
            raise HTTPException(400, f"Image too large. Max {config.MAX_IMAGE_SIZE // 1024 // 1024}MB")

        # Step 1: Analyze image
        print("🔬 Running disease detection...")
        if not classifier:
            raise HTTPException(503, "Model not loaded")

        img_results = await analyze_image_local(image_bytes)
        top_result = img_results[0]
        print(f"✅ Detected: {top_result['label']} ({top_result['confidence_percent']})")

        # Step 2: Weather
        print("🌤️ Fetching weather...")
        weather = await get_weather(lat, lon)
        print(f"✅ Weather: {len(weather)} days")

        # Step 3: Recommendations
        print("📋 Generating plan...")
        if config.OPENAI_API_KEY:
            plan = await generate_plan_with_openai(
                disease=top_result["label"],
                confidence=top_result["confidence"],
                weather=weather,
                crop_type=crop_type
            )
        else:
            plan = generate_fallback_plan(
                disease=top_result["label"],
                confidence=top_result["confidence"],
                weather=weather,
                crop_type=crop_type
            )

        processing_time = round((time.time() - start_time) * 1000)
        print(f"✅ Done in {processing_time}ms")

        return {
            "success": True,
            "crop_type": crop_type,
            "detection": {
                "disease": top_result["label"],
                "confidence": top_result["confidence"],
                "confidence_percent": top_result["confidence_percent"],
                "top_predictions": img_results[:3]
            },
            "weather_forecast": weather,
            "action_plan": plan,
            "processing_time_ms": processing_time,
            "model_used": loaded_model_info["name"] if loaded_model_info else "unknown"
        }

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.detail, "status_code": exc.status_code}
    )


# ========== RUN ==========
if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("🚀 STARTING AGRI-AI API SERVER")
    print("=" * 60)
    print("📍 http://127.0.0.1:8000")
    print("📚 http://127.0.0.1:8000/docs")
    print("❤️ http://127.0.0.1:8000/health")
    print("=" * 60 + "\n")

    uvicorn.run(app, host="127.0.0.1", port=8000)