import os, sys, json, logging, io
from flask import Flask, request, jsonify, send_file # type: ignore
from flask_cors import CORS # type: ignore
from typing import Optional, List, Dict, Any, cast

# Basic Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Core Imports
try:
    from ml_pipeline import get_schema, run_clustering, run_prediction, segment_profile, get_feature_importance, explain_prediction # type: ignore
    from report_gen import make_report # type: ignore
except ImportError as e:
    logger.error(f"Import Error: {str(e)}. Ensure ml_pipeline.py and report_gen.py are in the backend folder.") # type: ignore
    raise

from models import db, Campaign, Result # type: ignore
from datetime import datetime
from sqlalchemy.orm import joinedload # type: ignore
from sqlalchemy import select, delete, desc # type: ignore

REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

app = Flask(__name__)
# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(BASE_DIR, 'project.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

from models import Campaign, Result
from models import db
db.init_app(app)

with app.app_context():
    db.create_all()

# Robust CORS for development
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "DELETE", "OPTIONS", "PUT"]}})

LABELS = {0: "Low", 1: "Medium", 2: "High"}

def compute_daily_budget(payload: dict):
    try:
        if payload.get("daily_budget_usd") is None:
            spend = float(payload.get("ad_spend_usd") or 0)
            days = float(payload.get("ad_duration_days") or 1)
            payload["daily_budget_usd"] = round(spend / max(days, 1), 2) # type: ignore
    except (ValueError, TypeError):
        payload["daily_budget_usd"] = 0
    return payload

def save_campaign(payload: dict) -> str:
    platform = payload.get("platform", "Instagram")
    prefix = "CIG" if platform == "Instagram" else "CFB"
    
    # Robust Sequential ID Generation to avoid primary key collisions
    try:
        stmt = select(Campaign.id).where(Campaign.id.like(f"{prefix}%"))
        existing_ids = db.session.execute(stmt).all()
        current_max = 0
        for (eid,) in existing_ids:
            try:
                # Use list comprehension to avoid Pyre2 filter errors
                digits = [char for char in str(eid) if char.isdigit()]
                num_str = "".join(digits)
                if num_str:
                    val = int(num_str)
                    if val > current_max:
                        current_max = val
            except (ValueError, TypeError):
                continue
        
        new_num = int(current_max) + 1
        cid = f"{prefix}{new_num:05d}"
        
        # Absolute safety: ensure no collision if some non-numeric IDs exist
        while db.session.get(Campaign, cid):
            new_num += 1
            cid = f"{prefix}{new_num:05d}"
            
    except Exception as e:
        logger.error(f"Error in ID generation: {e}")
        # Fallback to a random ID using secrets to avoid slicing lint issues
        import secrets
        cid = f"{prefix}{secrets.token_hex(3).upper()}"
    
    # Save to Database
    new_campaign = Campaign(
        id=cid,
        platform=platform,
        post_type=payload.get("post_type"),
        ad_spend_usd=payload.get("ad_spend_usd"),
        ad_duration_days=payload.get("ad_duration_days"),
        daily_budget_usd=payload.get("daily_budget_usd"),
        likes=payload.get("likes", 0),
        comments=payload.get("comments", 0),
        shares=payload.get("shares", 0),
        saves=payload.get("saves", 0),
        most_engaged_gender=payload.get("most_engaged_gender"),
        most_engaged_age_group=payload.get("most_engaged_age_group"),
        region=payload.get("region"),
        device_type=payload.get("device_type")
    )
    db.session.add(new_campaign)
    db.session.commit()
    return cid

def load_campaign(campaignId: str) -> dict:
    c = db.session.get(Campaign, campaignId)
    if not c:
        raise FileNotFoundError(f"Campaign {campaignId} not found.")
    
    # Convert to dict for ML pipeline
    data = {
        "campaign_id": c.id,
        "platform": c.platform,
        "post_type": c.post_type,
        "ad_spend_usd": c.ad_spend_usd,
        "ad_duration_days": c.ad_duration_days,
        "daily_budget_usd": c.daily_budget_usd,
        "likes": c.likes,
        "comments": c.comments,
        "shares": c.shares,
        "saves": c.saves,
        "most_engaged_gender": c.most_engaged_gender,
        "most_engaged_age_group": c.most_engaged_age_group,
        "region": c.region,
        "device_type": c.device_type
    }
    if c.result:
        data.update({
            "clusterId": c.result.cluster_id,
            "predictionLabel": c.result.prediction_label,
            "predictionText": c.result.prediction_text,
            "prob_high": c.result.prob_high
        })
    return data

def update_campaign(campaignId: str, data: dict):
    c = db.session.get(Campaign, campaignId)
    if not c:
        return
    
    # Check if we are updating prediction results
    if "predictionLabel" in data or "prob_high" in data:
        if not c.result:
            c.result = Result(campaign_id=campaignId)
        
        if "clusterId" in data: c.result.cluster_id = data["clusterId"]
        if "predictionLabel" in data: c.result.prediction_label = data["predictionLabel"]
        if "predictionText" in data: c.result.prediction_text = data["predictionText"]
        if "prob_high" in data: c.result.prob_high = data["prob_high"]
        if "recommendations" in data: c.result.recommendations = data["recommendations"]
        if "shap_explanation" in data: c.result.shap_explanation = data["shap_explanation"]
    
    db.session.commit()

def generate_optimization_plan(payload: dict, cluster_id: int, pred_label: int, probs: List[float], profile: Optional[dict], explanation: Optional[dict] = None):
    prob_high = probs[2] if len(probs) >= 3 else max(probs)
    current_spend = float(payload.get("ad_spend_usd", 0))
    current_duration = int(payload.get("ad_duration_days", 7))
    
    # Dynamic tactical multipliers based on prediction and confidence
    # Instead of returning the exact same budget, we predict an optimized budget
    # Low (0): Significant pivot (+20%), Med (1): Moderate scaling (+15%), High (2): Aggressive scaling (+40%)
    base_mult = [1.20, 1.15, 1.40][pred_label]
    prob_bonus = (prob_high - 0.5) * 0.4 if prob_high is not None else 0
    budget_mult = max(1.05, base_mult + prob_bonus)
    
    duration_offset = [-2, 1, 4][pred_label]
    
    suggested_budget = int(round(current_spend * budget_mult))
    if suggested_budget == int(current_spend):
        suggested_budget += 50 # Ensure prediction is always distinct from input
        
    suggested_duration = int(max(3, current_duration + duration_offset))
    
    # Goal selection logic based on Platform and Performance
    platform = payload.get("platform", "Instagram")
    goals = ["Visit Profile", "Visit Your Website", "More Messages", "A Mix of Actions"]
    
    if pred_label == 0: # Low performance, needs objective shift
        goal = goals[3] if platform == "Instagram" else goals[1]
    elif pred_label == 1:
        goal = goals[2] if platform == "Facebook" else goals[0]
    else: # High performance, maintain & scale
        goal = goals[0] if platform == "Instagram" else goals[2]

    # Dynamic strings based on user input
    device = payload.get("device_type", "mobile").lower()
    region = payload.get("region", "target")
    post_type = payload.get("post_type", "Reel")

    # Data-driven improvement steps based on SHAP (explanation)
    improvement_steps = []
    if explanation:
        # Sort features by SHAP value (most negative first for improvement)
        items = list(explanation.items())
        sorted_features = sorted(items, key=lambda x: float(str(x[1])))
        # Even slightly negative features should trigger a tip if we don't have enough
        all_neg = [str(f) for f, v in sorted_features if float(str(v)) < -0.001]
        neg_features: List[str] = []
        for i, feat_name in enumerate(all_neg):
            if i < 3:
                neg_features.append(feat_name)
        
        feature_tips = {
            "likes": "Your 'Likes' count is a performance detractor. Focus on more engaging visual hooks or trending audio.",
            "comments": "Low comment volume is hurting visibility. Try using direct questions or CTA stickers to spark conversation.",
            "shares": "Content shareability is low. Aim for 'Relatable' or 'Educational' value to drive organic reach.",
            "saves": "Save rate is below benchmark. Create 'Value-add' content (checklists, tutorials) that users want to bookmark.",
            "ad_spend_usd": "Your current budget allocation is sub-optimal for your target reach. Consider the suggested increase.",
            "ad_duration_days": "The campaign duration is misaligned with audience behavior in this segment.",
            "platform": f"Platform-specific engagement on {platform} is weak. Re-align your creative with {platform}-native trends.",
            "post_type": f"The {post_type} format is currently underperforming. Test a different content style soon.",
            "cluster_id": "Your campaign is in a low-performing audience cluster. Consider a fundamental creative pivot."
        }
        
        for feat in neg_features:
            if feat in feature_tips:
                improvement_steps.append(feature_tips[feat])

    # Fallback/Default steps if no SHAP or few negative features
    default_steps = [
        f"Optimize your Call-to-Action for {device} users to drive higher conversion.",
        "Schedule posts during identified peak engagement windows (6 PM - 10 PM).",
        "Leverage lookalike audience segments to reach users similar to your top leads."
    ]
    for step in default_steps:
        if len(improvement_steps) < 3:
            improvement_steps.append(step)

    # Explicitly handle prob_high to satisfy strict linters
    safe_prob: float = 0.0
    if prob_high is not None:
        try:
            safe_prob = float(prob_high)
        except (ValueError, TypeError):
            safe_prob = 0.0

    # Manual rounding to avoid built-in function lint issues
    confidence_score: float = float(int(safe_prob * 1000) / 1000.0)
    
    status_label = str(LABELS.get(int(pred_label), "Unknown"))
    
    plan: Dict[str, Any] = {
        "status": status_label,
        "confidence": confidence_score, # type: ignore
        "next_campaign": {
            "budget_recommendation": f"Next Optimization: ${suggested_budget}",
            "duration_days": suggested_duration,
            "tactical_goal": goal,
            "suggested_daily_spend": int(round(suggested_budget / suggested_duration)),
            "strategic_pivot": [
                f"Identify and pivot: Your current strategy for {post_type} in {region} needs a creative overhaul.",
                f"Stability and Scale: Maintain {post_type} engagement while gradually scaling budget.",
                f"High-Impact Expansion: Aggressively scale {post_type} in {region} to capture peak audience intent."
            ][int(pred_label)],
            "improvement_steps": improvement_steps
        }
    }

    if profile is not None:
        p_dict = cast(Dict[str, Any], profile)
        plan["segment_insight"] = f"Top Platform for your segment: {p_dict.get('top_platform', 'N/A')}"
        plan["audience_logic"] = f"Strategy focus: {payload.get('most_engaged_age_group', 'N/A')} on {payload.get('device_type', 'N/A')} (aligned with your current targets)."

    return plan

def recommendations(payload: dict, cluster_id: int, pred_label: int, probs: List[float], profile: Optional[dict], explanation: Optional[dict] = None):
    plan = generate_optimization_plan(payload, cluster_id, pred_label, probs, profile, explanation)
    
    # Cast values to satisfy linter indexing into a Mixed dictionary
    confidence = cast(float, plan['confidence'])
    next_campaign = cast(Dict[str, Any], plan['next_campaign'])
    
    recs = [
        f"Predicted engagement: {plan['status']} ({int(confidence*100)}% confidence).",
        f"Strategy: {next_campaign['budget_recommendation']}.",
        f"Next Suggested Daily: ${next_campaign['suggested_daily_spend']}."
    ]
    recs.extend(next_campaign['improvement_steps'][:2])
    return recs

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "port": 8080})

@app.route("/")
def home():
    return "Backend running ✅"

@app.route("/schema", methods=["GET"])
def schema():
    return jsonify(get_schema())

@app.route("/campaign", methods=["POST"])
def create_campaign():
    try:
        payload = request.get_json(force=True)
        payload = compute_daily_budget(payload)
        
        # Casting
        payload["ad_spend_usd"] = float(payload.get("ad_spend_usd", 0))
        payload["ad_duration_days"] = int(payload.get("ad_duration_days", 1))
        payload["daily_budget_usd"] = float(payload.get("daily_budget_usd", 0))
        if payload.get("likes") is None:
            return jsonify({"error": "Likes count is mandatory."}), 400

        payload["likes"] = int(payload.get("likes", 0))
        payload["comments"] = int(payload.get("comments", 0))
        payload["shares"] = int(payload.get("shares", 0))
        payload["saves"] = int(payload.get("saves", 0))

        if any(v < 0 for v in [payload["likes"], payload["comments"], payload["shares"], payload["saves"]]):
            return jsonify({"error": "Metrics cannot be negative."}), 400

        campaignId = save_campaign(payload)
        return jsonify({"campaignId": campaignId, "message": "Campaign saved successfully"})
    except Exception as e:
        logger.exception("Error in /campaign")
        return jsonify({"error": str(e)}), 400

@app.route("/segment", methods=["POST"])
def segment():
    try:
        data = request.get_json(force=True)
        campaignId = data.get("campaignId")
        if not campaignId: return jsonify({"error": "campaignId required"}), 400
            
        payload = load_campaign(campaignId)
        cluster_id = run_clustering(payload)
        profile = segment_profile(cluster_id)

        return jsonify({
            "campaignId": campaignId,
            "clusterId": cluster_id,
            "segmentProfile": profile
        })
    except Exception as e:
        logger.exception("Error in /segment")
        return jsonify({"error": str(e)}), 400

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        campaignId = data.get("campaignId")
        if not campaignId: return jsonify({"error": "campaignId required"}), 400
            
        payload = load_campaign(campaignId)
        cluster_id = run_clustering(payload)
        profile = segment_profile(cluster_id)
        pred, probs = run_prediction(payload, cluster_id)
        
        # Explain prediction for charts and strategy accuracy
        explanation = explain_prediction(payload, cluster_id)
        
        recs = recommendations(payload, cluster_id, pred, probs, profile, explanation)
        
        # Persist results for tracking
        p_list = cast(List[float], probs)
        prob_high = p_list[2] if len(p_list) >= 3 else (max(p_list) if p_list else 0.0)
        update_campaign(campaignId, {
            "predictionLabel": pred,
            "predictionText": str(LABELS.get(pred, "Unknown")),
            "prob_high": round(float(prob_high), 3), # type: ignore
            "shap_explanation": explanation,
            "recommendations": recs
        })

        return jsonify({
            "campaignId": campaignId,
            "clusterId": cluster_id,
            "predictionLabel": pred,
            "predictionText": LABELS.get(pred, "Unknown"),
            "probabilities": probs,
            "recommendations": recs,
            "inputs": {
                "likes": payload.get("likes", 0),
                "comments": payload.get("comments", 0),
                "shares": payload.get("shares", 0),
                "saves": payload.get("saves", 0)
            }
        })
    except Exception as e:
        logger.exception("Error in /predict")
        return jsonify({"error": str(e)}), 400

@app.route("/dashboard", methods=["GET"])
def dashboard_route():
    try:
        campaignId = request.args.get("campaignId")
        if not campaignId: return jsonify({"error": "campaignId required"}), 400
            
        payload = load_campaign(campaignId)
        cluster_id = run_clustering(payload)
        profile = segment_profile(cluster_id)
        pred, probs = run_prediction(payload, cluster_id)
        explanation = explain_prediction(payload, cluster_id)
        p_list = cast(List[float], probs)
        prob_high = p_list[2] if len(p_list) >= 3 else (max(p_list) if p_list else 0.0)

        return jsonify({
            "campaignId": campaignId,
            "kpis": {
                "platform": payload.get("platform"),
                "post_type": payload.get("post_type"),
                "ad_spend_usd": int(round(float(payload.get("ad_spend_usd", 0)))),
                "ad_duration_days": payload.get("ad_duration_days"),
                "daily_budget_usd": int(round(float(payload.get("daily_budget_usd", 0)))),
                "likes": payload.get("likes", 0),
                "comments": payload.get("comments", 0),
                "shares": payload.get("shares", 0),
                "saves": payload.get("saves", 0),
                "region": payload.get("region"),
                "device_type": payload.get("device_type"),
                "most_engaged_gender": payload.get("most_engaged_gender"),
                "most_engaged_age_group": payload.get("most_engaged_age_group"),
                "clusterId": cluster_id,
                "predicted_engagement": str(LABELS.get(pred, "Unknown")),
                "prob_high": round(float(prob_high), 3) # type: ignore
            },
            "segmentProfile": profile
        })
    except Exception as e:
        logger.exception("Error in /dashboard")
        return jsonify({"error": str(e)}), 400

@app.route("/report", methods=["GET"])
def report():
    try:
        campaignId = request.args.get("campaignId")
        if not campaignId: return jsonify({"error": "campaignId required"}), 400
        
        c = db.session.get(Campaign, campaignId)
        if not c: return jsonify({"error": f"Campaign {campaignId} not found."}), 404
        
        payload = load_campaign(campaignId)
        cluster_id = run_clustering(payload)
        profile = segment_profile(cluster_id)
        pred, probs = run_prediction(payload, cluster_id)
        explanation = explain_prediction(payload, cluster_id)
        plan = generate_optimization_plan(payload, cluster_id, pred, probs, profile, explanation)

        # Cleanup old reports: Delete files older than 1 hour
        try:
            import time
            now = time.time()
            for filename in os.listdir(REPORTS_DIR):
                filepath = os.path.join(REPORTS_DIR, filename)
                # Cleanup both .pdf and .png leftovers older than 1 hour
                if os.path.isfile(filepath) and (filename.endswith(".pdf") or filename.endswith(".png")):
                    # If file is older than 3600 seconds (1 hour)
                    if os.stat(filepath).st_mtime < now - 3600:
                        os.remove(filepath)
        except Exception as e:
            logger.warning(f"Failed to cleanup old reports: {e}")

        import secrets
        report_id = secrets.token_hex(8)
        out_path = os.path.join(REPORTS_DIR, f"report_{report_id}.pdf")
        # Technical key mapping for clean display
        ui_labels = {
            "platform": "Primary Platform",
            "post_type": "Content Format",
            "ad_spend_usd": "Total Budget (USD)",
            "ad_duration_days": "Duration (Days)",
            "daily_budget_usd": "Daily Allocation",
            "likes": "Target Engagement (Likes)",
            "comments": "Target Engagement (Comments)",
            "shares": "Target Engagement (Shares)",
            "saves": "Target Engagement (Saves)",
            "most_engaged_gender": "Dominant Gender",
            "most_engaged_age_group": "Target Age Group",
            "region": "Target Region",
            "device_type": "Primary Device"
        }

        # Filter and rename payload items
        clean_context = {ui_labels.get(k, k.replace("_", " ").title()): str(v) for k, v in payload.items() if k in ui_labels}

        make_report(
            out_path,
            "Campaign Optimization Strategy",
            [
                ("Executive Summary", [
                    f"This strategic report details the AI-driven optimization roadmap for Campaign {campaignId}.",
                    f"Our model predicts a '{LABELS.get(pred, 'Unknown')}' engagement tier with {int(plan['confidence']*100)}% confidence.",
                    f"The campaign has been categorized into Performance Cluster {cluster_id}."
                ]),
                ("Campaign Parameters (Your Input)", clean_context),
                ("Market Segment Benchmarks (Similar Campaigns)", {
                    "Segment Primary Platform": profile.get("top_platform", "N/A"),
                    "Segment Primary Device": profile.get("top_device", "N/A"),
                    "Benchmark Segment Size": f"{profile.get('size', 0)} campaigns",
                    "Avg Engagement Score": f"{profile.get('avg_engagement_score', 0):.2f}"
                }),
                ("Strategic Optimization Blueprint", {
                    "Tactical Goal": plan["next_campaign"]["tactical_goal"],
                    "Budget Recommendation": plan["next_campaign"]["budget_recommendation"],
                    "Ideal Duration": f"{plan['next_campaign']['duration_days']} Days",
                    "Strategic Pivot": plan["next_campaign"]["strategic_pivot"]
                }),
                ("Tactical Implementation Steps", plan["next_campaign"]["improvement_steps"]),
                ("Note", ["For security and storage optimization, this physically cached PDF file is permanently purged from the server after 1 hour."])
            ]
        )
        return send_file(out_path, as_attachment=True)
    except Exception as e:
        logger.exception("Error in /report")
        return jsonify({"error": str(e)}), 400

@app.route("/optimize", methods=["GET"])
def optimize_route():
    try:
        campaignId = request.args.get("campaignId")
        if not campaignId: return jsonify({"error": "campaignId required"}), 400
        
        payload = load_campaign(campaignId)
        cluster_id = run_clustering(payload)
        profile = segment_profile(cluster_id)
        pred, probs = run_prediction(payload, cluster_id)
        explanation = explain_prediction(payload, cluster_id)
        
        plan = generate_optimization_plan(payload, cluster_id, pred, probs, profile, explanation)
        return jsonify(plan)
    except Exception as e:
        logger.exception("Error in /optimize")
        return jsonify({"error": str(e)}), 400

@app.route("/xai/global", methods=["GET"])
def xai_global():
    try:
        importance = get_feature_importance()
        return jsonify(importance)
    except Exception as e:
        logger.exception("Error in /xai/global")
        return jsonify({"error": str(e)}), 400

@app.route("/xai/local", methods=["GET"])
def xai_local():
    try:
        campaignId = request.args.get("campaignId")
        if not campaignId: return jsonify({"error": "campaignId required"}), 400
        
        payload = load_campaign(campaignId)
        cluster_id = run_clustering(payload)
        explanation = explain_prediction(payload, cluster_id)
        return jsonify(explanation)
    except Exception as e:
        logger.exception("Error in /xai/local")
        return jsonify({"error": str(e)}), 400

@app.route("/campaigns", methods=["GET"])
def list_campaigns():
    try:
        # Use joinedload to prevent N+1 queries and remove the inner join
        # so that campaigns without results yet still show up in history.
        stmt = select(Campaign).options(joinedload(Campaign.result)).order_by(desc(Campaign.created_at))
        campaigns = db.session.execute(stmt).unique().scalars().all()
        
        result_list = []
        for c in campaigns:
            data = {
                "id": c.id,
                "campaign_id": c.id,
                "platform": c.platform,
                "post_type": c.post_type,
                "ad_spend_usd": c.ad_spend_usd,
                "ad_duration_days": c.ad_duration_days,
                "likes": c.likes,
                "comments": c.comments,
                "shares": c.shares,
                "saves": c.saves,
                "predictionLabel": c.result.prediction_label if c.result else None,
                "predictionText": c.result.prediction_text if c.result else None,
                "prob_high": c.result.prob_high if c.result else None
            }
            result_list.append(data)
        
        return jsonify(result_list)
    except Exception as e:
        logger.exception("Error in /campaigns")
        return jsonify({"error": str(e)}), 400

@app.route("/campaign/<campaignId>", methods=["DELETE"])
def delete_campaign(campaignId):
    try:
        c = db.session.get(Campaign, campaignId)
        if c:
            db.session.delete(c)
            db.session.commit()
            return jsonify({"message": f"Campaign {campaignId} deleted"})
        return jsonify({"error": "Campaign not found"}), 404
    except Exception as e:
        logger.exception(f"Error deleting campaign {campaignId}")
        return jsonify({"error": str(e)}), 400

@app.route("/campaigns/all", methods=["DELETE"])
def clear_all_campaigns():
    try:
        # Delete all campaigns (cascades will handle results)
        num_deleted = db.session.execute(delete(Campaign)).rowcount
        db.session.commit()
        return jsonify({"message": f"Successfully cleared all campaigns"})
    except Exception as e:
        logger.exception("Error clearing all campaigns")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    # Pre-load ML models at startup to avoid delay on the first request
    with app.app_context():
        try:
            from ml_pipeline import load_all # type: ignore
            load_all()
            logger.info("ML Models and Schema loaded successfully at startup.")
        except Exception as e:
            logger.error(f"Failed to pre-load ML models: {e}")

    # Disable debug mode to prevent reloader from restarting the server on DB updates
    import os

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port, debug=False)


