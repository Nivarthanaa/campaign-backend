from flask_sqlalchemy import SQLAlchemy # type: ignore
from datetime import datetime
import json

db = SQLAlchemy()

class Campaign(db.Model):
    __tablename__ = 'campaigns'
    
    id = db.Column(db.String(20), primary_key=True) # e.g., CIG00001
    platform = db.Column(db.String(50))
    post_type = db.Column(db.String(50))
    ad_spend_usd = db.Column(db.Float)
    ad_duration_days = db.Column(db.Integer)
    daily_budget_usd = db.Column(db.Float)
    likes = db.Column(db.Integer, default=0)
    comments = db.Column(db.Integer, default=0)
    shares = db.Column(db.Integer, default=0)
    saves = db.Column(db.Integer, default=0)
    
    # New fields for ML validation
    most_engaged_gender = db.Column(db.String(50))
    most_engaged_age_group = db.Column(db.String(50))
    region = db.Column(db.String(50))
    device_type = db.Column(db.String(50))
    
    # Tracking results directly in Campaign for simplicity if desired, 
    # but a separate table is cleaner. Let's keep a relationship.
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    result = db.relationship('Result', backref='campaign', uselist=False, cascade="all, delete-orphan")

class Result(db.Model):
    __tablename__ = 'results'
    
    id = db.Column(db.Integer, primary_key=True)
    campaign_id = db.Column(db.String(20), db.ForeignKey('campaigns.id'), nullable=False)
    cluster_id = db.Column(db.Integer)
    prediction_label = db.Column(db.Integer)
    prediction_text = db.Column(db.String(50))
    prob_high = db.Column(db.Float)
    
    # Store complex objects as JSON strings for simplicity
    recommendations_json = db.Column(db.Text)
    shap_explanation_json = db.Column(db.Text)
    
    @property
    def recommendations(self):
        return json.loads(self.recommendations_json) if self.recommendations_json else []
    
    @recommendations.setter
    def recommendations(self, value):
        self.recommendations_json = json.dumps(value)

    @property
    def shap_explanation(self):
        return json.loads(self.shap_explanation_json) if self.shap_explanation_json else {}
    
    @shap_explanation.setter
    def shap_explanation(self, value):
        self.shap_explanation_json = json.dumps(value)
