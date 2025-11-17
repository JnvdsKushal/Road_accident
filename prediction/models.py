from django.db import models
from django.contrib.auth.hashers import make_password, check_password

class PredictionLog(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    input_json = models.JSONField()
    predicted_code = models.IntegerField()
    predicted_label = models.CharField(max_length=32)
    sms_sent = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.timestamp} - {self.predicted_label}"

class CustomUser(models.Model):
    """Custom user model for storing registered users"""
    name = models.CharField(max_length=255)
    email = models.EmailField(unique=True)
    password_hash = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def set_password(self, raw_password):
        """Hash and store password securely"""
        self.password_hash = make_password(raw_password)
    
    def check_password(self, raw_password):
        """Verify password against stored hash"""
        return check_password(raw_password, self.password_hash)
    
    def __str__(self):
        return f"{self.name} ({self.email})"

class RiskZone(models.Model):
    """Model to store risk zone data"""
    state = models.CharField(max_length=100)
    latitude = models.FloatField()
    longitude = models.FloatField()
    accidents = models.IntegerField()
    risk_level = models.CharField(max_length=20)
    cluster = models.IntegerField()
    
    class Meta:
        ordering = ['risk_level', 'accidents']
    
    def __str__(self):
        return f"{self.state} - {self.risk_level}"