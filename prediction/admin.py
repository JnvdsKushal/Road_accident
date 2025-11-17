from django.contrib import admin
from .models import PredictionLog, CustomUser

@admin.register(PredictionLog)
class PredictionLogAdmin(admin.ModelAdmin):
    list_display = ('timestamp','predicted_label','predicted_code','sms_sent')
    list_filter = ('predicted_label','sms_sent')
    search_fields = ('predicted_label',)

@admin.register(CustomUser)
class CustomUserAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'created_at')
    search_fields = ('name', 'email')
    readonly_fields = ('password_hash', 'created_at', 'updated_at')