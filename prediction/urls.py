from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('analytics/', views.analytics, name='analytics'),
    path('safe_route/', views.safe_route_view, name='safe_route'),
    path('alerts/', views.alerts, name='alerts'),
    path('api/risk-zones/', views.risk_zones, name='risk_zones'),
    path('predict/', views.predict_page, name='prediction'),
    path('predict/json/', views.predict_json, name='predict_json'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path("api/compute-routes/", views.compute_routes, name="compute_routes"),
    path("debug-data/", views.debug_data),
    path("api/geocode/", views.geocode_proxy, name="geocode_proxy"),
    path("proxy-search/", views.nominatim_proxy, name="proxy_search"),
]
