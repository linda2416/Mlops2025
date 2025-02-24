from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_view, name='index'),
    path('predict/', views.predict_view, name='predict'),
]
