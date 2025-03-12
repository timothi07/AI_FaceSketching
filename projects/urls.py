from django.urls import path
from . import views

from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.projects, name="projects"),
    path('single-project/<str:pk>/', views.singleproject, name="project"),
    path('create-project/', views.createproject, name="create-project"),
    path('update-project/<str:pk>/', views.updateproject, name="update-project"),
    path('delete-project/<str:pk>/', views.deleteproject, name="delete-project"),
]

urlpatterns += static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)
urlpatterns += static(settings.STATIC_URL, document_root = settings.STATIC_ROOT)