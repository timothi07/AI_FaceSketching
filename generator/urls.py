
from django.urls import path
from . import views
from .views import save_image
from .views import saved_images_view


app_name = 'generator'

urlpatterns = [
    path('', views.index, name='index'),
    path('generate/', views.generate_image, name='generate'),
    path('result/<int:image_id>/', views.result, name='result'),
    path('enhance/<int:image_id>/', views.enhance_image, name='enhance'),
    path('save/<int:image_id>/', save_image, name='save_image'),
    path('inbox/', saved_images_view, name='saved_images'),
    # path("save-image/<int:image_id>/", saved_images_view, name="save_image"),
]
