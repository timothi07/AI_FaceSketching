
from django.urls import path
from . import views

app_name = 'generator'

urlpatterns = [
    path('', views.index, name='index'),
    path('generate/', views.generate_image, name='generate'),
    path('result/<int:image_id>/', views.result, name='result'),
    path('enhance/<int:image_id>/', views.enhance_image, name='enhance'),
]

# from django.urls import path
# from . import views
# from django.conf import settings
# from django.conf.urls.static import static

# urlpatterns = [
#     path('', views.generate_image, name='generate_image'),
#     path('enhance/<int:image_id>/', views.enhance_image, name='enhance_image'),
# ]

# if settings.DEBUG:
#     urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

