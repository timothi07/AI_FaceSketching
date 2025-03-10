
from django.db import models

class GeneratedImage(models.Model):
    description = models.TextField()
    image = models.ImageField(upload_to='generated_images/')
    enhanced_image = models.ImageField(upload_to='enhanced_images/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)