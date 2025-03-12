from django.db import models
from django.contrib.auth.models import User


# Create your models here.
class GeneratedImage(models.Model):
    description = models.TextField()
    image = models.ImageField(upload_to='generated_images/')
    enhanced_image = models.ImageField(upload_to='enhanced_images/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class SavedImage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    enhanced_image = models.ImageField(upload_to='saved_images/')  # Ensure this exists
    description = models.TextField(blank=True, null=True)
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Saved Image by {self.user.username}"