# Generated by Django 3.2.5 on 2025-03-11 15:56

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('users', '0007_generatedimage'),
    ]

    operations = [
        migrations.CreateModel(
            name='SavedImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='saved_images/')),
                ('saved_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='saved_images_users', to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
