# Generated by Django 3.2.5 on 2025-03-11 16:15

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('generator', '0002_savedimage'),
    ]

    operations = [
        migrations.RenameField(
            model_name='savedimage',
            old_name='saved_at',
            new_name='created',
        ),
        migrations.RenameField(
            model_name='savedimage',
            old_name='image',
            new_name='enhanced_image',
        ),
        migrations.AddField(
            model_name='savedimage',
            name='description',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AlterField(
            model_name='savedimage',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
    ]
