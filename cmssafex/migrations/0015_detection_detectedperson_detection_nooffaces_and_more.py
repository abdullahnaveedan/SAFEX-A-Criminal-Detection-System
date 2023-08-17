# Generated by Django 4.1.3 on 2023-08-05 08:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("cmssafex", "0014_detection_datatype"),
    ]

    operations = [
        migrations.AddField(
            model_name="detection",
            name="detectedperson",
            field=models.CharField(default="", max_length=50),
        ),
        migrations.AddField(
            model_name="detection",
            name="nooffaces",
            field=models.CharField(default="NULL", max_length=50),
        ),
        migrations.AddField(
            model_name="detection",
            name="outputimage",
            field=models.ImageField(default="", upload_to="detected/"),
        ),
    ]