# Generated by Django 4.1.3 on 2023-06-17 19:20

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("cmssafex", "0002_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="contactfm",
            name="message",
            field=models.TextField(default="", max_length=1000),
        ),
    ]
