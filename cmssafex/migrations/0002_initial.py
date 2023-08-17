# Generated by Django 4.1.3 on 2023-06-17 19:19

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ("cmssafex", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="contactfm",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("useremail", models.CharField(default="", max_length=50)),
                ("subject", models.CharField(default="", max_length=50)),
                ("message", models.CharField(default="", max_length=1000)),
            ],
        ),
    ]
