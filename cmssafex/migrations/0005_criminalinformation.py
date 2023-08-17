# Generated by Django 4.1.3 on 2023-06-19 17:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("cmssafex", "0004_staffinformation"),
    ]

    operations = [
        migrations.CreateModel(
            name="criminalinformation",
            fields=[
                (
                    "criminalid",
                    models.AutoField(
                        auto_created=True,
                        default=300,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("name", models.CharField(default="", max_length=50)),
                ("fathername", models.CharField(default="", max_length=50)),
                ("cnic", models.CharField(default="", max_length=50)),
                ("dob", models.DateField(default="0000-00-00")),
                ("city", models.CharField(default="", max_length=50)),
                ("meritalst", models.CharField(default="", max_length=50)),
                ("currentst", models.CharField(default="", max_length=50)),
                ("frontimg", models.ImageField(default="", upload_to="criminals/")),
                ("leftimg", models.ImageField(default="", upload_to="criminals/")),
                ("rightimg", models.ImageField(default="", upload_to="criminals/")),
                ("staff", models.CharField(default="", max_length=50)),
            ],
        ),
    ]
