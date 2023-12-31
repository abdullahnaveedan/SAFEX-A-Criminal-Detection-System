# Generated by Django 4.1.3 on 2023-06-20 03:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("cmssafex", "0006_criminalinformation_province"),
    ]

    operations = [
        migrations.AlterField(
            model_name="criminalinformation",
            name="frontimg",
            field=models.ImageField(default="", upload_to="media/criminals/"),
        ),
        migrations.AlterField(
            model_name="criminalinformation",
            name="leftimg",
            field=models.ImageField(default="", upload_to="media/criminals/"),
        ),
        migrations.AlterField(
            model_name="criminalinformation",
            name="rightimg",
            field=models.ImageField(default="", upload_to="media/criminals/"),
        ),
    ]
