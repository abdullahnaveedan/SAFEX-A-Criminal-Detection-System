# Generated by Django 4.1.3 on 2023-07-08 17:19

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ("cmssafex", "0011_alter_criminalrecord_recordid"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="criminalinformation",
            name="leftimg",
        ),
        migrations.RemoveField(
            model_name="criminalinformation",
            name="rightimg",
        ),
    ]