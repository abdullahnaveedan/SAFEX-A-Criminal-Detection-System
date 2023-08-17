from django.contrib import admin
from .models import contactfm , staffinformation,criminalinformation,criminalrecord,detection

admin.site.register(contactfm)
admin.site.register(staffinformation)
admin.site.register(criminalinformation)
admin.site.register(criminalrecord)
admin.site.register(detection)