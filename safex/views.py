from django.http import HttpResponse 
from django.shortcuts import render, redirect
from django.db import models

def index(request):
    return render(request,"index.html")

def goals(request):
    return render(request,"goals.html")

def about(request):
    return render(request,"about.html")
    
def contact(request):
    return render(request,"contact.html")