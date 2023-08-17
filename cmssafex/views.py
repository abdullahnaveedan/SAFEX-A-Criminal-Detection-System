from django.http import HttpResponse 
from django.shortcuts import render, redirect
from django.db import models
import os 
import cv2 as cv
import re
import uuid
from datetime import datetime
from django.contrib.auth import authenticate, login , logout
from django.contrib.auth.models import User
from .models import contactfm,staffinformation,criminalinformation,criminalrecord,detection
from django.db.models import Max
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import subprocess
import shutil
from django.contrib import messages

def cmssafex(request):
    return render(request,"login.html")

def signup(request):
    return redirect("cmssafex")

def loginvalidation(request):

    if request.method == "POST":
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request,username = username , password = password)
        if user:
            login(request,user)
            return redirect("dashboard")
        else:
            pass
    else:
        return redirect("cmssafex")
    return redirect("cmssafex")

def contactform(request):
    if request.method == 'POST':
        email = request.POST["email"]
        subject = request.POST["subject"]
        message = request.POST["message"]
        form = contactfm(useremail = email,subject = subject , message = message)
        form.save()
        message.success(request,'Send Sucessfully')
    return redirect("home")

def adduser(request):
    return render(request,"adduser.html")

def contact(request):
    if request.user.id == 1:
        contactdata = contactfm.objects.all()
        param = {'cont':contactdata}
        return render(request , "contacts.html",param)
    else:
        return HttpResponse("Page not found")

def newuser(request):
    if request.method == "POST":
        pass1 = request.POST["pass1"]
        password = request.POST["pass2"]
        pattern = r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]+$'
        
        username = request.POST["username"]
        nameuser = staffinformation.objects.filter(username = username)
        if len(nameuser) == 0:
            if re.match(pattern, password):
                if pass1 == password:
                    fname = request.POST["fname"]
                    beltno = request.POST["beltno"]
                    lname = request.POST["lname"]
                    email = request.POST["inputEmail"]
                    myuser = User.objects.create_user(username, email, password)
                    myuser.first_name = fname
                    myuser.last_name = lname
                    myuser.save()
                    newuser = staffinformation(fname = fname,beltno = beltno,lname = lname,username = username,email = email)
                    newuser.save()
                    return redirect("userinfo")
    return redirect("adduser")
    
def profileview(request , myid):
    user = User.objects.filter(id = myid)
    date = user[0].date_joined
    lastlogin = user[0].last_login
    staff = staffinformation.objects.filter(username = user[0].username)
    param = {'staff' : staff[0], 'users' : user[0], 'date' : date ,'lastlogin' : lastlogin }
    return render(request,"profileview.html",param)

def user_logout(request):
    logout(request)
    return redirect("cmssafex")

def dashboard(request):
    un = request.user.username
    user = User.objects.filter(username = un )
    staff = staffinformation.objects.filter(username = un)
    data1 = criminalinformation.objects.count()
    data2 = criminalinformation.objects.filter(currentst = "In Prision").count()
    data3 = criminalinformation.objects.filter(currentst = "Escaped").count()
    param = {'staff' : staff[0], 'users' : user[0], 'total': data1 , 'pri':data2 , 'esc':data3}
    return render(request,"dashboard.html",param)

def allcriminalrecord(request):
    data = criminalinformation.objects.all()
    param = {'data' : data}
    return render(request,"criminalpage.html",param)

def addcriminals(request):
    return render(request,"addcriminals.html")

def allrecord(request):
    data = criminalinformation.objects.all()
    param = {'data' : data}
    return render(request,"addrecord.html" , param)
    
def userinfo(request):
    if request.user.id == 1:
        users = User.objects.all()
        param = {'users':users}
        return render(request,"userinfo.html",param)
    else:
        return HttpResponse("Page not found")

def profile(request , myid):
    loginid = request.user.id

    if loginid == 1:
        user = User.objects.filter(id = myid)
        user = staffinformation.objects.filter(username = user[0].username)
        param = {'users' : user[0],'id' : myid}
        return render(request , "showrecord.html",param)
    else:
        users = User.objects.filter(id = loginid)
        if myid == loginid:
            users = staffinformation.objects.filter(username = users[0].username)
            param = {'users' : users[0],'id' : myid}
            return render(request , "showrecord.html",param)
        else:
            return HttpResponse("<h1>Page not found</h1>")

def update(request):
    if request.method == "POST":
        myid = request.POST["myid"]
        myuser = User.objects.filter(id=myid)
        if myuser.exists():
            uname = myuser[0].username
            staffinformation.objects.filter(username=uname).update(
                
                fname=request.POST["fname"],
                lname=request.POST["lname"],
                username=request.POST["username"],
                fathername=request.POST["fathername"],
                email=request.POST["inputEmail"],
                phoneno=request.POST["phoneno"],
                cnic=request.POST["cnic"],
                rank=request.POST["rank"],
                city=request.POST["city"],
                address=request.POST["address"],
                station=request.POST["station"],
                division=request.POST["division"]
            )

    return redirect("userinfo")

def newrecord(request):
    if request.method == "POST":
        name = request.POST.get('name','')
        fathername = request.POST.get('fname','')
        cnic = request.POST.get('cnic','')
        dob = request.POST.get('dob','')
        city = request.POST.get('city','')
        province = request.POST.get('province','')
        meritalst = request.POST.get('merriage','')
        currentst = request.POST.get('status','')
        staff = request.user.username
        
        image_file = request.FILES.get('front', None)
        if image_file:
            img = name + '.jpg'
            file_name = default_storage.save('criminals/' + img, ContentFile(image_file.read()))
            image = file_name
        else:
            image = ''
        if meritalst == "1":
            meritalst = "Single"
        elif meritalst == "2":
            meritalst = "Married"
        elif meritalst == "3":
            meritalst = "Divorced"
        else:
            meritalst = "NOT DEFINE"

        if currentst == "1":
            currentst = "Escaped"
        elif currentst == "2":
            currentst = "In Prision"
        else:
            currentst = "NOT DEFINE"
        
        if province == "0":
            province = "NOT DEFINE"
        elif province == "1":
            province = "Punjab"
        elif province == "2":
            province = "Sindh"
        elif province == "3":
            province = "Balochistan"
        elif province == "4":
            province = "Khyber Pakhtunkhwa"
        elif province == "5":
            province = "Kashmir"


        last_id = criminalinformation.objects.aggregate(Max('criminalid'))['criminalid__max']
        if last_id is not None:
            criminalid =  last_id + 1
       
        data = criminalinformation(criminalid, name , fathername , cnic , dob , city , province , meritalst , currentst ,image,staff)
        data.save()        
    return render(request , "addcriminals.html")

def editrecordcriminal(request , myid):
    data = criminalinformation.objects.filter(criminalid = myid)
    record = criminalrecord.objects.filter(criminalid = myid)
    param = {'users' : data[0], 'record' :record,'count':record.count()}
    return render(request , "editcrecord.html", param)

def caserecord(request):
    if request.method == "POST":
        criminalid_value = request.POST["criminalnumber"]
        criminalinfo_instance = criminalinformation.objects.get(criminalid=criminalid_value)
        sublocality = request.POST["sublocality"]
        locality = request.POST["locality"]
        ctype = request.POST["ctype"]
        detail = request.POST["detail"]
        city = request.POST["city"]
        province = request.POST["province"]
        staffid = request.user.username
        
        user = criminalrecord(criminalid=criminalinfo_instance, sublocality=sublocality, locality=locality, ctype=ctype, detail=detail, city=city, province=province, staffid=staffid)
        user.save()

    return redirect("allcriminal")
    
def search_view(request): 
    if 'id' in request.POST:
        id = request.POST['id']
        try:
            instance = criminalinformation.objects.get(criminalid = id)
            return render(request, 'criminalpage.html', {'ins': instance})
        except criminalinformation.DoesNotExist:
            count = 0
            return render(request, 'criminalpage.html', {'count': count})

def search_name(request): 
    if 'name' in request.POST:
        id = request.POST['name']
        try:
            instance = criminalinformation.objects.filter(name__icontains = id)
            print(instance)
            
            return render(request, 'criminalpage.html', {'instance': instance})
        except criminalinformation.DoesNotExist:
            count = 0
            return render(request, 'criminalpage.html', {'count': count})
    return redirect("allcriminal")

def search_city(request): 
    if 'city' in request.POST:
        id = request.POST['city']
        try:
            instance = criminalinformation.objects.get(cnic = id)
            return render(request, 'criminalpage.html', {'ins': instance})
        except criminalinformation.DoesNotExist:
            count = 0
            return render(request, 'criminalpage.html', {'count': count})
    return redirect("allcriminal")

def editcriminalrecord(request,myid):
    person = criminalinformation.objects.get(criminalid=myid)
    return render(request,"editcriminalpage.html",{'person':person})

def editcriminalrec(request):
    if request.method == "POST":
        myid = request.POST["id"]
        data = criminalinformation.objects.get(criminalid = myid)
        
        data.name = request.POST["fname"]
        data.fathername = request.POST["fthname"]
        data.cnic = request.POST["cnic"]
        data.city = request.POST["city"]
        data.province = request.POST["province"]
        data.meritalst = request.POST["ms"]
        data.currentst = request.POST["cs"]
        image_file = request.FILES.get('image', None)
        if image_file is not None:
            img = data.name + '.jpg'
            file_name = default_storage.save('criminals/' + img, ContentFile(image_file.read()))
            data.frontimg = file_name 

        data.save()
    return redirect("allcriminal")

def imagedetection(request):
    return render(request, "detimg.html")

def imgdetec(request):
    if request.method == "POST":
        image_file = request.FILES.get('front', None)
        data = detection.objects.all()
        
        param = {}
        recid = 0
        if image_file is not None:
            name = image_file.name
            file_name = default_storage.save('test/' + name, ContentFile(image_file.read()))
            image = file_name
            dt = "Image"
            last_id = detection.objects.aggregate(Max('recordid'))['recordid__max']
            if last_id is None:
                last_id = 0
            recid = last_id + 1
            print(recid)
            last_id = detection.objects.aggregate(Max('recordid'))['recordid__max']
            record = detection(filename = name ,datatype = dt, datafile = image)
            
            record.save()
            pathofdata = "media/test/" + name
            try:
                command = [
                    'python',
                    'cmssafex/yolo/detect.py',
                    '--weights',
                    'cmssafex/yolo/best.pt',
                    '--conf',
                    '0.25',
                    '--source',
                    pathofdata,
                    '--npz_file',
                    'cmssafex/yolo/data.npz'
                ]
                subprocess.run(command, check=True)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script: {e}")
            
            filename = "detecthistory.txt"
            arr=[]
            with open(filename,"r") as file:
                data = file.readlines()
            no_of_faces = str(data[0].strip())  
            face_info = eval(data[1].strip())
            path = data[2].strip()        
            arr=[]
            if len(face_info) == 0:
                arr.append("No Person found!")
            else:
                for face in face_info:
                    arr.append(face[0])
                    arr.append(face[1]*100)
            image_path = ""
            for x in path:
                if x == "\\":
                    image_path+= '/'
                else:
                    image_path+=x
            print(image_path)
            image_path =image_path
            source_image_path = image_path
            destination_image_path = os.path.join('media', 'detected')  
            shutil.copy(source_image_path, destination_image_path)
            output_img_name = os.path.basename(source_image_path)
            user = detection.objects.get(recordid = recid)

            user.nooffaces = no_of_faces
            user.detectedperson = arr
            user.outputimage = "detected/"+ output_img_name
            user.save()
            print("user is saved")
            records = detection.objects.get(recordid = recid)
            param={'rec':records,'count':1}
        else:
            print("Error")
            param={'count': 2}
    return render(request , "detimg.html",param)

def videodetect(request):
    if request.method == "POST":
        image_file = request.FILES.get('front', None)
        recid = 0
        if image_file is not None:
            param={}
            name = image_file.name
            file_name = default_storage.save('test/' + name, ContentFile(image_file.read()))
            image = file_name
            dt = "Video"
            last_id = detection.objects.aggregate(Max('recordid'))['recordid__max']
            if last_id is None:
                last_id = 0
            recid = last_id + 1
            print(recid)
            last_id = detection.objects.aggregate(Max('recordid'))['recordid__max']
            record = detection(filename = name ,datatype = dt, datafile = image)
            
            record.save()
            pathofdata = "media/test/" + name
            try:
                command = [
                    'python',
                    'cmssafex/yolo/detect.py',
                    '--weights',
                    'cmssafex/yolo/best.pt',
                    '--conf',
                    '0.25',
                    '--source',
                    pathofdata,
                    '--npz_file',
                    'cmssafex/yolo/data.npz'
                ]
                subprocess.run(command, check=True)
                print("Script executed successfully.")
            except subprocess.CalledProcessError as e:
                print("Error executing script: {e}")
            
            filename = "detecthistory.txt"
            arr=[]
            with open(filename,"r") as file:
                data = file.readlines()
            no_of_faces = str(data[0].strip())
            path = data[1].strip() 

            arr=[]
            image_path = ""
            for x in path:
                if x == "\\":
                    image_path+= '/'
                else:
                    image_path+=x
            
            source_image_path = image_path
            destination_image_path = os.path.join('media', 'detected')  
            shutil.copy(source_image_path, destination_image_path)
            output_img_name = os.path.basename(source_image_path)
            user = detection.objects.get(recordid = recid)
            user.nooffaces = no_of_faces
            user.outputvideo = "detected/"+ output_img_name
            file_name = default_storage.save('detected/' + output_img_name, ContentFile(image_file.read()))
            user.outputvideo = file_name
            user.save()
            records = detection.objects.get(recordid = recid)
            param = {'rec':records,'count':1}
        else:
            print("Error")
            param = {'count': 2}

    return render(request,"detvideo.html",param)
    
def videodetection(request):
    return render(request,"detvideo.html")

def cameradetect(request):
    dt = "Video"
    last_id = detection.objects.aggregate(Max('recordid'))['recordid__max']
    if last_id is None:
        last_id = 0
    recid = last_id + 1
    try:
        command = [
            'python',
            'cmssafex/yolo/detect.py',
            '--weights',
            'cmssafex/yolo/best.pt',
            '--conf',
            '0.25',
            '--source',
            '0',
            '--npz_file',
            'cmssafex/yolo/data.npz'
        ]
        subprocess.run(command, check=True)
        print("Script executed successfully.")
    except subprocess.CalledProcessError as e:
        print("Error executing script: {e}")
    
    filename = "detecthistory.txt"
    arr=[]
    with open(filename,"r") as file:
        data = file.readlines()
    no_of_faces = str(data[0].strip())
    path = data[1].strip()
    arr=[]
    image_path = ""
    for x in path:
        if x == "\\":
            image_path+= '/'
        else:
            image_path+=x
    
    source_image_path = image_path
    destination_image_path = os.path.join('media', 'detected')  
    shutil.copy(source_image_path, destination_image_path)
    videoname = os.path.basename(source_image_path)
    user = detection(
    nooffaces = no_of_faces,
    outputvideo = default_storage.save('detected/' + videoname, ContentFile(image_path)),
    datatype = "Camera")
    user.save()
    records = detection.objects.get(recordid = recid)
    param = {'rec':records,'count':1}
    
    return render(request,"detcamera.html",param)

def cameradetection(request):
    return render(request,"detcamera.html")

def localstoragebuffer():
    return 0


def deleterecordcriminal(request,myid):
    data = criminalinformation.objects.get(criminalid = myid)
    data.delete()
    return redirect("allcriminal")