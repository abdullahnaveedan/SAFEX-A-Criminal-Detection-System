from django.db import models
from autoslug import AutoSlugField

class contactfm(models.Model):
    msgid = models.AutoField 
    useremail = models.CharField(max_length=50, default = "")
    subject = models.CharField(max_length=50,default="")
    message = models.TextField(max_length=1000,default="")

    def __str__(self):
        return self.useremail

class staffinformation(models.Model):
    staffid = models.AutoField
    beltno = models.CharField(primary_key = True,max_length = 10)
    fname = models.CharField(max_length=50 , default = "")
    lname = models.CharField(max_length=50 , default = "")
    username = models.CharField(max_length=50 , default = "")
    fathername = models.CharField(max_length=50 , default = "")
    email = models.CharField(max_length=50 , default = "")
    phoneno = models.CharField(max_length=50 , default = "")
    cnic = models.CharField(max_length=50 , default = "")
    rank = models.CharField(max_length=50 , default = "")
    city = models.CharField(max_length=50 , default = "")
    address = models.CharField(max_length=500)
    station = models.CharField(max_length=50 , default = "")
    division = models.CharField(max_length=50 , default = "")
    

    def __str__(self):
        return self.username

class criminalinformation(models.Model):
    criminalid = models.AutoField(primary_key = True,serialize=False, auto_created=True, default=300)
    name = models.CharField(max_length=50,default = "")
    fathername = models.CharField(max_length=50,default = "")
    cnic = models.CharField(max_length=50,default = "")
    dob = models.DateField(default = "0000-00-00")
    city = models.CharField( max_length=50 , default = "")
    province = models.CharField( max_length=50 , default = "")
    meritalst = models.CharField( max_length=50 , default = "")
    currentst = models.CharField( max_length=50 , default = "")
    frontimg = models.ImageField( upload_to="criminals/" , default = "")
    staff =  models.CharField( max_length=50 , default = "")

    def __str__(self):
        # show = str(self.criminalid) + 
        return str(self.criminalid) + str(' ') + self.name
    
class criminalrecord(models.Model):
    recordid = models.AutoField(primary_key =True)
    criminalid = models.ForeignKey(criminalinformation, on_delete=models.CASCADE)
    sublocality = models.CharField(default = "NULL", max_length=50)
    locality = models.CharField(default = "NULL", max_length=50)
    ctype = models.CharField(default = "NULL", max_length=50)
    detail = models.TextField(default = "NULL", max_length=50)
    city = models.CharField(default = "NULL", max_length=50)
    province = models.CharField(default = "NULL", max_length=50)
    staffid = models.CharField(default = "", max_length=50)

    def __str__(self):
        return str(self.criminalid)

class detection(models.Model):
    recordid = models.AutoField(primary_key =True)
    filename = models.CharField(default = "NULL", max_length=100)
    datatype = models.CharField(default = "Image", max_length=50)
    datafile = models.ImageField(upload_to="test/",default = "")
    outputvideo = models.FileField(default = "", upload_to="detected/")
    outputimage = models.ImageField(default = "", upload_to="detected/")
    nooffaces = models.CharField(default = "NULL", max_length=50)
    detectedperson = models.CharField(default = "", max_length=50)
    slug = AutoSlugField(populate_from='recordid', unique=True, null=True, default=None)
    def __str__(self):
        return str(self.recordid)
    
    