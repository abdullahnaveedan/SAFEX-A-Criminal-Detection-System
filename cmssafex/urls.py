from django.contrib import admin
from django.urls import path,include
from .import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("",views.cmssafex,name="cmssafex"),
    path("contact",views.contactform,name="contactfm"),

    path("signup/",views.signup,name="signup"),
    path("loginvalidation/",views.loginvalidation,name="loginvalidation"),
    path("logout/",views.user_logout,name="logout"),
    path("contacts/",views.contact,name = "contact"),
    path("dashboard/",views.dashboard,name="dashboard"),

    path("newuser/",views.newuser,name="newuser"),
    path("userinfo/<int:myid>",views.profile,name="profile"),
    path("update/",views.update,name="update"),
    path("viewprofile/<int:myid>",views.profileview,name="profileview"),
    path("newrecord/",views.newrecord,name="newrecord"),
    path("allcriminals/<int:myid>" , views.editrecordcriminal, name="editrecordcriminal"),
    path("allcriminalrecord/",views.allcriminalrecord,name="allcriminal"),

    path("addcriminals/",views.addcriminals,name="addcriminals"),
    path("addcriminalrecord/",views.allrecord,name="addcriminalrecord"),
    path("userinfo/",views.userinfo,name="userinfo"),
    path("adduser/",views.adduser,name="adduser"),
    path("caserecord/", views.caserecord,name = "caserecord"),
    path("editcriminalrecord/<int:myid>", views.editcriminalrecord,name = "editcriminalrecord"),
    path("editcriminalrec/",views.editcriminalrec, name="editcriminalrec"),
    path("deleterecordcriminal/<int:myid>",views.deleterecordcriminal, name="deleterecordcriminal"),

    # Searches
    path("searchcid/", views.search_view, name="searchcid"),
    path("searchname/", views.search_name, name="searchcid"),
    path("searchcnic/", views.search_city, name="searchcid"),
    # Detection
    path("imagedetection/",views.imagedetection,name="imagedetect"),
    path("imgdetec/",views.imgdetec,name="imgdetec"),

    path("videodetection/",views.videodetection,name="videodetection"),
    path("videodetect/",views.videodetect,name="videodetect"),

    path("cameradetection/",views.cameradetection,name="cameradetection"),
    path("cameradetect/",views.cameradetect,name="cameradetection"),

] +  static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)